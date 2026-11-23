"""Candidate-facing objects: opaque point handles, the per-trial context `h` (counted classical oracle),
quantum runs, and the exact run sampler.

Exactness of the run sampler (see README "Why the sampler is exact"): a run prepares product states on its
exponent registers, applies controlled translations x -> x + sum_i c_i x_i on the point register (labels in
Z_r, c_i the discrete log of the point added under qubit i), discards the point register and measures the
exponent registers in the Fourier basis. Tracing out the point register equals measuring it in the Fourier
basis of Z_r, which makes every translation diagonal: the reduced state of the exponent registers is the
uniform mixture over u in Z_r of product states with qubit phases exp(2 pi i u c_i / r). So: draw u uniformly,
then draw each register's outcome from its semiclassical-QFT chain, whose conditional probabilities are
cos^2 of exactly reduced rationals. No approximation besides float64 rounding of each probability.
"""
import math, numbers, random

from cost import run_shape_cost


class _AttemptCap(Exception):
    """internal: a try_subset() attempt reached its work cap"""


class BudgetExceeded(Exception):
    pass


class HarnessError(Exception):
    pass


def _dbl_add_cost(k):
    """Group operations of plain double-and-add for scalar k >= 0."""
    return 0 if k <= 1 else (k.bit_length() - 1) + (bin(k).count('1') - 1)


class Point:
    """Opaque handle to an element of the prime-order group <G>. All arithmetic is charged to the trial.
    _q marks points built from Q (their windows' tables depend on the instance; see Run.measure)."""
    __slots__ = ('_h', '_l', '_q')

    def __init__(self, h, label, q=False):
        self._h, self._l, self._q = h, label % h.r, q

    def __add__(self, other):
        self._h._charge('group', 1)
        return Point(self._h, self._l + other._l, self._q or other._q)

    def __sub__(self, other):
        self._h._charge('group', 1)
        return Point(self._h, self._l - other._l, self._q or other._q)

    def __neg__(self):
        return Point(self._h, -self._l, self._q)

    def __mul__(self, k):
        if not isinstance(k, numbers.Integral):
            raise TypeError('scalar must be an integer')
        r = self._h.r
        k = int(k) % r
        self._h._charge('group', _dbl_add_cost(min(k, r - k)))
        return Point(self._h, k * self._l, self._q)

    __rmul__ = __mul__

    def __eq__(self, other):
        self._h._charge('cmp', 1)
        return isinstance(other, Point) and self._l == other._l

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self._h._salt, self._l))

    def endo(self):
        """phi(x, y) = (beta x, y) = [lam] P on j = 0 curves; one field multiplication."""
        if self._h.lam is None:
            raise HarnessError('no endomorphism on this curve')
        self._h._charge('field', 1)
        return Point(self._h, self._h.lam * self._l, self._q)

    def is_zero(self):
        return self._l == 0

    def __repr__(self):
        return '<Point>'


class Reg:
    __slots__ = ('index', 'bits', 'run')

    def __init__(self, index, bits, run):
        self.index, self.bits, self.run = index, bits, run


class HexReg:
    __slots__ = ('index', 'Ws', 'run')

    def __init__(self, index, Ws, run):
        self.index, self.Ws, self.run = index, Ws, run


class Run:
    """One coherent quantum run at the group-operation level (see README for the primitive set)."""

    def __init__(self, h):
        self._h = h
        self._regs = []        # per register: list of per-qubit labels c_i (mod r)
        self._adds = []        # (register index, lo, hi) per add() call, for the cost model
        self._add_q = []       # per add() call: is the added point built from Q (instance-dependent tables)?
        self._start_zero = True
        self._done = False
        self._hash = None
        self._amps = {}        # register index -> (normalised amplitude vector, preparation Toffoli)
        self._prep_declared = False   # some amplitude register has a declared (not counted) preparation
        self._offset = False   # random offset folded into the first lookup (offset())
        self._win_offsets = False  # a fresh random offset in every later window too (offset(per_window=True))
        self._global = False   # one uniform scalar c for every table of the run (offset(global_scalar=True))
        self.flagged = False   # visible flag of a Lemma G run (paper Sec. V.D and App. B): decoders drop it (Theorem G)
        self.id = None         # index of the run in h.runs, set by measure()
        self._hex = []         # hex registers: dict(Ws, c, kinds, q) (hexreg / hexadd)
        self._order = []       # ('std' | 'hex', index) per register, in creation order (outcome order)

    def reg(self, bits, amplitudes=None, prep_toffoli=0.0, sine=None):
        """New exponent register of `bits` qubits in uniform superposition over [0, 2^bits), or with
        `amplitudes` (any 2^k complex numbers, normalised here; k <= bits, k <= 24) prepared at your declared
        Toffoli cost, or with sine=k: the amplitudes sin(pi (y + 1/2) / 2^k) at the counted cost of
        harness/sineprep.py (3k - 2 Toffoli, checked in tests/test_sineprep.py; the grade then carries no
        declared preparation). k = bits gives the state sum_x amplitudes[x] |x>; k < bits is an envelope register, its top
        k qubits in sum_y amplitudes[y] |y> and the low bits - k qubits uniform, i.e. amplitude
        amplitudes[x >> (bits - k)] / sqrt(2^(bits - k)) on x. The k shaped qubits are not a product state, so
        they cannot be recycled by the semiclassical QFT: they count toward the run's peak, except those covered by
        an add() call that lies in the run's first window (the exact first lookup) and reaches the register's top
        bit (the QFT measures them right after that lookup; cost.shaped_held)."""
        if self._done or bits < 1:
            raise HarnessError('bad register')
        if sine is not None:
            import sineprep
            if amplitudes is not None or not isinstance(sine, numbers.Integral) or not 1 <= sine <= min(bits, 24):
                raise HarnessError('sine=k needs 1 <= k <= min(bits, 24) and no amplitudes')
            amplitudes, prep_toffoli = sineprep.amplitudes(int(sine)), sineprep.toffoli(int(sine))
        elif amplitudes is not None:
            self._prep_declared = True
        if amplitudes is not None:
            import numpy as np
            a = np.asarray(amplitudes, dtype=np.complex128).ravel()
            k = a.size.bit_length() - 1
            if a.size != 1 << k or not 1 <= k <= min(bits, 24) or not np.isfinite(a).all() or not (np.abs(a) > 0).any():
                raise HarnessError('amplitudes must be 2^k finite numbers, 1 <= k <= min(bits, 24), not all zero')
            self._amps[len(self._regs)] = (a / np.linalg.norm(a), float(prep_toffoli))
        self._regs.append([0] * bits)
        self._order.append(('std', len(self._regs) - 1))
        return Reg(len(self._regs) - 1, bits, self)

    def hexreg(self, W, m=None, top=None, split=0):
        """Hex (unit-orbit) exponent register for j = 0 curves (harness/hexreg.py, construction): windows of
        W-bit-pair digits, m of them, or W a list of window sizes (lowest window first; sizes 1..10). hexadd() adds
        [s1 + s2 lam] P with s1 + s2 w = sum_i digit_i 2^(W_0 + ... + W_(i-1)); measure() returns the register's
        outcome as a pair (Y1, Y2) in [0, 2^(sum W))^2, peaked at Y / 2^(sum W) = (u c / r, u c lam / r) mod 1.
        top='sine': the top window holds box digits s + t w (s, t < 2^W) in the product of two sine states (two
        counted preparations of harness/sineprep.py, 2 (3W - 2) Toffoli), which removes most outliers of the
        decoded u at small sigma (h1/hex_outliers.py); it must be read by the run's first lookup ('first').
        split = P1 - P2 in {-1, 0, 1} (top='sine' only): the top box has one more bit on s1 (split = 1) or s2, so the
        Y2 < 2^(sum W + max(-split, 0))."""
        Ws = [W] * m if m is not None else list(W)
        if self._done or not Ws or not all(isinstance(w, numbers.Integral) and 1 <= w <= 10 for w in Ws) or \
                top not in (None, 'sine') or split not in (-1, 0, 1) or (split and top != 'sine'):
            raise HarnessError('bad hex register')
        if self._h.lam is None:
            raise HarnessError('hex registers need the j = 0 endomorphism (h.lam)')
        self._hex.append(dict(Ws=[int(w) for w in Ws], c=0, kinds=None, q=False, top=top, split=int(split)))
        self._order.append(('hex', len(self._hex) - 1))
        return HexReg(len(self._hex) - 1, list(Ws), self)

    def mixreg(self, spec):
        """Mixed-radix Eisenstein exponent register for j = 0 curves (harness/mixreg.py, construction in
        o1/mixrad.py): windows, lowest first, 'hW' (hex, ideal norm 4^W, W <= 10), 'nw' / 'mw' (pi = (3 + w) 2^w /
        (2 - w) 2^w, norm 7 x 4^w, w <= 8) and an optional sine box top 'sW' (last; read by the first lookup, kind
        'first'), with 'x k' repeating a token, e.g. 'h9,n7,m7x10,h8x3,s4'. hexadd() adds [phi(s)] P, s = sum_i Pi_i d_i
        (d_i the digit of window i's residue, Pi_i = pi_0 ... pi_(i-1)); measure() returns (Y1, Y2) = N eta in
        [0, N)^2, N = norm(Pi_m) (mixreg.register_norm), one character eta of Z[w] / (Pi_m), sampled exactly, peaked at
        (u c / r, u c lam / r) mod 1. Window labels for kinds and --global-mc: W for hex windows, 'nw' / 'mw' for norm-7
        windows (V-B KIND n7_window_add / m7_window_add)."""
        import mixreg
        if self._done:
            raise HarnessError('run already measured')
        if self._h.lam is None:
            raise HarnessError('mixed-radix registers need the j = 0 endomorphism (h.lam)')
        try:
            wins = mixreg.cached(spec)
        except (ValueError, KeyError, IndexError) as ex:
            raise HarnessError('bad mixed-radix layout %r: %s' % (spec, ex))
        labs = [mixreg.label(w) for w in wins]
        if not wins or any(w.kind == 'sine' for w in wins[:-1]) or not all(mixreg.check_label(x) for x in labs):
            raise HarnessError("bad mixed-radix layout %r: windows hW (W <= 10), nw / mw (w <= 8), a sine top 'sW' "
                               "only last" % spec)
        top = 'sine' if wins[-1].kind == 'sine' else None
        self._hex.append(dict(Ws=labs, c=0, kinds=None, q=False, top=top, split=0, mix=wins))
        self._order.append(('hex', len(self._hex) - 1))
        return HexReg(len(self._hex) - 1, list(labs), self)

    def hexadd(self, reg, P, kinds):
        """Accumulator += [s1 + s2 lam] P over every window of the hex register reg (one call per register).
        kinds: per window (lowest first), or one name for all, the V-B-checked circuit that does the window's
        addition (run.py --kind NAME=FILE, a KIND hex_window_add result whose window equals the window's size);
        'first' on the top window puts it into the run's first lookup, which a run from the identity uses to load
        every standard register and these windows at once (priced by run.py --first-lookup or by formula)."""
        if self._done:
            raise HarnessError('run already measured')
        if not isinstance(reg, HexReg) or reg.run is not self:
            raise HarnessError('hexadd needs a hex register of this run')
        e = self._hex[reg.index]
        if e['kinds'] is not None:
            raise HarnessError('one hexadd per hex register')
        ks = [kinds] * len(e['Ws']) if isinstance(kinds, str) else list(kinds)
        if len(ks) != len(e['Ws']) or not all(isinstance(k, str) for k in ks) or 'first' in ks[:-1]:
            raise HarnessError("kinds: one name per window, 'first' only on the top window")
        if e['top'] and ks[-1] != 'first':
            raise HarnessError("a shaped top window (top='sine') is read by the first lookup: kind 'first'")
        e.update(c=P._l, kinds=ks, q=P._q)

    def _shaped(self, i):
        """Number of shaped (top) qubits of register i: k for 2^k amplitudes, 0 for a uniform register."""
        return self._amps[i][0].size.bit_length() - 1 if i in self._amps else 0

    def _shaped_weight(self, windows):
        """Sum over the approximate windows (all but the first lookup) of the product of max|phi| / RMS(phi) over
        the amplitude registers whose shaped qubits the window reads (1 for a window that reads none). A call
        reading shaped qubits marks every window it touches (its bit order within a window is not modelled)."""
        from cost import MODEL, call_windows
        fac = {}
        for (i, lo, hi), ws in zip(self._adds, call_windows([hi - lo for _, lo, hi in self._adds], MODEL)):
            if i in self._amps and hi > len(self._regs[i]) - self._shaped(i):
                a = self._amps[i][0]
                for j in ws:
                    fac.setdefault(j, {})[i] = float(abs(a).max()) * math.sqrt(a.size)
        first = 1 if self._start_zero and windows else 0
        return sum(math.prod(fac.get(j, {}).values()) for j in range(first, windows))

    def start(self, P):
        """Initialise the accumulator to the known point P instead of the identity."""
        if self._adds:
            raise HarnessError('start() must precede add()')
        self._start_zero = P._l == 0

    def offset(self, per_window=False, global_scalar=False):
        """Random offset: the first window is loaded by a direct lookup of T0[k] + R0 for a fresh uniformly
        random point R0 (a full unsigned 2^w-entry table, cost.MODEL['first_lookup_offset']). The accumulator then
        holds R0 + (ideal partial sum) before every later addition, which is uniform over the group for every
        exponent value: each addition's input law is the one V-B's run-input Monte Carlo samples (uniform
        accumulator). R0 only relabels the discarded point register, so the outcome law is unchanged. Required
        for the coupling model with a measured failure rate (README "kappa model").
        T_w[k] + R_w with its own fresh uniform R_w. Every addition's operand pair is then uniform for every
        exponent value, whatever the tables are (Q-built ones included), so every input misbehaves with the same
        probability and flagged failures cost runs, not the 1/kappa factor (README "kappa model", per-window
        offsets). The harness charges the per-run tables (2^w group operations per approximate window); the
        circuit must read unsigned tables (V-B KIND windowed_point_add, else estimate grade)."""
        if self._adds or not self._start_zero:
            raise HarnessError('offset() must precede add() and needs an accumulator starting at the identity')
        self._offset = True
        self._win_offsets = bool(per_window)
        # global_scalar=True (paper Sec. V.D and App. B; , Lemma G; hex runs): every table of the run, the first lookup's included, is
        # built for [c] G with a fresh uniform c, so every addition sees a uniform operand pair on every input's path;
        # the outcome law does not change (c relabels the discarded accumulator's character). With run.py
        # --global-mc the run is kept by Lemma G's rule (README "Lemma G runs"); the tables are charged per run.
        self._global = bool(global_scalar)
        if self._global and self._win_offsets:
            raise HarnessError('global_scalar replaces per-window offsets')

    def add(self, reg, P, lo=0, hi=None):
        """Accumulator += (value of qubits lo..hi-1 of reg) * P, i.e. qubit i adds [2^(i-lo)] P (windowed)."""
        if self._done:
            raise HarnessError('run already measured')
        if not isinstance(reg, Reg) or reg.run is not self:
            raise HarnessError('register belongs to another run')
        hi = reg.bits if hi is None else hi
        if not 0 <= lo < hi <= reg.bits:
            raise HarnessError('bad qubit range')
        r, c, labs = self._h.r, P._l, self._regs[reg.index]
        for i in range(lo, hi):
            labs[i] = (labs[i] + (c << (i - lo))) % r
        self._adds.append((reg.index, lo, hi))
        self._add_q.append(P._q)

    def hash(self, fn, toffoli, qubits=0):
        """After the additions: compute fn(accumulator) into a fresh register, then uncompute the accumulator
        (its additions are charged a second time); the exponent registers stay coherent until the QFT (no
        qubit recycling). measure() then returns the register outcomes followed by fn's value.
        fn: 'legendre_x' (1 if x is a quadratic non-residue, else 0; 0 at the identity), 'x_low:k', 'y_low:k'
        (k <= 8 low bits of the coordinate; 0 at the identity). toffoli, qubits: your declared cost of
        evaluating fn reversibly at this size (grade 'declared' until V-B measures it).
        Simulated exactly on the full register grid for n <= GRID_MAX_BITS (see README)."""
        if self._done or self._hash is not None:
            raise HarnessError('hash() once, before measure()')
        _hash_fn(fn)
        self._hash = (fn, float(toffoli), int(qubits))

    def measure(self):
        """Fourier-measure every register (semiclassical QFT); returns the outcomes in register order
        (plus the hash value for hash runs)."""
        if self._done:
            raise HarnessError('run already measured')
        if self._hex:
            return self._measure_hex()
        if self._global:
            raise HarnessError('offset(global_scalar=True) is for hex runs (Lemma G); a binary run uses offset()')
        self._done = True
        h = self._h
        top = [len(x) - self._shaped(i) for i, x in enumerate(self._regs)]   # first shaped qubit per register
        shape = dict(reg_bits=[len(x) for x in self._regs], add_bits=[hi - lo for _, lo, hi in self._adds],
                     # add() calls whose first-window controls need not stay live (cost.first_window_held): the
                     # call reaches its register's top bit, or it reads only shaped qubits of an amplitude register
                     # (cost.shaped_held counts those)
                     add_top=[hi == len(self._regs[i]) or lo >= top[i] for i, lo, hi in self._adds],
                     add_q=list(self._add_q),
                     start_zero=self._start_zero, offset=self._offset, window_offsets=self._win_offsets)
        if self._amps:
            # shaped qubits in total, and per add() call the shaped qubits it covers if it reaches the top bit
            shape.update(prep_toffoli=sum(t for _, t in self._amps.values()), prep_declared=self._prep_declared,
                         amp_bits=sum(self._shaped(i) for i in self._amps),
                         add_free=[hi - max(lo, top[i]) if i in self._amps and hi == len(self._regs[i]) else 0
                                   for i, lo, hi in self._adds])
        if self._hash:
            shape.update(hash=self._hash[0], hash_toffoli=self._hash[1], hash_qubits=self._hash[2])
        cost = run_shape_cost(shape, h.n)
        h._record_run(shape, cost)
        self.id = len(h.runs) - 1
        rng = h._qrng
        if self._win_offsets and cost['additions']:
            from cost import MODEL
            h._charge('group', cost['additions'] * 2 ** MODEL['w'])     # the run's tables T_w[k] + R_w
        eps = min(1.0, h.fail_per_add * cost['additions'])
        star = None             # (pi, e_S) of the run under Lemma 1* (per-window offsets)
        if eps and self._win_offsets and h.fail_split:
            star = tuple(min(1.0, x * cost['additions']) for x in h.fail_split)
        if eps and (self._hash or not h.kappa or not self._offset or
                    (star is None if self._win_offsets else cost['q_tables'])):
            # unflagged arithmetic, runs without the random offset (the measured failure rate assumes a uniform
            # accumulator), runs whose approximate additions read tables built from Q (their failure weight
            # (fail_split) or shapes without a coherence bound: the old independent-garbage-run model
            # (Häner's 4 pf per addition; heuristic only)
            h.heuristic_runs += 1
            if rng.random() < min(1.0, 4 * eps):
                h.garbage_runs += 1
                h._garbage.add(self.id)
                out = [rng.getrandbits(len(x)) for x in self._regs]
                return out + [0] if self._hash else out
            eps = 0.0
        if self._hash:
            if self._amps:
                raise HarnessError('hash runs with amplitude registers are not supported')
            return grid_sample(h, self._regs, self._hash[0], rng)
        u = rng.randrange(h.r)
        out, amp_p = [], {}
        for i, labs in enumerate(self._regs):
            if i in self._amps:
                j, amp_p[i] = sample_register_amps(rng, labs, u, h.r, self._amps[i][0], with_prob=True)
            else:
                j = sample_register(rng, labs, u, h.r)
            out.append(j)
        if eps:
            # run's expected bad weight E[|X_f| + |X_d| + 2 |X_s|] / N (flagged, dirty-ancilla and silent inputs,
            # silent ones counted twice). The joint probability of "no flag and outcome o" given u is at least
            # (kappa - e)^2 for the run's own bad weight e, kappa = sqrt(P(o | u)); averaged over the random offsets
            # (convexity) at least P(o | u) (1 - eps / kappa)^2. Coupling: keep the exact outcome with that factor,
            # else the run is bad (flagged runs included, no rerun, no normalisation): its outcome is uniform here and
            # try_subset() treats it as unusable (worst case for decoders that only use verified answers).
            # Amplitude registers (README "kappa model", amplitude registers): input x weighs |c_x| / sqrt(N) instead
            # of 1 / N, so an addition whose window reads shaped qubits of an amplitude register with amplitudes phi
            # gets at most max|phi| / RMS(phi) times its measured rate (the product over such registers), and any
            # other addition at most its rate. kappa^2 is the product of the registers' exact outcome probabilities.
            # probability <= pi and the misbehaving inputs that end unflagged weigh <= e_S, so
            # P(no flag, o | u) >= ((1 - pi) kappa - e_S)^2: keep factor ((1 - pi) - e_S / kappa)^2, with no
            # envelope factor and no restriction on the tables.
            if self._amps and star is None:
                eps = min(1.0, h.fail_per_add * self._shaped_weight(cost['windows']))
            pr = 1.0
            for i, (labs, j) in enumerate(zip(self._regs, out)):
                pr *= amp_p[i] if i in amp_p else register_prob(labs, u, h.r, j)
            kap = math.sqrt(pr)
            if star is None:
                keep = max(0.0, 1 - eps / kap) ** 2 if kap > 0 else 0.0
            else:
                keep = max(0.0, (1 - star[0]) - star[1] / kap) ** 2 if kap > 0 else 0.0
            if rng.random() >= keep:
                h.garbage_runs += 1
                h._garbage.add(self.id)
                return [rng.getrandbits(len(x)) for x in self._regs]
        return out


def _measure_hex(self):
    """measure() of a run with hex registers (cost.hex_run_cost prices it window by window, one V-B kind per
    window). Standard registers must be loaded whole by the first lookup; the coupling model is the one of
    Run.measure (eps = the run's summed failure weight, keep factor (1 - eps / kappa)^2, kappa^2 = the product of
    the registers' exact outcome probabilities); no envelope factor is needed because no approximate window reads a
    standard register."""
    import cost as _cost, hexreg
    self._done = True
    h = self._h
    if self._hash or self._win_offsets:
        raise HarnessError('hex runs: no hash outputs and no per-window offsets')
    if any(e['kinds'] is None for e in self._hex):
        raise HarnessError('every hex register needs its hexadd before measure()')
    if any(lo or hi != len(self._regs[i]) for i, lo, hi in self._adds):
        raise HarnessError('in a hex run, standard registers are added whole (they belong to the first lookup)')
    shape = dict(reg_bits=[len(x) for x in self._regs] + [_hex_qubits(e) for e in self._hex],
                 add_bits=[hi - lo for _, lo, hi in self._adds], add_q=list(self._add_q),
                 hex=[dict(Ws=e['Ws'], kinds=e['kinds'], q=e['q'], split=e['split']) for e in self._hex],
                 start_zero=self._start_zero, offset=self._offset)
    tops = [e for e in self._hex if e['top'] == 'sine']
    if self._amps or tops:
        import sineprep
        shape.update(prep_toffoli=sum(t for _, t in self._amps.values()) +
                     sum(sineprep.toffoli(e['Ws'][-1] + max(e['split'], 0)) + sineprep.toffoli(e['Ws'][-1] + max(
                         -e['split'], 0)) for e in tops), prep_declared=self._prep_declared)
    cost = _cost.hex_run_cost(shape, h.n)
    h._record_run(shape, cost)
    self.id = len(h.runs) - 1
    rng = h._qrng
    lemma_g = self._global and h.approx and h.global_mc is not None and not cost['q_tables']
    if self._global:
        h._charge('group', 2 * cost['table_entries'])     # [c] T for every table entry, 2 group ops each (declared)
    if cost['term_classical']:
        h._charge('other', cost['term_classical'])
    if lemma_g and h.global_mc.get('run_cap') is not None:
        # run-level abort cap (cost.global_mc_constants): the streams are charged run_cap in all instead of their
        # per-addition caps (the per-addition match below ties sum(stream_caps) to this run's kinds); every
        # non-stream Toffoli, the digit registers, the terminal's tail and the first lookup stay charged in full
        if 'run_cap' in cost:
            raise HarnessError('run cap applied twice (an old run-cap wrapper on top of the harness?)')
        g = h.global_mc
        cost['run_cap_saving'] = sum(g['stream_caps']) - g['run_cap']
        cost.update(run_cap=g['run_cap'], toffoli=cost['toffoli'] - cost['run_cap_saving'])
        cost['cost'] = cost['qubits'] * cost['toffoli']
    eps = 0.0 if lemma_g else (min(1.0, cost['fail_weight']) if h.approx else 0.0)

    def garbage():               # a bad or visibly flagged run: try_subset() fails on subsets that hold it
        h.garbage_runs += 1
        h._garbage.add(self.id)
        return [rng.getrandbits(len(self._regs[i])) if t == 'std' else _hex_garbage(rng, self._hex[i])
                for t, i in self._order]
    if eps and (not h.kappa or not self._offset or cost['q_tables']):
        h.heuristic_runs += 1                       # as Run.measure: the old garbage-run model, heuristic grade
        if rng.random() < min(1.0, 4 * eps):
            return garbage()
        eps = 0.0
    u = rng.randrange(h.r)
    out, pr = [], 1.0
    for t, i in self._order:
        if t == 'hex':
            e = self._hex[i]
            if 'mix' in e:
                import mixreg
                o, p = mixreg.sample_Y(rng, e['mix'], e['c'], u, h.r, h.lam)
            else:
                o, p = hexreg.sample(rng, e['Ws'], e['c'], u, h.r, h.lam, e['top'], e['split'])
        elif i in self._amps:
            o, p = sample_register_amps(rng, self._regs[i], u, h.r, self._amps[i][0], with_prob=True)
        else:
            o = sample_register(rng, self._regs[i], u, h.r)
            p = register_prob(self._regs[i], u, h.r, o) if (eps or lemma_g) else 1.0
        out.append(o)
        pr *= p
    if eps:
        kap = math.sqrt(pr)
        keep = max(0.0, 1 - eps / kap) ** 2 if kap > 0 else 0.0
        if rng.random() >= keep:
            return garbage()
    if lemma_g:
        # paper Sec. V.D Theorem G: a visible flag with probability phi_LB (whatever the outcome); otherwise the exact
        # outcome is kept with probability K / (1 - phi_LB), K = (g0 - e0 / kappa)_+^2, else the run is an unflagged
        # bad run. Then P(F) = phi_LB and P(G, o) = P_I(o) K(o), the worst case Theorem G allows.
        g = h.global_mc
        if cost['add_ws'] != g['windows']:
            raise HarnessError('--global-mc covers runs whose %d additions have windows %s (lowest first), this run '
                               'has %d with %s' % (g['J'], g['windows'], cost['additions'], cost['add_ws']))
        if g.get('stream_caps') is not None:     # caps listed per addition (a terminal replayed with its own cap):
            for i, k in enumerate(cost['add_kinds']):     # addition i must be V-B'd at exactly entry i's cap
                m = _cost.KINDS[k]
                pair = (m.get('stream_cap'), m.get('stream_toffoli'))
                if m.get('abort_cap') is None or pair != (g['stream_caps'][i], g['stream_toffolis'][i]):
                    raise HarnessError('addition %d (kind %r, abort cap %s, stream cap %s of %s): the path Monte Carlo '
                                       'replayed it at stream cap %s of %s' % (
                                           i, k, m.get('abort_cap'), pair[0], pair[1], g['stream_caps'][i],
                                           g['stream_toffolis'][i]))
        else:
            for k in cost['kinds']:              # a kind charged at an abort cap: the Monte Carlo counted its aborts
                m = _cost.KINDS[k]
                if m.get('abort_cap') is not None and (m.get('stream_cap') is None or (
                        m['stream_cap'], m['stream_toffoli']) != (g.get('stream_cap'), g.get('stream_toffoli'))):
                    raise HarnessError('kind %r is charged at an abort cap (stream cap %s of %s), the path Monte Carlo '
                                       'counted aborts at %s of %s' % (k, m.get('stream_cap'), m.get('stream_toffoli'),
                                                                       g.get('stream_cap'), g.get('stream_toffoli')))
        g0 = 1 - g['eS'] - cost['additions'] * 8 / h.r     # eS = UB E[N_DO + 2 N_P] per path (cost.global_mc_constants)
        e0 = self._lemma_g_weight() * g['e0_paths']
        if g0 * g0 > 1 - g['phi']:
            raise HarnessError('Lemma G keep rule needs g0^2 <= 1 - phi_LB')
        h.lemma_g_runs += 1
        if rng.random() < g['phi']:
            self.flagged = True
            return garbage()
        kap = math.sqrt(pr)
        K = max(0.0, g0 - e0 / kap) ** 2 if kap > 0 else 0.0
        if rng.random() >= K / (1 - g['phi']):
            return garbage()
    return out


def _hex_qubits(e):
    """register qubits of a hex (Run.hexreg) or mixed-radix (Run.mixreg) register"""
    if 'mix' in e:
        import mixreg
        return sum(mixreg.win_qubits(x) for x in e['Ws'])
    return 2 * sum(e['Ws']) + abs(e['split'])


def _hex_garbage(rng, e):
    """outcome of a bad or flagged run for a hex or mixed-radix register (any value of the outcome's range)"""
    if 'mix' in e:
        import mixreg
        N = mixreg.register_norm(e['mix'])
        return (rng.randrange(N), rng.randrange(N))
    return (rng.getrandbits(sum(e['Ws']) + max(e['split'], 0)), rng.getrandbits(sum(e['Ws']) + max(-e['split'], 0)))


def _lemma_g_weight(self):
    """Lambda of paper Sec. V.D (Section 4, item 3): the weight of e0's average sum_x |a_x| 2^(-n/2) F(x). In a hex run
    no approximate window reads a shaped register (standard registers and a top='sine' window belong to the exact
    first lookup), so each shaped register contributes its L1 factor sum |a| / 2^(k/2) <= 1 (a normalised) and
    uniform registers 1."""
    import hexreg, numpy as np
    lam = 1.0
    for a, _ in self._amps.values():
        lam *= float(np.abs(a).sum()) / math.sqrt(a.size)
    for e in self._hex:
        if e['top'] == 'sine':
            for a in (hexreg.sine(e['Ws'][-1] + max(e['split'], 0)), hexreg.sine(e['Ws'][-1] + max(-e['split'], 0))):
                lam *= float(a.sum()) / math.sqrt(a.size * float((a * a).sum()))
    return lam


Run._measure_hex = _measure_hex
Run._lemma_g_weight = _lemma_g_weight


def sample_register(rng, labs, u, r):
    """Outcome j of an l-qubit register whose qubit i carries phase theta_i = u labs[i] / r, measured by the
    QFT: P(j) = prod_i cos^2(pi (theta_i - j 2^i / 2^l)). Sampled bit by bit (y_0 = least significant
    output bit first, conditioned on the qubit of weight 2^(l-1)), each conditional exactly normalised."""
    l = len(labs)
    j = 0
    for t in range(l):
        num = u * labs[l - 1 - t] % r
        phi = num / r - j / (1 << (t + 1))     # correctly rounded big-int divisions
        phi -= round(phi)
        p0 = math.cos(math.pi * phi) ** 2
        if rng.random() >= p0:
            j |= 1 << t
    return j


def register_prob(labs, u, r, j):
    """P(j | u) of sample_register's outcome j (the same product of cos^2 factors)."""
    l = len(labs)
    pr = 1.0
    for t in range(l):
        phi = (u * labs[l - 1 - t] % r) / r - (j & ((1 << t) - 1)) / (1 << (t + 1))
        phi -= round(phi)
        p0 = math.cos(math.pi * phi) ** 2
        pr *= (1 - p0) if (j >> t) & 1 else p0
    return pr


GRID_MAX_BITS = 20        # hash runs: curve size limit (point table of r entries)
GRID_MAX_CELLS = 1 << 24  # hash runs: product of register sizes (one complex FFT of this size per run)
_POINT_TABLES = {}


def sample_register_amps(rng, labs, u, r, amps, with_prob=False):
    """Same as sample_register for a register with 2^k amplitudes (Run.reg): given u, qubit i carries the phase
    theta_i = u labs[i] / r. The top k qubits (shaped) hold amps[y] exp(2 pi i sum_i y_i theta_(l-k+i)), the low
    L = l - k qubits are uniform. The QFT kernel exp(-2 pi i x j / 2^l) splits x = y 2^L + z: the top factor
    depends only on j mod 2^k (one FFT of 2^k), the low qubits give the cos^2 chain of sample_register, so
    P(j) = |FFT(...)[j mod 2^k]|^2 / 2^k * prod_(i < L) cos^2(pi (theta_i - j 2^i / 2^l)) (exact). Output bits
    0..k-1 are drawn from the FFT, then bits k..l-1 one at a time, each conditional exactly normalised.
    with_prob=True returns (j, P(j)) with the same random draws."""
    import numpy as np
    l, k = len(labs), amps.size.bit_length() - 1
    L = l - k
    y = np.arange(1 << k, dtype=np.int64)
    ph = np.zeros(1 << k)
    for i in range(k):
        ph += ((y >> i) & 1) * ((u * labs[L + i] % r) / r)
    # kernel exp(-2 pi i y j / 2^k), so outcomes peak at j / 2^l = +theta as in sample_register
    p = np.abs(np.fft.fft(amps * np.exp(2j * np.pi * (ph - np.round(ph))))) ** 2
    cdf = np.cumsum(p)
    j = min(cdf.size - 1, int(np.searchsorted(cdf, rng.random() * cdf[-1], side='right')))
    pr = float(p[j] / cdf[-1])
    for t in range(k, l):
        phi = (u * labs[l - 1 - t] % r) / r - j / (1 << (t + 1))
        phi -= round(phi)
        p0 = math.cos(math.pi * phi) ** 2
        if rng.random() >= p0:
            j |= 1 << t
            pr *= 1 - p0
        else:
            pr *= p0
    return (j, pr) if with_prob else j


def _hash_fn(fn):
    """Vectorised map (x array, y array, identity mask) -> small ints, for a hash-function name."""
    import numpy as np
    if fn == 'legendre_x':
        return 'legendre'
    for pre in ('x_low:', 'y_low:'):
        if fn.startswith(pre):
            k = int(fn[len(pre):])
            if 1 <= k <= 8:
                return (pre[0], k)
    if fn == 'label':           # test only: the full point (equivalent to a linear run)
        return 'label'
    raise HarnessError('unknown hash function %r' % fn)


def _point_table(h):
    """x, y coordinates of [k]G for k = 0..r-1 (k = 0: identity, coordinates -1). Cached per instance."""
    import numpy as np
    from ecc import Curve
    key = h.name
    if key not in _POINT_TABLES:
        E = Curve(h.p, h.a, h.b)
        G = (h._inst['Gx'], h._inst['Gy'])
        xs = np.full(h.r, -1, dtype=np.int64)
        ys = np.full(h.r, -1, dtype=np.int64)
        P = None
        for k in range(1, h.r):
            P = E.add(P, G)
            xs[k], ys[k] = P
        _POINT_TABLES[key] = (xs, ys)
    return _POINT_TABLES[key]


def _hash_grid(h, regs, fn):
    """Grid of hash values V[x_1, ..., x_k] = fn(sum_i x_i P_i) over all register values."""
    import numpy as np
    sizes = [1 << len(labs) for labs in regs]
    r = h.r
    lab = np.zeros(1, dtype=np.int64)
    for labs, z in zip(regs, sizes):
        x = np.arange(z, dtype=np.int64)
        li = np.zeros(z, dtype=np.int64)
        for t, c in enumerate(labs):
            li = (li + ((x >> t) & 1) * c) % r
        lab = (lab[..., None] + li) % r
    lab = lab.reshape(sizes)
    kind = _hash_fn(fn)
    if kind == 'label':
        return lab
    xs, ys = _point_table(h)
    if kind == 'legendre':
        key = h.name + '/legendre'
        if key not in _POINT_TABLES:
            ex = (h.p - 1) // 2
            _POINT_TABLES[key] = np.array([0] + [1 if pow(int(t), ex, h.p) == h.p - 1 else 0
                                                 for t in xs[1:]], dtype=np.int64)
        return _POINT_TABLES[key][lab]
    coord = xs if kind[0] == 'x' else ys
    return np.where(lab == 0, 0, coord[lab] & ((1 << kind[1]) - 1))


def grid_dist(h, regs, fn):
    """Exact outcome distribution of a hash run: the state sum_x |x>|f(x)> with f = fn(sum_i x_i P_i) after the
    accumulator is uncomputed; P(j, v) = |FFT(1[f = v])(j)|^2 / N^2 on the full grid of register values.
    Returns (sizes, {v: array of P(j, v) over the grid})."""
    import numpy as np
    V = _hash_grid(h, regs, fn)
    N = V.size
    return list(V.shape), {int(v): np.abs(np.fft.fftn((V == v).astype(np.complex128))) ** 2 / N ** 2
                           for v in np.unique(V)}


def grid_sample(h, regs, fn, rng):
    """Same distribution, sampled with one FFT: P(v) = |{x: f(x) = v}| / N exactly (Parseval), then j from the
    fibre's spectrum."""
    import numpy as np
    sizes = [1 << len(labs) for labs in regs]
    N = 1
    for z in sizes:
        N *= z
    if h.r.bit_length() > GRID_MAX_BITS or N > GRID_MAX_CELLS:
        h.unsimulated_runs += 1        # cost is still charged; outcomes are placeholders
        return [rng.getrandbits(len(labs)) for labs in regs] + [0]
    V = _hash_grid(h, regs, fn)
    vals, counts = np.unique(V, return_counts=True)
    i = min(len(vals) - 1, int(np.searchsorted(np.cumsum(counts), rng.random() * N, side='right')))
    v = int(vals[i])
    p = np.abs(np.fft.fftn((V == v).astype(np.complex128))).ravel() ** 2
    cdf = np.cumsum(p)
    idx = min(cdf.size - 1, int(np.searchsorted(cdf, rng.random() * cdf[-1], side='right')))
    return [int(t) for t in np.unravel_index(idx, sizes)] + [v]


class Context:
    """The object `h` passed to a candidate's solve(h). Public data, counted oracle, quantum runs."""

    def __init__(self, inst, d, seed, max_classical=2.0 ** 40, max_runs=10000, fail_per_add=0.0, kappa=False,
                 fail_split=None, approx=None, global_mc=None):
        self.name, self.n = inst['name'], inst['n']
        self.p, self.a, self.b, self.r = inst['p'], inst['a'], inst['b'], inst['r']
        self.lam, self.beta = inst.get('lam'), inst.get('beta')
        rng = random.Random(seed)
        self._qrng = random.Random(rng.getrandbits(64))
        self.rng = random.Random(rng.getrandbits(64))     # for the candidate's own randomness
        self._salt = rng.getrandbits(62)
        self._d = d
        self.ops = dict(group=0, field=0, lattice=0, other=0, cmp=0)
        self.max_classical, self.max_runs, self.fail_per_add = max_classical, max_runs, fail_per_add
        self.runs = []          # (shape, cost) per executed run
        self.garbage_runs = 0
        self._garbage = set()               # ids of bad runs (simulator-internal; used only by try_subset)
        self._attempt_left = None           # work left in the running try_subset() attempt (None: no attempt)
        self.capped_attempts = 0            # attempts stopped at their cap
        self.max_attempt = 0                # largest work of an executed attempt that finished under its cap
        self.heuristic_runs = 0
        self.kappa = kappa                  # flagged arithmetic: kappa coupling model (README), else heuristic
        self.fail_split = fail_split        # (pi, e_S) per addition on uniform operand pairs, for per-window offsets
        self.approx = fail_per_add > 0 if approx is None else approx   # failure weights apply (run.py --approx)
        self.global_mc = global_mc          # Lemma G constants (run.py --global-mc), for offset(global_scalar=True)
        self.lemma_g_runs = 0
        self.unsimulated_runs = 0
        self._inst = inst
        self.G = Point(self, 1)
        self.Q = Point(self, d, q=True)
        self.O = Point(self, 0)

    # -- accounting
    def _charge(self, kind, units):
        stop = False
        if self._attempt_left is not None:          # inside a try_subset() attempt: stop at its cap
            if units > self._attempt_left:
                units, stop = self._attempt_left, True
            self._attempt_left -= units
        self.ops[kind] += units
        if self.classical_ops() > self.max_classical:
            raise BudgetExceeded('classical work above %.3g units' % self.max_classical)
        if stop:
            raise _AttemptCap()

    def classical_ops(self):
        o = self.ops
        return o['group'] + o['field'] + o['lattice'] + o['other']

    def _record_run(self, shape, cost):
        if len(self.runs) >= self.max_runs:
            raise BudgetExceeded('more than %d runs' % self.max_runs)
        self.runs.append((shape, cost))

    # -- quantum
    def run(self):
        return Run(self)

    def try_subset(self, run_ids, fn, cap):
        """Decode from the outcomes of the runs `run_ids` with fn() (returns a verified answer or None) under a
        work cap of `cap` classical units for this attempt: counted work that would pass the cap stops the attempt
        there (it is charged exactly `cap` and returns None), so no attempt costs more than `cap` whatever the
        outcome values are. A subset holding a bad run (approximate-arithmetic
        model) is unusable: its residual outcome law is unknown, so the worst case for a decoder that only
        returns check()-verified answers is that the subset fails; such a call returns None and is charged the
        full cap, the most the attempt could have cost."""
        cap = int(cap)
        if cap <= 0 or self._attempt_left is not None:
            raise HarnessError('try_subset needs a positive cap and cannot be nested')
        if not self._garbage.isdisjoint(run_ids):
            self._charge('lattice', cap)
            return None
        before = self.classical_ops()
        self._attempt_left = cap
        try:
            out = fn()
        except _AttemptCap:
            self.capped_attempts += 1
            return None
        finally:
            self._attempt_left = None
        self.max_attempt = max(self.max_attempt, self.classical_ops() - before)
        return out

    # -- counted classical primitives
    def check(self, d):
        """Is [d] G == Q?  Charged as one double-and-add scalar multiplication."""
        k = int(d) % self.r
        self._charge('group', _dbl_add_cost(min(k, self.r - k)))
        return k == self._d

    def dlog_interval(self, base, target, lo, hi):
        """Some t in [lo, hi] with [t] base == target, else None. Baby-step giant-step on target - [lo] base:
        charged one scalar multiplication by lo plus 2 * ceil(sqrt(hi - lo + 1)) group operations."""
        lo, hi = int(lo), int(hi)
        if hi < lo:
            return None
        k = lo % self.r
        self._charge('group', _dbl_add_cost(min(k, self.r - k)) + 2 * math.isqrt(hi - lo) + 2)
        if base._l == 0:
            return lo if target._l == 0 else None
        t = target._l * pow(base._l, -1, self.r) % self.r
        t = lo + (t - lo) % self.r
        return t if t <= hi else None

    def charge(self, units, kind='other'):
        """Self-report classical work that is not done through the oracle (e.g. own enumeration loops)."""
        self._charge(kind, int(units))

    def lll(self, rows, delta=0.99):
        """LLL-reduce rows, charged (swaps + dim) dim^2 words units. Inside a try_subset() attempt the reduction
        gets the swap budget that keeps this charge within the attempt's remaining cap and stops inside its loops
        when the budget is used up; the attempt then fails. If not even the fixed part
        (dim^3 words) fits, the reduction does not start."""
        import lattice
        dim = len(rows)
        bits = max(1, max(abs(x) for row in rows for x in row).bit_length())
        unit = dim * dim * max(1, bits // 64)
        max_swaps = None
        if self._attempt_left is not None:
            max_swaps = self._attempt_left // unit - dim
            if max_swaps < 0:
                raise _AttemptCap()
        out, swaps = lattice.lll(rows, delta, max_swaps)
        self._charge('lattice', (swaps + dim) * unit)
        if out is None:
            raise _AttemptCap()
        return out

    def babai(self, rows, target):
        """Babai nearest plane on the given (preferably reduced) basis; returns (coefficients, lattice vector).
        Charged dim^3 words units; inside a try_subset() attempt it does not start if that would pass the cap."""
        import lattice
        dim = len(rows)
        bits = max(1, max(abs(x) for row in rows for x in row).bit_length())
        units = dim ** 3 * max(1, bits // 64)
        if self._attempt_left is not None and units > self._attempt_left:
            raise _AttemptCap()
        self._charge('lattice', units)
        return lattice.babai(rows, target)
