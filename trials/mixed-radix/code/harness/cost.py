"""Cost model (per run: peak logical qubits x Toffoli) and the project's reference numbers.

Baseline (Häner, Tripier, Young, Naehrig et al., arXiv 2609.05625v1, body text; re-read by h1 2026-11-12 after
windowed point additions of "approximately 1.196e6" Toffoli each (the largest of 100 sampled counts; no exact
integer is printed) plus 3 lookups of 2^16 entries each (Sect. III: 28 (1.196e6 + 3 2^16) = 38,993,024 ~ 39.0e6),
plus an initial lookup of 2^18 entries replacing the first windowed addition (Sect. II.2; Toffoli not printed,
charged as 2^18): 39,255,168 Toffoli. B0 = 1,457 x 39,255,168 = 5.7195e10 (the abstract's rounded 1,450 x 40e6 =
pf <= 0.0335 for the 28 additions (Sect. IV.2).
Grade "estimate": every windowed point addition is charged Häner's per-window figure (1,392,608 Toffoli incl. its 3
lookups). A run's exponent bits are cut into windows per add() call (ceil(bits / w)); an accumulator that starts
at the identity loads its first window by a direct lookup (no addition), charged separately: 2^w Toffoli at
grade estimate, the circuit's own counted initial lookup at grade measured (a whole addition if the V-B file has
none). The semiclassical QFT recycles control qubits, so exponent qubits do not add to the peak. Only meaningful
at n = 256; at toy sizes the harness reports the shape (bits, additions).
"""
import math, os

W = 16
HANER_QUBITS = 1457
HANER_TOFFOLI_PER_ADD = 1.196e6 + 3 * 2 ** 16      # 1,392,608: one windowed addition incl. its 3 lookups
HANER_FIRST_LOOKUP = 2 ** 18
HANER_TOFFOLI = 28 * HANER_TOFFOLI_PER_ADD + HANER_FIRST_LOOKUP   # 39,255,168
HANER_FAIL_PER_ADD = 0.0335 / 28          # their pf bound spread over 28 additions

B0 = HANER_QUBITS * HANER_TOFFOLI
B0_ROUNDED = 1450 * 40e6
T0_ATTEMPTS = 3                           # Häner's heuristic logical success 0.861 -> 3 attempts reach 0.99
T0 = T0_ATTEMPTS * B0                     # (their rigorous bound 0.553 would give 6 attempts)
CLASSICAL_CEILING = 2.0 ** 40


HANER = dict(qubits=HANER_QUBITS, toffoli_per_add=HANER_TOFFOLI_PER_ADD, w=W, fail_per_add=HANER_FAIL_PER_ADD,
             grade='estimate', first_lookup=2 ** W, exact=False)
MODEL = dict(HANER)       # run.py --measured replaces it with a V-B-checked circuit's counts


KINDS = {}    # name -> model of one more V-B-checked circuit (run.py --kind NAME=FILE; add(..., kind=NAME))
VB_KINDS = ('windowed_point_add', 'signed_windowed_point_add', 'hex_window_add', 'n7_window_add', 'm7_window_add',
            'hex_terminal_add')
# window of label W whose output point is measured
HEX_KINDS = ('hex_window_add', 'n7_window_add', 'm7_window_add', 'hex_terminal_add')


def terminal_classical_units(W):
    """Classical work of one run whose last addition is root-sign terminal (KIND hex_terminal_add, window W),
    worst case per run. After x_J is measured: decoding and
    Y_J = sqrt(x_J^3 + 7) by one exponentiation (<= 512 multiplications) and one inversion for the batch (<= 512);
    then for every one of the E = 4^W digits k of the 2W-bit residue-form address: the table point's unit image
    (1 multiplication; the orbit representatives are the run's tables, already charged 2 group operations each),
    lambda_+(k) and lambda_-(k) = (y_A(k) +- Y_J) / (x_A(k) - x_J) by Montgomery's batch inversion (3 multiplications
    per entry + 2 numerators = 5), and the phase-table bit (-1)^(m . payload(k) + m_Y . lambda_s(k)): 48 + 256 = 304
    bit products and their parity, charged one unit per bit (a bit operation is far less than a field
    multiplication, so this over-counts). Total 310 E + 1,032 units: 81,265,672 = 2^26.28 at W = 9."""
    return 310 * 4 ** W + 1032


def kind_label(kind, window):
    """the window label a V-B result can add (harness/mixreg.py label): W for a hex window, 'nw' / 'mw' for the norm-7
    windows (3 + w) 2^w (KIND n7_window_add) / (2 - w) 2^w (m7_window_add)"""
    return {'n7_window_add': 'n%d', 'm7_window_add': 'm%d'}[kind] % window if kind in ('n7_window_add', 'm7_window_add') \
        else window


def use_measured(path, n=256):
    """Price windowed additions with a V-B result (vb.py --json) for KIND windowed_point_add at width n."""
    m = model_from_vb(path, n)
    if m['hex']:
        raise SystemExit('--measured prices add() calls of binary registers; a hex window circuit is a --kind')
    for k in ('fail_split', 'alpha_mc'):
        MODEL.pop(k, None)
    MODEL.update(m)
    return MODEL


def model_from_vb(path, n=256):
    """The cost model of one V-B result (vb.py --json) at width n: counts per addition, failure weight, first
    lookup. With abort_cap in the entry the cap on the executed Toffoli is
    charged instead of the emitted count; the failure weight must then come from a source that counted the aborts
    (V-B's own lanes, or a run-input Monte Carlo that records the same cap). A file's extra_toffoli (work declared
    outside the checked circuit, e.g. correction-phase digit register of a hex window) is added per addition."""
    import json
    rep = json.load(open(path))
    if rep.get('kind') not in VB_KINDS:
        raise SystemExit('%s: a priced V-B result must be of KIND %s' % (path, ' / '.join(VB_KINDS)))
    MODEL = {}
    ent = [e for e in rep['sizes'] if e['n'] == n]
    if not ent:
        raise SystemExit('--measured file has no n = %d entry' % n)
    e = ent[0]
    if e['verdict'] != 'PASS':
        raise SystemExit('--measured circuit did not pass V-B at n = %d' % n)
    fail = e['random_failures'] / max(1, e['lanes']) if not rep.get('exact', True) else 0.0
    mc = e.get('run_input_mc')
    if not rep.get('exact', True) and mc is not None:
        # accumulator via the random offset; h1/vb_mc.py): upper confidence bound of flagged + 2 x unflagged-bad
        fail, src = mc['bound_per_add'], 'run-input Monte Carlo (%d lanes, %s)' % (mc['lanes'], mc['confidence'])
        # its error probability, for the joint confidence of the success claim (run.py): 'one-sided ..., 0.998 total'
        MODEL['alpha_mc'] = 1.0 - float(mc['confidence'].rsplit(', ', 1)[1].split()[0])
    else:
        src = 'declared estimate' if not rep.get('exact', True) else 'exact'
        if not rep.get('exact', True) and e.get('declared_failure') is not None:
            fail = max(fail, e['declared_failure'])      # rare failures the sampled lanes cannot see
    mcu = e.get('run_input_mc_uniform')
    MODEL.pop('fail_split', None)
    if not rep.get('exact', True) and mcu is not None:
        # misbehaving ones split into flagged and unflagged; 4 / r per addition for the exceptional accumulators
        # (O, B, -B, -2B), counted as unflagged. pi = all misbehaviour, e_S = the unflagged part.
        exc = 4 / 2 ** (n - 1)                           # >= 4 / r: the group order has n bits
        MODEL['fail_split'] = (mcu['ub_flagged'] + mcu['ub_bad'] + exc, mcu['ub_bad'] + exc)
        conf_u = 1.0 - float(mcu['confidence'].rsplit(', ', 1)[1].split()[0])
        MODEL['alpha_mc'] = max(MODEL.get('alpha_mc', 0.0), conf_u)
        if mc is None:
            fail = max(fail, MODEL['fail_split'][0])
            src = 'run-input Monte Carlo, uniform operand pairs (%d lanes, %s)' % (mcu['lanes'], mcu['confidence'])
    fl = e.get('first_lookup')
    flo = e.get('first_lookup_offset')
    MODEL['unsigned_tables'] = rep.get('kind') == 'windowed_point_add'   # the circuit reads unsigned 2^w tables
    tof, cap, extra = e['counts']['toffoli'], e.get('abort_cap'), rep.get('extra_toffoli', 0)
    note = ''
    if cap is not None:
        for m in (mc, mcu):
            if m is not None and m.get('abort_cap') != cap:
                raise SystemExit('%s: abort cap %d, but its run-input Monte Carlo was measured with cap %s (the '
                                 'aborts must be inside the failure weight)' % (path, cap, m.get('abort_cap')))
        note += '; abort cap %d executed Toffoli charged (%d emitted, %d of %d random-outcome lanes over it)' % (
            cap, tof, e.get('aborted', 0), e['counts'].get('outcome_lanes', 0))
        if rep.get('stream_cap') is not None and cap != tof - rep['stream_toffoli'] + rep['stream_cap']:
            # a capped hex window (h1/hexcap): the cap = its unconditional non-stream Toffoli + the stream's cap
            raise SystemExit('%s: abort cap %d is not the non-stream count %d plus the stream cap %d'
                             % (path, cap, tof - rep['stream_toffoli'], rep['stream_cap']))
        tof = min(tof, cap)
    if extra:
        note += '; + %d declared Toffoli outside the checked circuit (%s)' % (extra, rep.get('extra_toffoli_src', '?'))
    MODEL.update(stream_cap=rep.get('stream_cap'), stream_toffoli=rep.get('stream_toffoli'))
    MODEL.update(qubits=e['counts']['peak_qubits'], toffoli_per_add=tof + extra, w=e['window'],
                 toffoli_emitted=e['counts']['toffoli'] + extra, abort_cap=cap,
                 fail_per_add=fail, exact=bool(rep.get('exact', True)), pf_source=src,
                 flagged=bool(e['counts'].get('flag_used') or e.get('random_flagged') or e.get('exceptional_flagged')),
                 first_lookup=fl['toffoli'] if fl else tof + extra,
                 first_lookup_offset=flo['toffoli'] if flo else None,
                 qubits_add=e['counts']['peak_qubits'], hex=rep.get('kind') in HEX_KINDS, vb_kind=rep.get('kind'),
                 label=kind_label(rep.get('kind'), e['window']),
                 first_lookup_qubits=fl['peak_qubits'] if fl else e['counts']['peak_qubits'],
                 first_lookup_offset_qubits=flo['peak_qubits'] if flo else None,
                 grade='measured (%s%s%s%s)' % (os.path.basename(path),
                                                '' if n == 256 else ', n = %d: plumbing test' % n,
                                                '' if fl else '; initial lookup charged as one addition', note))
    if fl and fl['peak_qubits'] > MODEL['qubits']:
        MODEL['qubits'] = fl['peak_qubits']
    if flo and flo['peak_qubits'] > MODEL['qubits']:
        MODEL['qubits'] = flo['peak_qubits']
    return MODEL


def first_window_held(shape, M, windows):
    """Control qubits of the first lookup that stay live for the whole run. The
    semiclassical QFT measures a register's qubits from its most significant one down, so the first lookup's
    controls can be measured right after it only if they are the top part of their register: a whole add() call
    that reaches its register's top bit. Any other first-window control (e.g. the lowest window of a longer
    register) waits for the register's higher windows and adds to the peak. Hash runs keep their exponent qubits
    live and shaped qubits of amplitude registers are counted by shaped_held (both elsewhere)."""
    if not (shape['start_zero'] and windows) or 'hash' in shape:
        return 0
    held, left = 0, M['w']
    tops = shape.get('add_top', [False] * len(shape['add_bits']))
    for bits, top in zip(shape['add_bits'], tops):
        c = min(bits, left)
        if c < bits or not top:
            held += c
        left -= c
        if left == 0 or not M.get('pack'):     # unpacked: the first window lies inside the first add() call
            break
    return held


def call_windows(add_bits, M):
    """Window indices (0 = the first window) touched by each add() call, in run_shape_cost's window model."""
    out, pos = [], 0
    for bits in add_bits:
        if M.get('pack'):       # joint windows: positions in call order
            out.append(range(pos // M['w'], (pos + bits - 1) // M['w'] + 1))
            pos += bits
        else:                   # windows per call
            n = math.ceil(bits / M['w'])
            out.append(range(pos, pos + n))
            pos += n
    return out


def shaped_held(shape, M, windows):
    """Shaped qubits of amplitude registers (Run.reg with amplitudes) that add to the run's peak: all of them,
    except those covered by an add() call that lies entirely in the first window, the exact first lookup, and
    reaches its register's top bit. The semiclassical QFT measures those right after the lookup, and the lookup's
    own measured peak (V-B first_lookup / first_lookup_offset) already holds its address qubits."""
    free = 0
    if shape['start_zero'] and windows:
        for ws, f in zip(call_windows(shape['add_bits'], M), shape['add_free']):
            if f and ws[-1] == 0:
                free += f
    return shape['amp_bits'] - free


def q_tables(shape, M, windows):
    """Does an approximate addition read a window table built from Q? Only the
    first window, loaded by the exact first lookup, may hold bits of an add() call whose point depends on Q."""
    qs = shape.get('add_q', [False] * len(shape['add_bits']))
    first = M['w'] if shape['start_zero'] and windows else 0
    pos = 0
    for k, (bits, q) in enumerate(zip(shape['add_bits'], qs)):
        if M.get('pack'):
            lo, hi = pos, pos + bits                 # bit range of this call in the packed sequence
            pos = hi
            if q and hi > first:
                return True
        elif q and (k > 0 or bits > first):         # unpacked: windows per call, the first lookup in call 0
            return True
    return False


def run_shape_cost(shape, n):
    """Hash runs (Run.hash): the additions are done twice (compute, uncompute), the declared hash Toffoli are
    added, and the exponent registers stay live (no recycling), so they and the hash qubits add to the peak."""
    M = MODEL
    if M.get('pack'):           # joint windows over qubits of several add() calls (README: packed windows)
        windows = math.ceil(sum(shape['add_bits']) / M['w'])
    else:
        windows = sum(math.ceil(b / M['w']) for b in shape['add_bits'])
    additions = max(0, windows - (1 if shape['start_zero'] and windows else 0))
    qubits, grade, extra = M['qubits'], M['grade'], 0.0
    if 'hash' in shape:
        additions *= 2
        extra = shape['hash_toffoli']
        qubits += sum(shape['reg_bits']) + shape['hash_qubits']
        grade += ' + declared hash cost'
    if 'prep_toffoli' in shape:            # amplitude registers: preparation cost, shaped qubits held
        extra += shape['prep_toffoli']
        if 'hash' not in shape:
            qubits += shaped_held(shape, M, windows)
        grade += (' + declared state preparation' if shape.get('prep_declared', True)
                  else ' + counted sine-state preparation')
    first = M.get('first_lookup', 0) if shape['start_zero'] and windows else 0
    if shape.get('offset') and windows:                 # Run.offset(): unsigned 2^w table of T0[k] + R0
        if M.get('first_lookup_offset') is not None:
            first = M['first_lookup_offset']
        elif M['grade'] == 'estimate':
            first = 2 ** M['w']
        else:
            raise ValueError('offset runs need first_lookup_offset in the V-B file (vb.py records it)')
    if 'hash' in shape:
        first *= 2                          # the accumulator is uncomputed as well
    qubits += first_window_held(shape, M, windows)
    per_add = M['toffoli_per_add']
    if shape.get('window_offsets') and additions and not M.get('unsigned_tables'):
        # per-window offsets read unsigned 2^w tables T_w[k] + R_w; a signed circuit's counts do not apply
        if not M['grade'].startswith('estimate'):
            raise ValueError('per-window offsets need a V-B file measured with unsigned tables (KIND '
                             'windowed_point_add)')
        per_add += 2 ** (M['w'] - 1)
        grade += ' + unsigned tables (declared +2^(w-1) Toffoli per addition)'
    toffoli = additions * per_add + extra + first
    return dict(exp_bits=sum(shape['reg_bits']), windows=windows, additions=additions, first_lookup=first,
                qubits=qubits, toffoli=toffoli, cost=qubits * toffoli, grade=grade,
                q_tables=q_tables(shape, M, windows))


GLOBAL_MC = None   # run.py --global-mc FILE: Lemma G constants (global_mc_constants)
GLOBAL_MC_BY_N = {}   # run.py --global-mc N=FILE: the constants for size N (its runs' own window sequence)
FIRST = {}    # run.py --first-lookup FILE: V-B result of a first-lookup circuit (bits, toffoli, peak_qubits, src)


def declared_kind(spec):
    """run.py --kind NAME=declared:TOFFOLI:QUBITS:PF:W, an estimate-grade kind (e.g. a terminal addition whose
    circuit is not built yet): never meets a target. W is the window label: W for a hex window, nw / mw for a norm-7
    window of a mixed-radix register (harness/mixreg.py)."""
    import mixreg
    tof, q, pf, w = spec.split(':')[1:]
    lab = int(w) if w.isdigit() else w
    if not mixreg.check_label(lab):
        raise SystemExit('declared kind %s: the window must be W (1..10) or nw / mw (w = 1..8)' % spec)
    return dict(toffoli_per_add=float(tof), qubits_add=int(q), qubits=int(q), fail_per_add=float(pf),
                w=lab if isinstance(lab, int) else int(lab[1:]), label=lab, hex=True, exact=float(pf) == 0,
                flagged=True, pf_source='declared', abort_cap=None,
                grade='estimate (declared %s Toffoli x %s qubits, pf %s)' % (tof, q, pf))


def use_first_lookup(path, n=256):
    """A V-B result of the run's first lookup (e.g. o1/first_ss.py, its V-B size = the address bits, parameter P
    = the payload, which must be the 2n-bit accumulator)."""
    import json
    rep = json.load(open(path))
    pars = dict(kv.split('=', 1) for kv in rep.get('params', []))
    if int(pars.get('P', -1)) != 2 * n:
        raise SystemExit('%s: the first lookup must load the 2n = %d-qubit accumulator (-p P=%d)' % (path, 2 * n, 2 * n))
    FIRST.update({e['n']: dict(toffoli=e['counts']['toffoli'], peak_qubits=e['counts']['peak_qubits'],
                               src=os.path.basename(path)) for e in rep['sizes'] if e['verdict'] == 'PASS'})


def hex_run_cost(shape, n):
    """Whole-run cost of a run with hex registers (Run.hexreg / hexadd), window by window. The first lookup loads
    the standard registers (whole) and the top windows marked 'first' from the identity: priced by --first-lookup
    other window is one addition of its kind: Toffoli summed (the cap where the kind has one), peak = the largest
    of the parts, failure weight = the sum of the kinds' weights per addition (fail_weight)."""
    import hexreg, mixreg
    first_bits = sum(shape['add_bits']) + sum(mixreg.win_qubits(e['Ws'][-1]) + abs(e.get('split', 0))
                                              for e in shape['hex'] if e['kinds'][-1] == 'first')
    if first_bits and not shape['start_zero']:
        raise ValueError('the first lookup loads the identity accumulator: no start() in a run with one')
    if shape['start_zero'] and not first_bits:
        raise ValueError("a run from the identity needs a first lookup (standard registers or a 'first' window)")
    tof, qub, fw, parts, adds, q_tables, entries = 0.0, 0, 0.0, {}, 0, False, 0
    add_ws, add_kinds, term_units = [], [], 0         # window label and kind per addition, in run order
    for e in shape['hex']:
        for W, k in zip(e['Ws'], e['kinds']):
            if k == 'first':
                continue
            if k not in KINDS:
                raise ValueError('kind %r is not loaded (run.py --kind %s=VB.json)' % (k, k))
            m = KINDS[k]
            if not m['hex'] or m.get('label', m['w']) != W:   # window labels (mixreg.label): W, or 'nw' / 'mw'
                raise ValueError('kind %r (window %s%s) cannot add a window %s'
                                 % (k, m.get('label', m['w']), '' if m['hex'] else ', not a hex circuit', W))
            tof += m['toffoli_per_add']
            qub = max(qub, m['qubits_add'])
            fw += m['fail_per_add']
            parts[k] = parts.get(k, 0) + 1
            adds += 1
            add_ws.append(W)
            add_kinds.append(k)
            if m.get('vb_kind') == 'hex_terminal_add':
                term_units += terminal_classical_units(W)
            q_tables |= e['q']
            entries += mixreg.win_entries(W)          # the window's table of orbit representatives
    grade = ['%d x %s: %s' % (c, k, KINDS[k]['grade']) for k, c in sorted(parts.items())]
    first = 0
    if first_bits:
        f = FIRST.get(first_bits) if n == 256 else None
        if f is not None:
            grade.append('first lookup %d bits: measured (%s)' % (first_bits, f['src']))
        else:
            f = hexreg.first_ss_toffoli(first_bits, n)
            grade.append('first lookup %d bits: estimate (o1/first_ss.py formula)' % first_bits)
        first = f['toffoli']
        qub = max(qub, f['peak_qubits'])
    extra = shape.get('prep_toffoli', 0.0)
    if 'prep_toffoli' in shape:
        grade.append('declared state preparation' if shape.get('prep_declared', True)
                     else 'counted sine-state preparation')
    toffoli = tof + first + extra
    return dict(exp_bits=sum(shape['reg_bits']), windows=adds + (1 if first_bits else 0), additions=adds,
                add_ws=add_ws, add_kinds=add_kinds, term_classical=term_units, first_lookup=first, first_bits=first_bits, qubits=qub, toffoli=toffoli, cost=qub * toffoli,
                grade='; '.join(grade), q_tables=q_tables, fail_weight=fw, kinds=sorted(parts),
                table_entries=entries + (2 ** first_bits if first_bits else 0),
                measured=all(not g.split(': ', 1)[1].startswith('estimate') for g in grade if ': ' in g))


def _log_binom_cdf(k, n, p):
    if p <= 0:
        return 0.0
    if p >= 1:
        return 0.0 if k >= n else -math.inf
    t = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)
         for i in range(k + 1)]
    mx = max(t)
    return mx + math.log(sum(math.exp(x - mx) for x in t))


def cp_upper(k, n, alpha):
    """one-sided Clopper-Pearson upper bound: the p with P(X <= k | n, p) = alpha"""
    if k >= n:
        return 1.0
    lo, hi, la = k / n, 1.0, math.log(alpha)
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _log_binom_cdf(k, n, mid) > la else (lo, mid)
    return hi


def cp_lower(k, n, alpha):
    """one-sided Clopper-Pearson lower bound: the p with P(X >= k | n, p) = alpha"""
    return 0.0 if k <= 0 else 1.0 - cp_upper(n - k, n, alpha)


def mean_blocks(vmax):
    """The blocks [lo, hi] that ub_mean splits 1..vmax into: 1, 2, 3, 4-7, 8-15, 16-31, ... (clipped at vmax). They
    depend on vmax alone, so they are fixed before the data are seen."""
    out, lo = [], 1
    while lo <= vmax:
        hi = lo if lo < 4 else 2 * lo - 1
        out.append((lo, min(hi, vmax)))
        lo = hi + 1
    return out


def ub_mean(hist, n, vmax, a):
    """Upper confidence bound (level 1 - a) on E[V] for an integer variable V in [0, vmax], from n independent
    samples whose values are counted in hist (value -> count; values absent from hist have count 0):
    E[V] = sum_(j = 1..vmax) P(V >= j) <= sum over the blocks [lo, hi] of mean_blocks(vmax) of (hi - lo + 1) P(V >= lo),
    each P(V >= lo) bounded by one-sided Clopper-Pearson at a / (number of blocks) (union bound). Values never
    observed are covered by their block's bound; nothing is chosen from the data."""
    bl = mean_blocks(vmax)
    if any(not 0 <= v <= vmax for v in hist) or sum(hist.values()) != n:
        raise ValueError('ub_mean: values outside [0, %d] or counts that do not add up to n = %d' % (vmax, n))
    return sum((hi - lo + 1) * cp_upper(sum(c for v, c in hist.items() if v >= lo), n, a / len(bl)) for lo, hi in bl)


def global_mc_constants(path, a=0.001):
    """Lemma G constants (paper Sec. V.D and App. B;  Sections 2-4) from a path Monte Carlo of the run's additions under one global
    scalar per run (o3/sim/path_mc.py JSON; several files of one window sequence, separated by commas, are pooled).
    Every bound is on path-level variables: the paths are independent (each has its own c, R0, digits and records),
    Lemma G needs g0 = 1 - E[N_DO + 2 N_P] - 8 J / r, so eS = UB E[N_DO + 2 N_P] per path, and e0's UB E[corr] per
    path (corr in [0, 2J], paper Sec. V.D), both by ub_mean over blocks fixed by J (finding 3); UB P(recovered), UB P(class
    (iii) end) and LB phi of a visibly flagged end (an R flag or an off-curve accumulator; with the abort cap,
    h1/hexcap/path_mc_cap.py, an abort counts as a deviation of its addition and ends the path visibly flagged). Five
    one-sided bounds at level a each (alpha = 5 a enters the joint confidence). fDO / fP are the observed per-addition
    rates, printed only. stream_cap / stream_toffoli: the cap on the stream's executed CCX + CCZ under which the
    aborts were counted (None if the Monte Carlo did not count them), and the stream's emitted count; kinds charged at
    a cap must match both. A Monte Carlo whose additions ran under different caps (o1/s6/path_mc_term.py: the lowest
    first (stream_caps / stream_toffolis); then addition i's kind must match entry i (hcore, Lemma G runs), and its
    terminal's visible flags (end tflag: invalid x2, no root, not exactly one payload match, live ancilla) count in
    phi like R flags, aborts and off-curve ends (o1/s6/termva.py, folded into the harness in )."""
    import json
    pm = None
    for f in path.split(','):             # several batches of one window sequence are pooled
        d = json.load(open(f))
        import mixreg
        if d.get('layout'):                    # o3/s6/path_mc_n7.py: 'layout' (mixreg syntax) gives the labels,
            d['windows'] = mixreg.labels(d['layout'])                         # 'windows' holds stand-in sizes
        else:                                  # labels: W (int) or 'nw' / 'mw'; path_mc.py: J windows of W = 8
            d['windows'] = [w if isinstance(w, str) else int(w) for w in d.get('windows', [8] * d['J'])]
        if not all(mixreg.check_label(w) for w in d['windows']):
            raise ValueError('%s: window labels must be W (1..10) or nw / mw (w = 1..8): %s' % (f, d['windows']))
        if len(d['windows']) != d['J']:
            raise ValueError('%s: %d windows listed for J = %d' % (f, len(d['windows']), d['J']))
        d['end'].setdefault('aflag', 0)   # h1/hexcap/path_mc_cap.py: paths ended by an abort at the executed-Toffoli cap
        d['end'].setdefault('tflag', 0)   # o1/s6/path_mc_term.py: paths ended by the root-sign terminal's own flag
        d['stream_cap'], d['stream_toffoli'] = d.get('stream_cap'), d.get('stream_toffoli')
        d['stream_caps'], d['stream_toffolis'] = d.get('stream_caps'), d.get('stream_toffolis')
        if d['stream_caps'] is not None and (len(d['stream_caps']) != d['J'] or len(d['stream_toffolis']) != d['J']):
            raise ValueError('%s: per-addition cap lists of length %d / %d for J = %d'
                             % (f, len(d['stream_caps']), len(d['stream_toffolis']), d['J']))
        if pm is None:
            pm = dict(d, per_addition=dict(d['per_addition']), end=dict(d['end']), counts_DO_P=dict(d['counts_DO_P']))
            continue
        if (d['J'], d['windows'], d['stream_cap'], d['stream_toffoli'], d['stream_caps'], d['stream_toffolis']) != (
                pm['J'], pm['windows'], pm['stream_cap'], pm['stream_toffoli'], pm['stream_caps'], pm['stream_toffolis']):
            raise ValueError('%s: another window sequence or abort cap than %s' % (f, path.split(',')[0]))
        pm['paths'] += d['paths']
        for grp in ('per_addition', 'end'):
            for k in (('lanes', 'DO', 'P') if grp == 'per_addition' else
                      ('rflag', 'aflag', 'tflag', 'offcurve', 'oncurve_wrong', 'recovered')):
                pm[grp][k] += d[grp][k]
        for k, v in d['counts_DO_P'].items():
            pm['counts_DO_P'][k] = pm['counts_DO_P'].get(k, 0) + v
    pa, e, n, J, windows = pm['per_addition'], pm['end'], pm['paths'], pm['J'], pm['windows']
    S, corr = {}, {}                      # per path: N_DO + 2 N_P, and corr (0 unless N_DO + N_P >= 2)
    for key, cnt in pm['counts_DO_P'].items():
        nd, np_ = map(int, key.split(','))
        S[nd + 2 * np_] = S.get(nd + 2 * np_, 0) + cnt
        v = nd + 2 * np_ - 1 + ((-1) ** np_ if nd == 0 else 0) if nd + np_ >= 2 else 0
        corr[v] = corr.get(v, 0) + cnt
    eS, ecorr = ub_mean(S, n, 2 * J, a), ub_mean(corr, n, 2 * J, a)
    rec, iii = cp_upper(e['recovered'], n, a), cp_upper(e['oncurve_wrong'], n, a)
    return dict(J=J, windows=windows, eS=eS, fDO=pa['DO'] / pa['lanes'], fP=pa['P'] / pa['lanes'],
                e0_paths=ecorr + rec + iii, ecorr=ecorr, rec=rec, iii=iii,
                phi=cp_lower(e['rflag'] + e['aflag'] + e['tflag'] + e['offcurve'], n, a), alpha=5 * a,
                stream_cap=pm['stream_cap'], stream_toffoli=pm['stream_toffoli'],
                stream_caps=pm['stream_caps'], stream_toffolis=pm['stream_toffolis'],
                src='%s (%d paths of J = %d, %d per-addition lanes)' % (
                    ' + '.join(os.path.basename(f) for f in path.split(',')), n, J, pa['lanes']))


def targets(B, T):
    return dict(intermediate=(B <= B0 / 2 and T <= 4 * T0), main=(B <= B0 / 4 and T <= 10 * T0))
