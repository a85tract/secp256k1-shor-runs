#!/usr/bin/env python3
"""U1 window-size sweep: the ECDSA.fail leader stream in plain signed windows of size W.

o2/port/windowed_lead.py replays the leader's raw stream (sha256 fb14f5d5, 761,147 CCX) and changes only how the
addend enters: at five sites (y0, x1, x3, x4, y5) one select-swap read of the site's addend functions from the
2^(W-1)-entry signed table (s2/port/windowed3616.py _qload_ss), one measurement-based unload (_qunload), and for the
two y sites a negate_nonzero after the read and before the unload (h1/vbarith.py); the stream's 3 addend-conditioned
CX become CCX. Only this layer depends on W. The raw stream is not on this box (o2/ext/ is outside git), so this
script counts the layer itself, gate by gate, with those same functions on a counting builder, and adds the measured
stream: emitted(W) = 761,147 + 3 + layer(W); charged at the abort cap = 726,000 + 3 +
select-swap's one CSWAP per output bit, so their sum is fitted at W = 16 (769 = 257 + 256 + 256; negate_nonzero
counts 251 Toffoli, not n, because p + 1 has zero bits); the fit carries over to every W.
read layer's scratch (S, R1 and the 2^ceil((W-1)/2) one-hot qubits) fits under the stream's peak at W = 16, and the
one-hot register has the same size at W = 17 (2^8), so the formula is used for W <= 17 and is a lower bound above.

Run shape (harness/candidates/shor.py with NU_FRAC = 0.1875, Haener et al.'s configuration): a = 256 bits, b = 208
bits; with W the windows are cut per register, the top f bits of a go into the first lookup (o1/first_ss.py formula,
measured at 17 and 18 bits; the signed port's own W-bit first lookup when f = W). Writes base/lead_w16_cap.json (the
V-B file with the abort cap and the per-addition failure weight declared, for harness/run.py --measured).
usage: python3 base/u1_sweep.py [--json OUT]
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [os.path.join(REPO, 'h1'), os.path.join(REPO, 's2', 'vb'), os.path.join(REPO, 's2', 'port'),
                os.path.join(REPO, 'harness')]
import vbarith as va            # negate_nonzero
import windowed3616 as w3       # _qload_ss, _qunload
import hexreg                   # first_ss_toffoli (o1/first_ss.py formula)
import cost                     # B0, T0

STREAM_CCX = 761147
STREAM_CAP = 726000             # abort cap on the stream's executed CCX (0 of 25,600 lanes above it)
ADDEND_CCX = 3                  # the stream's addend-conditioned CX become CCX
VB_W16 = 'o2/results4/lead_w16_n256_flags.json'
P = 2 ** 256 - 2 ** 32 - 977    # secp256k1 field prime (negate_nonzero's constant)
PF = (727 + 82) / 327680        # leader core, signed port on fixed G tables: flagged + phase-only per addition
PF_SRC = ('s2 classical replay of the raw stream, 327,680 lanes on fixed G tables: 727 flagged + 82 phase-only '
          '')


class Count:
    """counting builder: Toffoli (ccx, ccz, and_), live and peak qubits; measurements free their qubit"""
    def __init__(self):
        self.n = self.live = self.peak = self.tof = 0

    def alloc(self, k):
        q = list(range(self.n, self.n + k))
        self.n += k
        self.live += k
        self.peak = max(self.peak, self.live)
        return q

    def free(self, qs):
        self.live -= len(qs)

    def x(self, q, cond=None): pass
    def cx(self, a, t, cond=None): pass
    def z(self, q, cond=None): pass
    def cz(self, a, b, cond=None): pass

    def ccx(self, a, b, t, cond=None):
        self.tof += 1

    def ccz(self, a, b, t, cond=None):
        self.tof += 1

    def and_(self, a, b):
        self.tof += 1
        return self.alloc(1)[0]

    def measure_x(self, q):
        self.live -= 1
        return 0

    def __getattr__(self, name):
        raise AttributeError('counting builder has no %s (extend it)' % name)


def layer(W, widths, ss=True):
    """Toffoli of the addend layer of one signed W window: five reads (select-swap, or plain unary iteration with
    ss=False, O2_SELECT=00000), five unloads, four y negations"""
    c = Count()
    idx = c.alloc(W - 1)
    sgn = c.alloc(1)[0]
    bl = [[] for _ in range(1 << (W - 1))]      # table bits: they set CNOT / CZ patterns only, not Toffoli
    for k, wd in enumerate(widths):
        reg = c.alloc(wd)
        (w3._qload_ss if ss else w3._qload)(c, idx, reg, bl)
        if k in (0, 4):                          # y0, y5: yA -> p - yA for a negative digit, and back
            va.negate_nonzero(c, reg, P, sgn)
            va.negate_nonzero(c, reg, P, sgn)
        w3._qunload(c, idx, reg, bl)
    return c.tof


def first_lookup(f, W, vb):
    """Toffoli, peak qubits of an f-bit first lookup: the signed port's own W-bit lookup if f == W, else the
    select-swap formula of o1/first_ss.py (V-B PASS, equal to the measured 17- and 18-bit circuits)"""
    if f == W == 16:
        fl = vb['first_lookup']
        return fl['toffoli'], fl['peak_qubits'], 'measured (signed port, %s)' % os.path.basename(VB_W16)
    g = hexreg.first_ss_toffoli(f, 256)
    return g['toffoli'], g['peak_qubits'], 'o1/first_ss.py formula'


def main():
    out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else os.path.join(HERE, 'lead_w16_cap.json')
    rep = json.load(open(os.path.join(REPO, VB_W16)))
    e16 = [e for e in rep['sizes'] if e['n'] == 256][0]
    meas = e16['counts']['toffoli']
    base16 = layer(16, [256, 0, 0, 0, 256])
    sw = meas - STREAM_CCX - ADDEND_CCX - base16          # fitted sum of the three x-site widths
    assert 3 * 250 <= sw <= 3 * 260, 'fitted x-site widths %d: not three registers of about 256 bits' % sw
    xw = [sw // 3 + (1 if i < sw % 3 else 0) for i in range(3)]
    widths = [256, xw[0], xw[1], xw[2], 256]
    print('calibration W = 16: layer %d + stream %d + %d = %d = V-B count %d; x-site widths %s'
          % (layer(16, widths), STREAM_CCX, ADDEND_CCX, STREAM_CCX + ADDEND_CCX + layer(16, widths), meas, xw))
    rows = []
    for W in range(12, 21):
        L = layer(W, widths)
        cap = STREAM_CAP + ADDEND_CCX + L
        q = 1227 + W + 1
        best = None
        for f in range(10, 21):
            if f > 256:
                continue
            adds = math.ceil((256 - f) / W) + math.ceil(208 / W)
            ft, fq, fsrc = first_lookup(f, W, e16)
            tof = adds * cap + ft
            qq = max(q, fq)
            if best is None or qq * tof < best[0]:
                best = (qq * tof, f, adds, ft, tof, qq, fsrc)
        rows.append((W, L, cap, q, cap * q / W) + best)
    print('%3s %8s %9s %6s %12s | %3s %4s %10s %11s %6s %8s' % ('W', 'layer', 'at cap', 'qubits', 'q*T per bit',
                                                             'f', 'adds', 'first', 'Toffoli', 'peak', 'B/B0'))
    for W, L, cap, q, pb, B, f, adds, ft, tof, qq, fsrc in rows:
        print('%3d %8d %9d %6d %12.4e | %3d %4d %10d %11d %6d %8.5f%s' % (W, L, cap, q, pb, f, adds, ft, tof, qq,
                                                                       B / cost.B0, '' if W <= 17 else ' (peak LB)'))
    Lu = layer(16, widths, ss=False)
    tu = 28 * (STREAM_CAP + ADDEND_CCX + Lu) + e16['first_lookup']['toffoli']
    print('W = 16 with plain unary-iteration reads instead of select-swap: layer %d, window %d at cap, run %d Toffoli '
          'x 1,244 = %.6f B0 (select-swap gain %.4f)' % (Lu, STREAM_CAP + ADDEND_CCX + Lu, tu, 1244 * tu / cost.B0,
                                                        tu / (28 * (STREAM_CAP + ADDEND_CCX + layer(16, widths)) +
                                                              e16['first_lookup']['toffoli'])))
    Wb = min(rows, key=lambda x: x[5])
    print('B-minimizing window: W = %d (first lookup %d bits, %d additions): B / B0 = %.6f' % (Wb[0], Wb[6], Wb[7],
                                                                                               Wb[5] / cost.B0))
    d = dict(rep)
    d['stream_cap'], d['stream_toffoli'] = STREAM_CAP, STREAM_CCX
    d['derived_by'] = ('base/u1_sweep.py from %s: abort cap = emitted - stream CCX + stream cap '
                       '(cost.model_from_vb checks it); declared_failure = %s' % (VB_W16, PF_SRC))
    s = [dict(x) for x in rep['sizes']]
    for x in s:
        if x['n'] == 256:
            x['abort_cap'] = meas - STREAM_CCX + STREAM_CAP
            x['declared_failure'] = PF
    d['sizes'] = s
    json.dump(d, open(out, 'w'), indent=1)
    print('wrote %s: abort cap %d, declared failure %.4e per addition' % (out, meas - STREAM_CCX + STREAM_CAP, PF))


if __name__ == '__main__':
    main()
