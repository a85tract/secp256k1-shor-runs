#!/usr/bin/env python3
"""Budget-constrained splitting frontier.

  - A run over |S| = 2^D exponent vectors reveals at most f(D) = log2(1 + (2^D - 1)/r) bits about d (Theorem 1;
    assumptions A1-A5: linear runs from a point start, the point's label unused, runs independent and fixed in advance,
    d uniform).
  - Success 0.99 with a list of K = 2^80 candidates (what 2^40 classical group operations can test, A5) needs, by Fano,
    sum_i f(D_i) >= I = 0.99 (log2 r - 80) - 1 = 173.24 bits over the runs, whatever their sizes (Corollary 1).
  - A run of D bits costs C(D) = (D/16 - 1) windows at Haener's per-window cost (numbers_n256.cost: estimate grade,
    fractional windows, the first 16 bits a free lookup), in units of B0. Runs are error-free, so T = sum_i C(D_i)
    with no reruns (failures and reruns only add cost, so they can only lower the frontier's X).
Frontier: with at most R runs of any sizes, the smallest largest run D*(R) with sum f >= I and sum C <= budget. As a
function of the cost y = C(D), f is convex (softplus of an affine map), so for a largest run D the information a budget
buys is largest at a vertex of {0 <= y_i <= C(D), sum y_i <= budget}: R runs at D if they fit, else k = floor(budget /
C(D)) runs at D and one run with the remaining cost. That is exact for unequal sizes; D*(R) is Corollary 1's equal-run
value until R runs at it exceed the budget and constant (the budget's minimum) from there on.
Toffoli x 1,244 q per window, about half of Haener's, so the same T buys twice the windows); the adaptive volume bound
(Theorem 2: d-averaged success <= K/r + sum_t (N_t - 1)/r, same knapsack, every history within the budget) at 10 T0.
X = B0 / C(D*) is a ratio of windows, the same at either core's per-window cost.
usage: python3 base/frontier.py [--csv OUT] [--rmax N]   (writes base/frontier.csv, R = 1..1000)
"""
import contextlib, io, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 's3', 'theory'))
with contextlib.redirect_stdout(io.StringIO()):
    import numbers_n256 as nb   # lr, win, B0, cost(), bits_nonadaptive(), bits_adaptive_volume()

LOGK = 80
I_NEED = 0.99 * (nb.lr - LOGK) - 1                      # bits, Corollary 1
V_NEED = 0.99 - 2.0 ** (LOGK - nb.lr)                   # Theorem 2: sum_t (N_t - 1)/r needed
WIN_U1 = 816538 * 1244                                  # q x Toffoli per window, base/lead_w16_cap.json (abort cap)
BUDGETS = (('10T0', 30.0), ('4T0', 12.0), ('10T0u1', 30.0 * nb.win / WIN_U1))   # in B0 at Haener's window cost


def info(D):
    return math.log2(1 + (2.0 ** D - 1) / 2.0 ** nb.lr)


def vol(D):
    return (2.0 ** D - 1) / 2.0 ** nb.lr


def bits_of(y):                                         # inverse of nb.cost on y > 0
    return 16 * (1 + y * nb.B0 / nb.win)


def bought(D, R, budget, g):
    """largest sum of g over at most R runs of sizes <= D with summed cost <= budget (the vertex above)"""
    c = nb.cost(D)
    if R * c <= budget:
        return R * g(D)
    k = int(budget // c)
    rem = budget - k * c
    return k * g(D) + (g(bits_of(rem)) if rem > 0 else 0.0)


def frontier(R, budget, g=info, need=I_NEED):
    lo, hi = 17.0, 700.0
    assert bought(hi, R, budget, g) >= need
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if bought(mid, R, budget, g) >= need else (mid, hi)
    return hi


def main():
    out = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else os.path.join(HERE, 'frontier.csv')
    rmax = int(sys.argv[sys.argv.index('--rmax') + 1]) if '--rmax' in sys.argv else 1000
    print('Theorem 1 + Fano: I = %.4f bits over the runs (K = 2^%d, success 0.99); window cost %.4e q x Toffoli '
          '(Haener), %.4e (U1) -> budget 10 T0 on U1 = %.2f B0 at Haener cost' % (I_NEED, LOGK, nb.win, WIN_U1,
                                                                              BUDGETS[2][1]))
    rows, prev = [], 0.0
    for R in range(1, rmax + 1):
        De = nb.bits_nonadaptive(R, LOGK)
        Te = R * nb.cost(De)
        assert Te >= prev, 'R x C(D_eq(R)) not increasing at R = %d' % R
        prev = Te
        row = dict(R=R, D_eq=De, B_eq=nb.cost(De), T_eq_T0=Te / 3)
        for name, b in BUDGETS:
            row['D_' + name] = frontier(R, b)
        row['D_ad10T0'] = frontier(R, 30.0, vol, V_NEED)
        rows.append(row)
    names = [n for n, _ in BUDGETS] + ['ad10T0']
    with open(out, 'w') as f:
        f.write('R,D_eq,B_eq_B0,T_eq_T0,' + ','.join('D_%s,B_%s_B0,X_%s' % (n, n, n) for n in names) + '\n')
        for r in rows:
            f.write('%d,%.4f,%.6f,%.4f,' % (r['R'], r['D_eq'], r['B_eq'], r['T_eq_T0']) +
                    ','.join('%.4f,%.6f,%.4f' % (r['D_' + n], nb.cost(r['D_' + n]), 1 / nb.cost(r['D_' + n]))
                             for n in names) + '\n')
    print('%5s | %8s %7s %7s | %8s %6s | %8s %6s | %8s %6s | %8s %6s' % (
        'R', 'D_eq', 'B_eq', 'T/T0', 'D 10T0', 'X', 'D 4T0', 'X', 'D 10T0u1', 'X', 'D ad10', 'X'))
    for r in rows:
        if r['R'] in (1, 2, 4, 8, 16, 21, 22, 32, 55, 56, 64, 110, 111, 128, 240, 1000):
            print('%5d | %8.3f %7.4f %7.2f | ' % (r['R'], r['D_eq'], r['B_eq'], r['T_eq_T0']) +
                  ' | '.join('%8.3f %6.4f' % (r['D_' + n], 1 / nb.cost(r['D_' + n])) for n in names))
    print('minimum over R (any run sizes):')
    for n in names:
        D = min(r['D_' + n] for r in rows)
        Rm = min(r['R'] for r in rows if r['D_' + n] <= D + 1e-9)
        b = dict(BUDGETS).get(n, 30.0)
        g, need = (vol, V_NEED) if n == 'ad10T0' else (info, I_NEED)
        eq = [r for r in rows if r['R'] * nb.cost(nb.bits_nonadaptive(r['R'], LOGK) if g is info else
                                                   nb.bits_adaptive_volume(r['R'], LOGK)) <= b]
        Re = eq[-1]['R']
        De = nb.bits_nonadaptive(Re, LOGK) if g is info else nb.bits_adaptive_volume(Re, LOGK)
        print('  %-7s budget %6.2f B0: D* = %.4f bits from R = %d runs on, B >= %.6f B0, X <= %.4f | equal runs: '
              'at most %d runs of %.4f bits, B = %.6f B0, T = %.3f T0, X = %.4f' % (
                  n, b, D, Rm, nb.cost(D), 1 / nb.cost(D), Re, De, nb.cost(De), Re * nb.cost(De) / 3,
                  1 / nb.cost(De)))
    print('wrote %s (R = 1..%d)' % (out, rmax))


if __name__ == '__main__':
    main()
