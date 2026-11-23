"""Floors of Section 3 at n = 256 (secp256k1), recomputed for the paper.

Same formulas as s3/theory/numbers_n256.py (the primary file):
  Corollary 1 (R runs with shapes fixed in advance, Fano with a list of K candidates, success 0.99):
      R * log2(1 + c) >= 0.99 * (log2 r - log2 K) - 1,  bits per run = log2 r + log2 c.
  Theorem 2 (adaptive schedules, volume bound, d-averaged success):
      c >= (0.99 - K / r) / R.
  Cost model (estimate grade, the baseline's own arithmetic): the first 16 exponent bits by a direct lookup charged at
  zero, then fractional windows of 16 bits, each at the baseline's per-window cost (39,255,168 - 2^18) / 28 Toffoli at
  1,457 qubits; B0 = 1,457 x 39,255,168.
  Equal runs: T = R * B, T0 = 3 B0.

Run: python3 paper/figures/floors.py  (prints the numbers quoted in Section 3, Table 1 and Section 7.3).
"""
import math

R_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
LOG_R = math.log2(R_ORDER)
B0 = 1457 * 39_255_168
WIN = (39_255_168 - 2 ** 18) / 28 * 1457          # qubit x Toffoli of one 16-bit window at the baseline
LOG_K = 80                                        # 2^40 generic classical operations test at most about 2^80 values
SUCC = 0.99


def bits_nonadaptive(R, logK=LOG_K, succ=SUCC):
    need = succ * (LOG_R - logK) - 1.0
    c = 2 ** (need / R) - 1
    return LOG_R + math.log2(c)


def bits_adaptive(R, logK=LOG_K, succ=SUCC):
    c = (succ - 2.0 ** (logK - LOG_R)) / R
    return LOG_R + math.log2(c)


def cost(bits):
    """B / B0 of a run that keeps `bits` exponent bits, at the baseline's per-window cost."""
    return max(bits / 16 - 1, 0) * WIN / B0


def t_over_t0(R, b_over_b0):
    return R * b_over_b0 / 3.0


def max_runs(t_budget, bits_fn=bits_nonadaptive):
    """Largest R whose equal runs fit T <= t_budget * T0."""
    best = None
    for R in range(1, 2000):
        if t_over_t0(R, cost(bits_fn(R))) <= t_budget:
            best = R
    return best


if __name__ == "__main__":
    print("log2 r = %.6f, window = %.4e qubit x Toffoli, B0 = %.4e" % (LOG_R, WIN, B0))
    print("%5s | %9s %7s %7s | %9s %7s %7s" % ("R", "bits", "B/B0", "T/T0", "adapt", "B/B0", "T/T0"))
    for R in (4, 8, 16, 32, 55, 64, 128, 240):
        a, b = bits_nonadaptive(R), bits_adaptive(R)
        print("%5d | %9.2f %7.4f %7.2f | %9.2f %7.4f %7.2f" % (
            R, a, cost(a), t_over_t0(R, cost(a)), b, cost(b), t_over_t0(R, cost(b))))
    for budget in (10, 4):
        R = max_runs(budget)
        b = cost(bits_nonadaptive(R))
        Ra = max_runs(budget, bits_adaptive)
        ba = cost(bits_adaptive(Ra))
        print("T <= %d T0, equal runs: R <= %d, B >= %.4f B0 (%.3fx); adaptive: R <= %d, B >= %.4f B0 (%.3fx)" % (
            budget, R, b, 1 / b, Ra, ba, 1 / ba))
    for R, D in ((79, 270.0), (102, 267.88), (110, 267.88), (25, 288.0)):
        f = bits_nonadaptive(R)
        print("R = %3d: floor %.2f bits, configuration %.2f bits, gap %.2f bits" % (R, f, D, D - f))
