"""n = 256 numbers of Section III (Table I). secp256k1 group order r.
Non-adaptive (Theorem 1 + Fano): R runs of size c*r, list of K candidates, success 0.99:
    R*log2(1+c) >= 0.99*(log2 r - log2 K) - 1.
Adaptive (Theorem 2, volume bound): success <= K/r + R*c  =>  c >= (0.99 - K/r)/R.
Cost model (estimate grade, fractional windows): baseline B0 = 1,457 qubits x 39,255,168 Toffoli = 5.7195e10 (the exact figure of
harness/README.md); a first direct lookup of 2^18 Toffoli, then 28 windowed additions of 16 bits each cost the rest, so one window is
(39,255,168 - 2^18)/28 Toffoli at 1,457 qubits; the first 16 bits of the exponent are the direct lookup. (Before 2026-11-13 this file used
the older 1,450 x 40.42M = 5.91e10; the ratios B/B0 move by < 0.3 %.)"""
import math, sys
r = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
lr = math.log2(r)
win = (39_255_168 - 2**18) / 28 * 1457
B0 = 1457 * 39_255_168
def bits_nonadaptive(R, logK, succ=0.99):
    need = succ * (lr - logK) - 1.0
    if need <= 0: return None
    per = need / R
    c = 2 ** per - 1
    return lr + math.log2(c)
def bits_adaptive_volume(R, logK, succ=0.99):
    c = (succ - 2.0 ** (logK - lr)) / R
    return lr + math.log2(c)
def cost(bits):  # fractional windows, first 16 bits by direct lookup
    return max(bits / 16 - 1, 0) * win / B0
print("log2 r = %.6f, win cost %.4e q*Toffoli, B0 %.3e" % (lr, win, B0))
print("%5s %6s | %10s %8s | %10s %8s | %7s" % ("R", "logK", "nonadapt", "B/B0", "adaptive>=", "B/B0", "gap bits"))
for logK in (80, 64, 0):
    for R in (4, 8, 16, 32, 64, 128, 240, 1000):
        a = bits_nonadaptive(R, logK); b = bits_adaptive_volume(R, logK)
        print("%5d %6d | %10.2f %8.3f | %10.2f %8.3f | %7.2f" % (R, logK, a, cost(a), b, cost(b), a - b))
