"""One-run information of the QFT measurement on a zero-padded register (exact, 2D box a < A, b < B, accumulator a + b d in Z_r).

The exponent registers hold values below A and B; the QFT is taken over Z_{Ma} x Z_{Mb} with Ma = 2^pa A, Mb = 2^pb B (pa, pb
extra measured output bits, input qubits that stay |0>; zero Toffoli).  P(j | d) = (1/(Ma Mb)) sum_{v in Lambda_d} Ca(v1) Cb(v2) w^(j.v),
Lambda_d = {(x, y): x + y d = 0 mod r, |x| < A, |y| < B}, C = autocorrelation of the unit-norm amplitude (separable).
Amplitude envelopes: 'u' uniform, 's<k>' sine^k.  Prints I(j; d) for uniform d, in bits, against the Theorem 1 bound.
usage: python3 oversample_info.py r A B pa,pb;pa,pb;... env_a env_b"""
import sys
import numpy as np

def autocorr(a):
    a = a / np.linalg.norm(a)
    return np.correlate(a, a, "full")          # index v + (len-1)

def env(kind, n):
    if kind == "u": return np.ones(n)
    k = int(kind[1:]); x = (np.arange(n) + 0.5) / n
    return np.sin(np.pi * x) ** k

def info(r, A, B, pa, pb, ea="u", eb="u"):
    Ma, Mb = A << pa, B << pb
    Ca, Cb = autocorr(env(ea, A)), autocorr(env(eb, B))
    M = np.zeros((r, Ma * Mb), dtype=np.float32)
    ys = np.arange(-(B - 1), B)
    for d in range(r):
        hist = np.zeros((Ma, Mb))
        x0 = (-ys * d) % r
        for k in range(-(A // r + 1), A // r + 2):
            x = x0 + r * k
            m = np.abs(x) < A
            if not m.any(): continue
            xv, yv = x[m], ys[m]
            np.add.at(hist, (xv % Ma, yv % Mb), Ca[xv + A - 1] * Cb[yv + B - 1])
        F = np.fft.fft2(hist).real.reshape(-1)
        M[d] = F / (Ma * Mb)
    M = np.clip(M, 0, None)
    assert np.allclose(M.sum(1, dtype=np.float64), 1.0, atol=1e-3), M.sum(1)[:3]
    Mbar = M.mean(0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = float(np.nansum(M * np.log2(np.where(M > 0, M / Mbar, 1.0)), dtype=np.float64)) / r
    return I

if __name__ == "__main__":
    r, A, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    pads = [tuple(int(t) for t in s.split(",")) for s in sys.argv[4].split(";")]
    ea, eb = sys.argv[5], sys.argv[6]
    c = A * B / r
    print("r=%d box %dx%d c=%.3f envelope a=%s b=%s; Theorem-1 bound log2(1+c)=%.3f" % (r, A, B, c, ea, eb, np.log2(1 + c)), flush=True)
    for pa, pb in pads:
        print("  pad bits (pa,pb)=(%d,%d): I = %.4f bits" % (pa, pb, info(r, A, B, pa, pb, ea, eb)), flush=True)
