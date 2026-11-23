"""Baseline U1: one unsplit Shor run in Haener et al.'s configuration with plain signed windows.

harness/candidates/shor.py (Ekera's Shor for the ECDLP, arXiv 1905.09084 Sect. 2 and 6.1, B_ETA = B_DELTA = 100) with
two changes: (1) the defaults are Haener et al.'s exponent shape, a = m bits and b = m - 48 bits at m = 256
(NU_FRAC = 0.1875, the harness's "Haener config"; the 48 missing bits of b are found by one interval discrete log
per eta, 2^36.9 classical work at most per run) and ATTEMPTS = 4 runs per solve; (2) the top TOPW bits of a are a
separate add() call made first, so the run's first lookup reads the top of a and its controls are measured at once
(cost.first_window_held; harness/README.md, a register's top window first). Priced with
cap) this is 28 windowed additions + the port's 16-bit first lookup per run. Tables of Q multiples are read by
approximate additions, so the harness grades the run with its heuristic garbage-run model (4 x pf per addition,
the model behind Haener's per-attempt success and T0).
"""
SIGMA = 0
NU = 0
NU_FRAC = 0.1875        # NU = -round(NU_FRAC * m) = -48 at m = 256 (Haener et al. / Google); None: NU above
B_ETA = 100
B_DELTA = 100
ATTEMPTS = 4
TOPW = 16               # bits of a in the first lookup (the window size of the priced circuit)


def budget(m):
    return ATTEMPTS


def solve(h):
    m = h.r.bit_length()
    l = m + (NU if NU_FRAC is None else -round(NU_FRAC * m))
    ma = m + SIGMA
    top = min(TOPW, ma)
    Gtop = h.G * (1 << (ma - top))
    for _ in range(ATTEMPTS):
        run = h.run()
        a = run.reg(ma)
        b = run.reg(l)
        run.add(a, Gtop, ma - top, ma)          # top window first: the first lookup
        if ma > top:
            run.add(a, h.G, 0, ma - top)
        run.add(b, -h.Q)
        j, k = run.measure()
        d = postprocess(h, j, k, m, l)
        if d is not None:
            return d
    return None


def postprocess(h, j, k, m, l):
    r = h.r
    A, L = 1 << (m + SIGMA), 1 << l
    z = (2 * r * j + A) // (2 * A)                  # round(r j / 2^(m+SIGMA))
    T0 = -((2 * r * k + L) // (2 * L))              # round(-r k / 2^l)
    R = -((-r * (2 * B_DELTA + 1)) // (2 * L))      # ceil(r (B_DELTA + 1/2) / 2^l)
    for i in range(2 * B_ETA + 1):
        eta = (i + 1) // 2 * (1 if i % 2 else -1)   # 0, 1, -1, 2, -2, ...
        c = (z + eta) % r
        if c == 0:
            continue
        inv = pow(c, -1, r)
        T = h.dlog_interval(inv * h.G, h.Q, T0 - R, T0 + R)
        if T is not None:
            return T * inv % r                      # [T inv] G = Q exactly
    return None
