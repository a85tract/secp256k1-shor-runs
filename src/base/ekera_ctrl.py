"""Third control: Ekera's tradeoff algorithm, plainly decoded, on U1's plain signed windows.

harness/candidates/ekera_tradeoff.py copied (the harness's source scan allows no imports of other files) with its
algorithm unchanged: Ekera arXiv 1905.09084 Sect. 2 and 6.2, thesis KTH 2024 Tabs. 5.4 / 5.5 (m = 256: tradeoff
factor s -> n runs and sigma). Each run: a in [0, 2^(m+sigma)), b in [0, 2^l), l = ceil(m/s); the accumulator receives
[a]G - [b]Q; both registers are Fourier-measured. Post-processing: one LLL and one Babai call on the lattice L0 of all n
runs, then the check, as in the thesis's Tab. 5.4 simulations. Plain decoding: an attempt is a fresh batch of n runs and
one decode; a failed attempt is followed by a new batch (ATTEMPTS batches at most); no subsets, flags or certificate.
One change, as in base/u1_shor.py: the top TOPW bits of a are a separate add() call made first, so the run's first
lookup reads the top of a and its controls are measured at once (cost.first_window_held). Priced with
built from Q, so the harness grades every run with its garbage-run model (4 pf per addition), as U1.
"""
import math

S = 4
SIGMA = None        # default: Tab. 5.5
RUNS = None         # default: Tab. 5.4 at m >= 256, Gaussian-heuristic count + 3 below
ATTEMPTS = 12
TOPW = 16

_TAB_RUNS_256 = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 9, 8: 11, 10: 15}   # Ekera's simulated values
_TAB_SIGMA_256 = {1: 7, 2: 8, 3: 8, 4: 8, 5: 9, 6: 9, 7: 9, 8: 9, 10: 10}


def params(m):
    sigma = SIGMA if SIGMA is not None else _TAB_SIGMA_256.get(S, 10)
    if RUNS is not None:
        n = RUNS
    elif m >= 256 and S in _TAB_RUNS_256:
        n = _TAB_RUNS_256[S]
    elif S == 8:
        n = 17
    else:
        c = math.log2(math.sqrt(2 * math.pi * math.e)) * S / m
        n = math.ceil((S + c) / (1 - c)) + 3 if c < 1 else 10 * S
    return sigma, n


def budget(m):
    return ATTEMPTS * params(m)[1]


def solve(h):
    m = h.r.bit_length()
    l = -(-m // S)
    sigma, n = params(m)
    ma = m + sigma
    top = min(TOPW, ma)
    Gtop = h.G * (1 << (ma - top))
    for _ in range(ATTEMPTS):
        pairs = []
        for _ in range(n):
            run = h.run()
            a = run.reg(ma)
            b = run.reg(l)
            run.add(a, Gtop, ma - top, ma)      # top window first: the first lookup
            if ma > top:
                run.add(a, h.G, 0, ma - top)
            run.add(b, -h.Q)
            pairs.append(run.measure())
        d = postprocess(h, pairs, m, l, sigma)
        if d is not None:
            return d
    return None


def postprocess(h, pairs, m, l, sigma):
    r = h.r
    n = len(pairs)
    A = 1 << (m + sigma)
    big = 1 << (m + l)
    zs = [(2 * r * j + A) // (2 * A) for j, _ in pairs]
    rows = [[big * z for z in zs] + [r]]
    for i in range(n):
        rows.append([0] * i + [r * big] + [0] * (n - i))
    target = [-(r * k << m) for _, k in pairs] + [0]
    red = h.lll(rows)
    _, vec = h.babai(red, target)
    if vec[-1] % r:
        return None
    d = (vec[-1] // r) % r
    return d if h.check(d) else None
