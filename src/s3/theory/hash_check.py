"""Exact toy checks for hash runs (paper App. A). Bits everywhere. Point group Z_r, f_d(a,b) = a + b d.

Test 1 (one run): state of the exponent register after compute-hash-uncompute, hash register discarded,
  rho_{d,H} = sum_y |Phi_y><Phi_y|,  Phi_y = sum_{z: H(f_d(z)) = y} alpha_z |z>.
  (a) average over a hidden uniform H equals (1-q) rho_d + q |psi><psi|  (q = 2^-b), checked by full enumeration on a small instance;
  (b) Holevo chi of the H-averaged ensemble <= (1-q) log2(1 + (N-1)/r);
  (c) a FIXED H known to the algorithm: chi over many draws of H, to show the excess o1 found (it is a known-H effect).
Test 2 (two runs sharing one hidden H, enumerated exactly): chi of {E_H rho^1_{d,H} (x) rho^2_{d,H}} against
  2 (1-q) log2(1 + (N-1)/r) and against the same two runs with independent hashes.
usage: python3 hash_check.py [test]  with test in {1, 2}"""
import sys, itertools
import numpy as np

def S(rho):
    w = np.linalg.eigvalsh(rho); w = w[w > 1e-12]; return float(-(w * np.log2(w)).sum())

def holevo(rhos):
    return S(sum(rhos) / len(rhos)) - sum(S(x) for x in rhos) / len(rhos)

def box(A, B, shift=(0, 0)):
    return np.array([(a + shift[0], b + shift[1]) for a in range(A) for b in range(B)])

def points(Z, d, r): return (Z[:, 0] + Z[:, 1] * d) % r

def rho_raw(Z, alpha, d, r):
    pts = points(Z, d, r); N = len(Z); rho = np.zeros((N, N), complex)
    for P in set(pts.tolist()):
        v = np.zeros(N, complex); i = np.where(pts == P)[0]; v[i] = alpha[i]; rho += np.outer(v, v.conj())
    return rho

def rho_hash(Z, alpha, d, r, H):   # H: array of length r with labels
    lab = H[points(Z, d, r)]; N = len(Z); rho = np.zeros((N, N), complex)
    for y in set(lab.tolist()):
        v = np.zeros(N, complex); i = np.where(lab == y)[0]; v[i] = alpha[i]; rho += np.outer(v, v.conj())
    return rho

def rho_hash_closed(Z, alpha, d, r, q):
    psi = alpha.astype(complex)
    return (1 - q) * rho_raw(Z, alpha, d, r) + q * np.outer(psi, psi.conj())

def test1():
    rng = np.random.default_rng(5)
    # (a) full enumeration of H on the image points, small instance
    r = 13; Z = box(3, 2); N = len(Z); alpha = rng.normal(size=N) + 1j * rng.normal(size=N); alpha /= np.linalg.norm(alpha)
    for b in (1, 2):
        q = 2.0 ** -b; err = 0.0
        for d in range(r):
            pts = sorted(set(points(Z, d, r).tolist())); acc = np.zeros((N, N), complex); cnt = 0
            for labs in itertools.product(range(2 ** b), repeat=len(pts)):
                H = np.zeros(r, int)
                for P, y in zip(pts, labs): H[P] = y
                acc += rho_hash(Z, alpha, d, r, H); cnt += 1
            err = max(err, np.abs(acc / cnt - rho_hash_closed(Z, alpha, d, r, q)).max())
        print("(a) b=%d r=%d N=%d: max |E_H rho - ((1-q) rho_d + q psi psi^+)| over all d = %.2e" % (b, r, N, err))
    # (b), (c)
    print("(b),(c) chi in bits; bound = (1-q) log2(1+(N-1)/r)")
    for (r, A, B) in [(101, 8, 8), (103, 10, 10), (211, 12, 12), (211, 20, 20)]:
        Z = box(A, B); N = len(Z); alpha = np.ones(N, complex) / np.sqrt(N)
        raw = holevo([rho_raw(Z, alpha, d, r) for d in range(r)])
        thm = np.log2(1 + (N - 1) / r)
        print("r=%d box %dx%d N=%d c=%.2f  raw chi=%.3f  Theorem-1 bound=%.3f" % (r, A, B, N, N / r, raw, thm))
        for b in (1, 2, 4):
            q = 2.0 ** -b
            avg = holevo([rho_hash_closed(Z, alpha, d, r, q) for d in range(r)])
            fixed = []
            for _ in range(8):
                H = rng.integers(0, 2 ** b, size=r)
                fixed.append(holevo([rho_hash(Z, alpha, d, r, H) for d in range(r)]))
            print("   b=%d: hidden-H chi = %.3f (bound %.3f, ok=%s) | fixed known H: mean %.3f max %.3f (x%.1f of Thm-1 bound)"
                  % (b, avg, (1 - q) * thm, avg <= (1 - q) * thm + 1e-9, np.mean(fixed), max(fixed), max(fixed) / thm))

def joint_enum(Z1, Z2, a1, a2, d, r, b):
    """E_H rho^1_{d,H} (x) rho^2_{d,H}, one shared hidden uniform H, by full enumeration over the image points."""
    n1, n2 = len(Z1), len(Z2)
    pts = sorted(set(points(Z1, d, r).tolist()) | set(points(Z2, d, r).tolist()))
    acc = np.zeros((n1 * n2, n1 * n2), complex); cnt = 0
    for labs in itertools.product(range(2 ** b), repeat=len(pts)):
        H = np.zeros(r, int)
        for P, y in zip(pts, labs): H[P] = y
        acc += np.kron(rho_hash(Z1, a1, d, r, H), rho_hash(Z2, a2, d, r, H)); cnt += 1
    return acc / cnt

def joint(Z1, Z2, a1, a2, d, r, q):
    """Same state in closed form: (a1 a1^+ (x) a2 a2^+) o K, K[(z1,z2),(z1',z2')] = Pr_H[H(P1)=H(P1'), H(P2)=H(P2')]
    = q^(number of distinct non-loop edges among {P1,P1'}, {P2,P2'})  (a spanning forest of the equalities)."""
    return joint_pts(points(Z1, d, r), points(Z2, d, r), a1, a2, q)

def joint_pts(p1, p2, a1, a2, q):
    P1, P1p = p1[:, None, None, None], p1[None, None, :, None]       # axes (z1, z2, z1', z2')
    P2, P2p = p2[None, :, None, None], p2[None, None, None, :]
    loop1, loop2 = (P1 == P1p), (P2 == P2p)
    same_edge = ((P1 == P2) & (P1p == P2p)) | ((P1 == P2p) & (P1p == P2))
    nedges = np.where(loop1, 0, 1) + np.where(loop2, 0, 1) - np.where(~loop1 & ~loop2 & same_edge, 1, 0)
    K = (q ** nedges).astype(float)
    n1, n2 = len(p1), len(p2)
    v = np.kron(a1, a2); base = np.outer(v, v.conj())
    return base * K.reshape(n1 * n2, n1 * n2)

def test2():
    rng = np.random.default_rng(7)
    # closed form against enumeration
    r = 13; Z1, Z2 = box(2, 2), box(2, 2, (1, 1))
    a1 = np.ones(4, complex) / 2; a2 = np.ones(4, complex) / 2
    for b in (1, 2):
        e = max(np.abs(joint_enum(Z1, Z2, a1, a2, d, r, b) - joint(Z1, Z2, a1, a2, d, r, 2.0 ** -b)).max() for d in range(r))
        print("closed form vs enumeration (r=13, 2x2 boxes shifted by (1,1), b=%d): max abs diff %.2e" % (b, e))
    print("two runs sharing ONE hidden hash; chi in bits; additive bound = 2 (1-q) log2(1+(N-1)/r)")
    cases = [("same 4x4 box", 61, box(4, 4), box(4, 4)), ("4x4 and shifted (2,1)", 61, box(4, 4), box(4, 4, (2, 1))),
             ("same 6x3 box", 61, box(6, 3), box(6, 3)), ("same 5x5 box", 101, box(5, 5), box(5, 5)),
             ("same 7x4 box", 53, box(7, 4), box(7, 4)), ("4x4 and 3x6", 61, box(4, 4), box(3, 6, (1, 0)))]
    for name, r, Z1, Z2 in cases:
        a1 = np.ones(len(Z1), complex) / np.sqrt(len(Z1)); a2 = np.ones(len(Z2), complex) / np.sqrt(len(Z2))
        raw2 = holevo([np.kron(rho_raw(Z1, a1, d, r), rho_raw(Z2, a2, d, r)) for d in range(r)])
        for b in (1, 2, 4):
            q = 2.0 ** -b
            shared = holevo([joint(Z1, Z2, a1, a2, d, r, q) for d in range(r)])
            indep = holevo([np.kron(rho_hash_closed(Z1, a1, d, r, q), rho_hash_closed(Z2, a2, d, r, q)) for d in range(r)])
            bd = (1 - q) * (np.log2(1 + (len(Z1) - 1) / r) + np.log2(1 + (len(Z2) - 1) / r))
            print("%-24s r=%d N=%d/%d b=%d: shared-H chi=%.4f  independent-H chi=%.4f  no-hash chi=%.4f  additive bound %.4f  shared<=bound: %s"
                  % (name, r, len(Z1), len(Z2), b, shared, indep, raw2, bd, shared <= bd + 1e-9), flush=True)

def test3():
    """E_d TV(real joint state, null joint state) against the proved volume bound sqrt(2 R (|U|-1) / r), R = 2 runs, hidden shared H.
    Null = every class its own point (labels fresh per class; classes shared between the runs keep one label)."""
    print("E_d TV(real, null), two runs, shared hidden H; bound = sqrt(2 R (|U|-1)/r), R = 2")
    for name, r, Z1, Z2 in [("same 2x4 box", 1009, box(2, 4), box(2, 4)), ("same 3x3 box", 1009, box(3, 3), box(3, 3)),
                            ("3x3 and shifted (1,1)", 1009, box(3, 3), box(3, 3, (1, 1))), ("same 3x3 box", 211, box(3, 3), box(3, 3)),
                            ("same 4x4 box", 1009, box(4, 4), box(4, 4))]:
        U = sorted({tuple(z) for z in Z1.tolist()} | {tuple(z) for z in Z2.tolist()}); idx = {z: i for i, z in enumerate(U)}
        n0 = np.array([idx[tuple(z)] for z in Z1.tolist()]); n1 = np.array([idx[tuple(z)] for z in Z2.tolist()])
        a1 = np.ones(len(Z1), complex) / np.sqrt(len(Z1)); a2 = np.ones(len(Z2), complex) / np.sqrt(len(Z2))
        for b in (1, 2, 4):
            q = 2.0 ** -b
            null = joint_pts(n0, n1, a1, a2, q); tv = []; viol = 0; Ws = []
            Uarr = np.array(U)
            for d in range(r):
                diff = joint(Z1, Z2, a1, a2, d, r, q) - null
                t_d = 0.5 * np.abs(np.linalg.eigvalsh(diff)).sum(); tv.append(t_d)
                pt = (Uarr[:, 0] + Uarr[:, 1] * d) % r
                nonfirst = np.array([pt[i] in set(pt[:i].tolist()) for i in range(len(U))])
                W = sum(abs(a1[k]) ** 2 for k in range(len(Z1)) if nonfirst[n0[k]]) + sum(abs(a2[k]) ** 2 for k in range(len(Z2)) if nonfirst[n1[k]])
                Ws.append(W); viol += t_d > np.sqrt(2 * W) + 1e-9
            bd = np.sqrt(2 * 2 * (len(U) - 1) / r)
            print("%-24s r=%d |U|=%d b=%d: E_d TV = %.4f  (max_d %.4f)  bound %.4f  ok=%s | pointwise TV_d <= sqrt(2 W_d): %d violations of %d; E_d W = %.4f <= R(|U|-1)/r = %.4f"
                  % (name, r, len(U), b, np.mean(tv), max(tv), bd, np.mean(tv) <= bd, viol, r, np.mean(Ws), 2 * (len(U) - 1) / r))


def classical_blocks(pc, pq, alpha, b):
    """For one d: dict c -> E_H[ 1[(H(pc_i))_i = c] rho_H ], rho_H = hash-run state of the quantum shape (points pq, amplitudes alpha),
    by full enumeration of H over the distinct points of pc and pq (b bits)."""
    pts = sorted(set(pc.tolist()) | set(pq.tolist())); idx = {p: i for i, p in enumerate(pts)}
    N = len(pq); out = {}; tot = 0
    for labs in itertools.product(range(2 ** b), repeat=len(pts)):
        lab_q = np.array([labs[idx[p]] for p in pq.tolist()]); c = tuple(labs[idx[p]] for p in pc.tolist())
        E = (lab_q[:, None] == lab_q[None, :]).astype(float)
        rho = np.outer(alpha, alpha.conj()) * E
        out[c] = out.get(c, 0) + rho; tot += 1
    return {c: v / tot for c, v in out.items()}

def test4():
    """One hash run + W classical evaluations fixed in advance (classical points first in the order): pointwise
    TV_d <= 1[two classical points collide at d] + sqrt(2 W_d), W_d = weight of quantum classes that collide with a classical point or an earlier quantum class."""
    rng = np.random.default_rng(3)
    Zq = box(2, 3); N = len(Zq); alpha = np.ones(N, complex) / np.sqrt(N)
    Zc = np.array([(3, 1), (0, 4), (2, 5)])
    for r in (31, 61):
        for b in (1, 2):
            if b == 2 and len(Zc) + N > 8: Zc_use = Zc[:1]
            else: Zc_use = Zc[:2]
            viol = 0; worst = 0.0; tvmax = 0.0
            for d in range(r):
                pq = points(Zq, d, r); pc = points(Zc_use, d, r)
                real = classical_blocks(pc, pq, alpha, b)
                # null: every class is its own point (classical ids 0.., quantum ids after)
                nc = np.arange(len(Zc_use)); nq = len(Zc_use) + np.arange(N)
                null = classical_blocks(nc, nq, alpha, b)
                tv = 0.5 * sum(np.abs(np.linalg.eigvalsh(real.get(c, 0 * np.eye(N)) - null.get(c, 0 * np.eye(N)))).sum() for c in set(real) | set(null))
                EC = len(set(pc.tolist())) < len(pc)
                seen = set(pc.tolist()); W = 0.0
                for k in range(N):
                    if pq[k] in seen: W += abs(alpha[k]) ** 2
                    seen.add(pq[k])
                bound = (1.0 if EC else 0.0) + np.sqrt(2 * W)
                tvmax = max(tvmax, tv)
                if tv > bound + 1e-9: viol += 1; worst = max(worst, tv - bound)
            print("r=%d b=%d W=%d classical points, quantum 2x3 box: max_d TV = %.4f, pointwise TV_d <= 1[E_C] + sqrt(2 W_d): %d violations of %d (worst excess %.2e)"
                  % (r, b, len(Zc_use), tvmax, viol, r, worst), flush=True)

if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "1"
    {"1": test1, "2": test2, "3": test3, "4": test4}[t]()
