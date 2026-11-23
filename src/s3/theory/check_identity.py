"""Numerical check of Steps 2-6 of Theorem 1 on random small instances (explicit density matrices).
For each instance: rho_d (dim |S|), sigma = class-dephased state, checks
  Tr[rho_d^2 sigma^-1] = 1 + x_d,  chi = S(rho_bar) - mean S(rho_d) <= mean_d D(rho_d||sigma) <= ln(1 + mean x_d) <= ln(1 + (N-1)/r)."""
import numpy as np
rng = np.random.default_rng(1)
def S(rho):
    w = np.linalg.eigvalsh(rho); w = w[w > 1e-12]; return float(-(w * np.log(w)).sum())
def relent(rho, sig):   # sig diagonal-in-class-basis, rank-full on supp
    # D(rho||sig) = Tr rho ln rho - Tr rho ln sig
    w, V = np.linalg.eigh(rho); m = w > 1e-12
    a = float((w[m] * np.log(w[m])).sum())
    ws, Vs = np.linalg.eigh(sig); ls = Vs @ np.diag(np.log(np.maximum(ws, 1e-300))) @ Vs.conj().T
    return a - float(np.real(np.trace(rho @ ls)))
worst = 0
for trial in range(300):
    r = int(rng.choice([11, 13, 17, 19, 23, 29, 31, 37]))
    k = int(rng.choice([2, 3]))
    N = int(rng.integers(4, 25))
    Z = rng.integers(0, 6, size=(N, k)); Z = np.unique(Z, axis=0); N = len(Z)
    u = rng.integers(0, r, size=k); v = rng.integers(0, r, size=k)
    alpha = rng.normal(size=N) + 1j * rng.normal(size=N); alpha /= np.linalg.norm(alpha)
    # trivial classes
    uz = (Z @ u) % r; vz = (Z @ v) % r
    cls = {}
    for i in range(N): cls.setdefault((uz[i], vz[i]), []).append(i)   # z ~ z' iff same (u.z, v.z) mod r
    classes = list(cls.values()); Nc = len(classes)
    sigma = np.zeros((N, N), complex)
    for c in classes:
        phi = np.zeros(N, complex); phi[c] = alpha[c]; sigma += np.outer(phi, phi.conj())
    chis = []; rhos = []; Ds = []; xs = []
    for d in range(r):
        pts = (uz + vz * d) % r
        rho = np.zeros((N, N), complex)
        for P in set(pts):
            Phi = np.zeros(N, complex); idx = np.where(pts == P)[0]; Phi[idx] = alpha[idx]
            rho += np.outer(Phi, Phi.conj())
        ws, Vs = np.linalg.eigh(sigma); mask = ws > 1e-12
        sinv = (Vs[:, mask] / ws[mask]) @ Vs[:, mask].conj().T
        tr = float(np.real(np.trace(rho @ rho @ sinv)))
        # x_d = sum_k w_k (n_k - 1), n_k = number of classes with same point as class k
        x = 0.0
        for c in classes:
            wk = float(np.sum(abs(alpha[c]) ** 2)); P = pts[c[0]]
            nk = sum(1 for c2 in classes if pts[c2[0]] == P); x += wk * (nk - 1)
        assert abs(tr - (1 + x)) < 1e-8, (tr, x)
        rhos.append(rho); Ds.append(relent(rho, sigma)); xs.append(x)
    rb = sum(rhos) / r
    chi = S(rb) - sum(S(x) for x in rhos) / r
    chain = [chi, float(np.mean(Ds)), float(np.log(1 + np.mean(xs))), float(np.log(1 + (Nc - 1) / r))]
    assert chain[0] <= chain[1] + 1e-9 <= chain[2] + 2e-9 <= chain[3] + 3e-9, chain
    worst = max(worst, chain[0] / chain[3])
print("300 random instances: identity and the chain chi <= E D <= ln(1+E x) <= ln(1+(N-1)/r) hold; max chi/bound = %.3f" % worst)
