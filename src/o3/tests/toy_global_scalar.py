"""o3: exact toy check of Lemma 1-zero (one random global scalar c per run plus the first-window offset R0).

Claim (paper Sec. V.D and App. B). Scale EVERY table of the run (the exact first lookup and every approximate window) by one
uniform c in Z_r^*, and load the first lookup as [c] T_1[x_1] + R0 with R0 uniform. Then along every input's path
the operand pair of every addition is (A_w, B_w) = (R0 + [c] S_w(x), [c] T_w[x_w]), uniform on E x (E minus O),
because R0 is independent of c and every table point is nonzero. [c] commutes with every unit +-lambda^j, so hex
tables (orbit representatives plus a unit applied by the circuit) are allowed. So every addition misbehaves on every
input with the SAME first-order probability, the one measured on uniform operand pairs, and the amplitude averaged
over (R0, c, records) is g0 A_I + (second-order terms), with

    g0 = 1 - sum_w (fD_w + fO_w + 2 fP_w),
    fD = dead (flagged / off-curve / dirty), fO = on-curve wrong point, fP = phase-only (right point, sign -1),

and the bound P_R(no flag, o, u' | rho) >= (1/r) (g0 kappa(o, u') - e0)_+^2, u' = u c (the accumulator's character
relative to [c] G). e0 collects only the x-dependent remainder: sum_x |a_x| 2^(-n/2) (|E g_x - g0| + P(x ends alive
at a wrong point)); the certifiable form replaces |E g_x - g0| by E[(N_D + N_O + 2 N_P) 1[N >= 2]] on x's correct path.

Toy: Z_r (points = labels), b-register (2 qubits) and a's low 2 qubits in the exact first lookup, J windows of w qubits
with signed odd digits (never the identity) as approximate additions. Each addition has a fixed class table on
(A, B, record bit): good / dead / phase / on-curve wrong (output + 1), in three adversarial settings: failures tied
to the addend B (what fixed tables expose), tied to the accumulator A (correlated along a path), and record-dependent
mixtures. Every (R0, c, records) is enumerated, for several rho; the variant "c = 1" (fixed tables, R0 only) shows
what the scalar removes.
Run: python3 o3/tests/toy_global_scalar.py [seed] [r] [J] [w]
"""
import math
import sys

import numpy as np


def classes_table(rng, r, setting, w_idx):
    """Class of an addition on (A, B, m): 0 good, 1 dead, 2 phase-only, 3 on-curve wrong (output + 1)."""
    F = np.zeros((r, r, 2), dtype=np.int8)
    if setting == 'B':                       # failures tied to the addend: some table points are bad
        SB = rng.choice(np.arange(1, r), size=max(2, r // 5), replace=False)
        u = rng.random((r, r, 2))
        badB = np.zeros(r, bool)
        badB[SB] = True
        pdead = np.where(badB[None, :, None], 0.45, 0.01)
        F[u < pdead] = 1
        u2 = rng.random((r, r, 2))
        F[(F == 0) & (u2 < 0.03)] = 2
        F[(F == 0) & (u2 > 0.985)] = 3
    elif setting == 'A':                     # failures tied to the accumulator: correlated along a path
        SA = rng.choice(r, size=max(2, r // 8), replace=False)
        SA2 = rng.choice(r, size=max(1, r // 16), replace=False)
        F[SA, :, :] = 1
        F[SA2, :, :] = np.where(F[SA2, :, :] == 0, 2, F[SA2, :, :])
        u = rng.random((r, r, 2))
        F[(F == 0) & (u < 0.02)] = 3
    else:                                    # record-dependent mixture
        u = rng.random((r, r, 2))
        F[u < 0.06] = 1
        F[(u >= 0.06) & (u < 0.09)] = 2
        F[(u >= 0.09) & (u < 0.11)] = 3
        F[:, :, 1] = np.where(rng.random((r, r)) < 0.5, F[:, :, 1], 0)
    F[:, 0, :] = 1                           # addend O is exceptional (never read: digits are nonzero)
    return F


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    r = int(sys.argv[2]) if len(sys.argv) > 2 else 31
    J = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    w = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    rng = np.random.default_rng(seed)
    nb, na0 = 2, 2
    na = na0 + w * J
    Na, Nb = 1 << na, 1 << nb
    N = Na * Nb
    xs = np.arange(N)
    a = xs // Nb
    b = xs % Nb
    a0 = a & ((1 << na0) - 1)
    digit = [((a >> (na0 + w * i)) & ((1 << w) - 1)) for i in range(J)]
    shift = [pow(2, na0 + w * i, r) for i in range(J)]
    Dw = [((2 * digit[i] - ((1 << w) - 1)) * shift[i]) % r for i in range(J)]     # signed odd digit, never 0
    assert all((D != 0).all() for D in Dw)
    amp = np.full(N, 1 / math.sqrt(N))
    Lam = np.abs(amp).sum() / math.sqrt(N)
    omega = np.exp(2j * np.pi / r)
    d = int(rng.integers(1, r))
    settings = ['B', 'A', 'M']
    tables = {s: [classes_table(rng, r, s, i) for i in range(J)] for s in settings}
    rhos = [0, 1 + int(rng.integers(0, r - 1)), 1 + int(rng.integers(0, r - 1))]
    print('toy: r = %d, d = %d, J = %d windows of %d qubits, b %d + a0 %d qubits in the exact first lookup, '
          'N = %d inputs, seed %d' % (r, d, J, w, nb, na0, N, seed))
    print('rows: setting, variant, rho | g0, e0 exact, e0 cert | min(P_R - bound) | bad mass: best coupling, '
          'Lemma 1-0, plain rule | P(flag), unknown-bad nu | E[N_D|x] spread')
    for s in settings:
        F = tables[s]
        # uniform-pair rates per window: A uniform on Z_r, B uniform on Z_r^*, m uniform
        f = np.zeros((J, 4))
        for i in range(J):
            cnt = np.bincount(F[i][:, 1:, :].ravel(), minlength=4)
            f[i] = cnt / cnt.sum()
        g0 = 1 - (f[:, 1] + f[:, 3] + 2 * f[:, 2]).sum()
        e_plain = (f[:, 1] + 2 * (f[:, 2] + f[:, 3])).sum()
        for variant in ['global c', 'c = 1']:
            cs = np.arange(1, r) if variant == 'global c' else np.array([1])
            for rho in rhos:
                t = (-(d + rho)) % r
                L = (a0 + sum(Dw) + b * t) % r                                    # ideal exponent map
                T1 = (a0 + b * t) % r
                # ideal law: P_I(o, u') for all u'
                ph = omega ** ((np.arange(r)[:, None] * L[None, :]) % r)          # [u', x]
                AI = np.fft.fft2((amp[None, :] * ph).reshape(r, Na, Nb), axes=(1, 2)).reshape(r, N) / math.sqrt(N)
                PI = (np.abs(AI) ** 2 / r).T                                         # [o, u']
                kap = np.sqrt(r * PI)
                PR = np.zeros((N, r))
                Eg = np.zeros(N)
                Pwrong = np.zeros(N)
                Ecert = np.zeros(N)
                END = np.zeros(N)
                Palive = 0.0
                nconf = 0
                R0 = np.arange(r)[:, None]
                for c in cs:
                    B = [(c * Dw[i]) % r for i in range(J)]
                    pref = [(c * T1) % r]
                    for i in range(J - 1):
                        pref.append((pref[-1] + B[i]) % r)
                    for mi in range(1 << J):
                        m = [(mi >> i) & 1 for i in range(J)]
                        G = (R0 + pref[0][None, :]) % r                           # [R0, x]
                        alive = np.ones((r, N), bool)
                        sign = np.ones((r, N))
                        wrong = np.zeros((r, N), bool)
                        nD = np.zeros((r, N))
                        nO = np.zeros((r, N))
                        nP = np.zeros((r, N))
                        for i in range(J):
                            # real evolution
                            cl = F[i][G, B[i][None, :], m[i]]
                            Gn = (G + B[i][None, :]) % r
                            alive &= cl != 1
                            sign = np.where(cl == 2, -sign, sign)
                            wrong |= cl == 3
                            G = np.where(cl == 3, (Gn + 1) % r, Gn)
                            # correct-path classes (operands of the ideal path)
                            Ac = (R0 + pref[i][None, :]) % r
                            cc = F[i][Ac, B[i][None, :], m[i]]
                            nD += cc == 1
                            nO += cc == 3
                            nP += cc == 2
                        good = alive & ~wrong
                        Eg += (np.where(good, sign, 0.0)).sum(0)
                        Pwrong += (alive & wrong).sum(0)
                        Ntot = nD + nO + nP
                        Ecert += ((nD + nO + 2 * nP) * (Ntot >= 2)).sum(0)
                        END += nD.sum(0)
                        Palive += (alive * np.abs(amp)[None, :] ** 2).sum()
                        # amplitudes for every (R0, u): sum_x amp s alive w^(u G)
                        v = np.where(alive, sign, 0.0)[:, None, :] * amp[None, None, :] * \
                            omega ** ((np.arange(r)[None, :, None] * G[:, None, :]) % r)
                        Aq = np.fft.fft2(v.reshape(r, r, Na, Nb), axes=(2, 3)).reshape(r, r, N) / math.sqrt(N)
                        P = (np.abs(Aq) ** 2).sum(0) / r                               # [u, o], summed over R0
                        up = (np.arange(r) * c) % r                                        # u' = u c
                        PR[:, up] += P.T
                        nconf += r
                PR /= nconf
                Eg /= nconf
                Pwrong /= nconf
                Ecert /= nconf
                END /= nconf
                Palive /= nconf
                e0x = (np.abs(amp) / math.sqrt(N) * (np.abs(Eg - g0) + Pwrong)).sum()
                e0c = (np.abs(amp) / math.sqrt(N) * (Ecert + Pwrong)).sum()
                bound = np.maximum(g0 * kap - e0c, 0) ** 2 / r
                slack = (PR - bound).min()
                bound_x = np.maximum(g0 * kap - e0x, 0) ** 2 / r
                slack_x = (PR - bound_x).min()
                with np.errstate(divide='ignore', invalid='ignore'):
                    K0 = np.where(kap > 0, np.maximum(g0 - e0c / kap, 0) ** 2, 0)
                    Kp = np.where(kap > 0, np.maximum(1 - e_plain / kap, 0) ** 2, 0)
                PGood = (PI * K0).sum()
                PI_o = PI.sum(1)
                PR_o = PR.sum(1)
                best = 1 - np.minimum(PI_o, PR_o).sum()
                print('%s  %-8s rho %2d | g0 %.4f  e0 %.4f / %.4f | slack %+.2e (exact e0 %+.2e) | best %.3f  L1-0 %.3f '
                      ' plain %.3f | flag %.3f  nu %.3f | E[N_D|x] %.4f..%.4f (J fD %.4f)'
                      % (s, variant, rho, g0, e0x, e0c, slack, slack_x, best, 1 - PGood, 1 - (PI * Kp).sum(),
                         1 - Palive, Palive - PGood, END.min(), END.max(), f[:, 1].sum()))


if __name__ == '__main__':
    main()
