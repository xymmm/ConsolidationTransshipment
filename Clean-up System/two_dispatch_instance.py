"""
two_dispatch_instance.py — search for, and report, an instance in which the
optimal policy dispatches at most twice
==========================================================================

Scope: Cf > 0 only; at Cf = 0 a warning is issued, see warn_if_cf0.

Criterion: from (I2_0, b1 = 0) at tau = T, under the optimal policy of
solver.py, P(3 or more dispatches) <= tol, and P(exactly 2) is as large as
possible. The law of the number of dispatches is computed exactly by forward
propagation on the same chain (no simulation). The closed-loop values V^(k)
with at most k dispatches confirm the instance: V^(2) = V* to solver accuracy.

Result with tol = 1e-4 (meeting parameters, Cf varied):
    T=5, lam1=5, lam2=3, h=1, cu=1, pi1=6, pi2=8, Cf=35, start (23, 0)
    P(0)=0.0001, P(1)=0.2628, P(2)=0.7370, P(>=3)=6e-5, V* = 294.1720
    V^(0..4) = 451.78, 302.09, 294.1722, 294.172, 294.172
"""
import numpy as np
from solver import Params
from transship_core import Chain, warn_if_cf0

BASE = dict(T=5.0, N=400, lam1=5, lam2=3, h=1, cu=1, pi1=6, pi2=8,
            c1=0, c2=0, v2=0, I2_max=40, I2_min=-31, b1_max=140)


def search(Cf_list=(25, 30, 35, 40, 50, 60), I0_range=range(10, 39), tol=1e-4,
           **over):
    out = []
    for Cf in Cf_list:
        p = Params(**{**BASE, **over, "Cf": Cf})
        warn_if_cf0(p, "search")
        ch = Chain(p)
        V, pol = ch.solve()
        best = None
        for I0 in I0_range:
            pm = ch.forward(pol, I0, 0, cmax=5)["count_pmf"]
            if pm[3:].sum() <= tol and (best is None or pm[2] > best[1][2]):
                best = (I0, pm, float(V[p.N, I0 - p.I2_min, 0]))
        if best:
            I0, pm, v = best
            out.append(dict(Cf=Cf, I2_0=I0, P0=pm[0], P1=pm[1], P2=pm[2],
                            P3plus=pm[3:].sum(), V_star=v))
            print(f"Cf={Cf}: I2_0={I0}  P0={pm[0]:.4f} P1={pm[1]:.4f} "
                  f"P2={pm[2]:.4f} P>=3={pm[3:].sum():.1e}  V*={v:.4f}")
    return out


def report(Cf=35, I2_0=23, kmax=4, **over):
    p = Params(**{**BASE, **over, "Cf": Cf})
    warn_if_cf0(p, "report")
    ch = Chain(p)
    V, pol = ch.solve()
    i0 = I2_0 - p.I2_min
    pm = ch.forward(pol, I2_0, 0, cmax=5)["count_pmf"]
    Vk, _ = ch.solve_k_limited(kmax)
    print(f"V* = {V[p.N, i0, 0]:.4f}   law of #dispatches: {np.round(pm, 5)}")
    print("V^(k), k = 0..%d:" % kmax,
          [round(float(Vk[k][p.N, i0, 0]), 4) for k in range(kmax + 1)])


if __name__ == "__main__":
    search()
    report()
