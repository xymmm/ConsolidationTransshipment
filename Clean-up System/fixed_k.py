"""
fixed_k.py — optimal policy when the number of dispatches is fixed at K
=======================================================================

Model: solver.py, state (I2, b1, tau), a dispatch of q >= 1 units costs
Cf + cu*q, at most one dispatch per period. The time and the quantity of
every dispatch are decided from the realised state.

Layer k = number of dispatches still to be made:

    V_0^n = Q_0(V_0^{n-1})                                   never dispatch
    V_k^n = min{ Q_0(V_k^{n-1}),  min_{q>=1} Q_q(V_{k-1}^{n-1}) }
    V_k^0 = terminal + k * penalty                            unused dispatches

    exact=False  penalty = 0          at most K dispatches
    exact=True   penalty = Cf + M     exactly K dispatches; M = 1e6 makes the
                                      constraint binding wherever it can be met

With exact=True the fixed cost is paid K times on every path, so the policy
does not depend on Cf. The reported cost excludes the penalty; p_fewer is the
probability that fewer than K dispatches could be made.

Usage (set K outside):
    fk = FixedK(p, K=2, exact=True).solve()
    out = fk.law(K=2, I2_0=15)      # cost, law of number, time, size
"""

import numpy as np
from solver import Params
from transship_core import Chain


class FixedK:
    def __init__(self, p: Params, K: int, exact=True, M=1e6, chain=None):
        self.p, self.K, self.exact = p, int(K), bool(exact)
        self.ch = chain or Chain(p)
        self.penalty = float(p.Cf) + M if exact else 0.0

    def solve(self):
        ch, N = self.ch, self.p.N
        term = ch.terminal()
        V0 = np.empty((N + 1,) + ch.shape); V0[0] = term
        for n in range(1, N + 1):
            V0[n] = ch.Q(0, V0[n - 1])
        self.V, self.pol = [V0], [np.zeros((N + 1,) + ch.shape, np.int16)]
        for k in range(1, self.K + 1):
            prev = self.V[k - 1]
            Vk = np.empty_like(V0); pk = np.zeros_like(self.pol[0])
            Vk[0] = term + k * self.penalty
            for n in range(1, N + 1):
                w = ch.Q(0, Vk[n - 1])
                d, bq = ch.best_dispatch(prev[n - 1])
                disp = d < w
                Vk[n] = np.where(disp, d, w); pk[n] = np.where(disp, bq, 0)
            self.V.append(Vk); self.pol.append(pk)
        return self

    def law(self, K, I2_0, b1_0=0):
        """Exact forward law from (I2_0, b1_0) at tau = T, K <= self.K."""
        ch, p, N = self.ch, self.p, self.p.N
        nB = ch.shape[1]; size = ch.shape[0] * nB
        mass = np.zeros((K + 1, size))
        mass[0, int(np.clip(I2_0, p.I2_min, p.I2_max) - p.I2_min) * nB
             + int(np.clip(b1_0, 0, p.b1_max))] = 1.0
        when = np.zeros((max(K, 1), N + 1)); qty = np.zeros((max(K, 1), ch.qmax + 1))
        cost = 0.0
        for n in range(N, 0, -1):
            new = np.zeros_like(mass)
            for u in range(K + 1):                  # u dispatches made so far
                A = self.pol[K - u][n].ravel()
                for q in np.unique(A[mass[u] > 0]):
                    q = int(q)
                    c = np.nonzero((A == q) & (mass[u] > 0))[0]
                    m = mass[u, c]
                    cost += float(np.sum(m * ch.g[q].ravel()[c]))
                    if q > 0:
                        when[u, n] += m.sum(); qty[u, q] += m.sum()
                    u2 = u + (q > 0)
                    for P_, ii, jj in ((p.p0, ch.ii0, ch.jj0), (p.p1, ch.ii0, ch.jj1),
                                       (p.p2, ch.ii2, ch.jj0)):
                        dst = ii[q].ravel()[c] * nB + jj[q].ravel()[c]
                        new[u2] += np.bincount(dst, m * P_, size)
            mass = new
        cost += float(np.sum(mass.sum(axis=0) * ch.terminal().ravel()))
        t = p.T - np.arange(N + 1) * p.dt
        rows = [dict(dispatch=u + 1, prob=when[u].sum(),
                     mean_time=float(when[u] @ t) / when[u].sum() if when[u].sum() else np.nan,
                     mean_qty=float(qty[u] @ np.arange(ch.qmax + 1)) / when[u].sum()
                     if when[u].sum() else np.nan) for u in range(K)]
        pmf = mass.sum(axis=1)
        return dict(cost=cost, count_pmf=pmf, p_fewer=float(pmf[:K].sum()),
                    dispatches=rows, time_law=when[:K], t_grid=t, qty_law=qty[:K])

    def threshold_table(self, k, pol=None):
        """b1bar_k for n = 1..N (rows) and I2 = 1..I2_max; NaN = no dispatch."""
        P = (self.pol[k] if pol is None else pol)[1:, 1 - self.p.I2_min:, 1:] > 0
        return np.where(P.any(axis=2), P.argmax(axis=2) + 1.0, np.nan)

    def myopic_table(self):
        """Note's one-shot rule: dispatch iff min_q Q_q(V_0) < Q_0(V_0)."""
        pol = np.zeros_like(self.pol[0])
        for n in range(1, self.p.N + 1):
            d, bq = self.ch.best_dispatch(self.V[0][n - 1])
            pol[n] = np.where(d < self.ch.Q(0, self.V[0][n - 1]), bq, 0)
        return pol


if __name__ == "__main__":
    K, EXACT, I2_0 = 2, True, 15          # change here
    p = Params(T=5, N=400, lam1=5, lam2=3, h=1, Cf=12, cu=1, pi1=6, pi2=8,
               c1=0, c2=0, v2=0, I2_max=40, I2_min=-31, b1_max=140)
    fk = FixedK(p, K, exact=EXACT).solve()
    out = fk.law(K, I2_0)
    print(f"{'exactly' if EXACT else 'at most'} K={K}: cost {out['cost']:.4f}, "
          f"P(fewer than K) {out['p_fewer']:.2e}")
    for r in out["dispatches"]:
        print(f"  dispatch {r['dispatch']}: prob {r['prob']:.4f}, "
              f"mean time {r['mean_time']:.3f}, mean qty {r['mean_qty']:.2f}")
