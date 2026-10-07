"""
k_dispatch_solver.py — optimal policy when only the number of dispatches is fixed
================================================================================

Problem
-------
The model is that of solver.py: state (I2, b1, tau), Retailer 2 serves its own
demand first, Retailer 1 demand is backlogged, a dispatch of q units costs
Cf + cu*q, and flow cost is charged on the post-dispatch state. The only
change is that at most K dispatches may be made over the horizon. WHEN each
dispatch is made and HOW MANY units it carries are decided by the model, as a
function of the realised state; nothing is scheduled in advance.

The state is augmented by k, the number of dispatches still allowed:

    V_0(I2, b1, tau)  = cost of never dispatching again
    V_k^n(I2, b1) = min{ Q_0(V_k^{n-1}),            wait, k unchanged
                         min_{q>=1} Q_q(V_{k-1}^{n-1}) }   dispatch, k -> k-1

with Q_q(V) = Cf*1{q>0} + cu*q + dt*(h (I2-q)^+ + pi1 (b1-q) + pi2 (I2-q)^-)
              + p0 V(I2-q, b1-q) + p1 V(I2-q, b1-q+1) + p2 V(I2-q-1, b1-q).

"Exactly K" is not used because under Poisson demand there are always paths
of positive probability on which a K-th dispatch is impossible (no backlog,
or Retailer 2 already empty). V_k is non-increasing in k and equals the
unrestricted optimum V* once k exceeds the number of dispatches that can pay.

Each layer k has its own policy table q_k(n, I2, b1) and threshold
b1bar_k(I2, tau) = smallest b1 >= 1 at which layer k dispatches.

Conventions are those of solver.py exactly: n = remaining periods,
tau = n*dt, time t = T - n*dt, same clipping at the bounds and ties to the
smallest q (waiting wins ties). With K large enough the solver returns the
policy and values of solver.py; this is checked by `verify_against_full`.

The file is self-contained: it needs numpy and the Params class of solver.py.
Everything is vectorised over the state grid, so one layer costs about as
much as one solve of the unrestricted DP.

Main entry points
-----------------
    kd = KDispatchDP(p, K=3); kd.solve()
    kd.value(k, I2, b1)                    V_k at tau = T
    kd.threshold_table(k)                  b1bar_k for every period and I2
    kd.path_law(K, I2_0, b1_0)             exact law of the number, timing and
                                           size of each dispatch
    kd.structure_report()                  monotonicity checks per layer
    kd.myopic_table()                      discrete analogue of the note's
                                           one-shot rule, for comparison
"""

import warnings
import numpy as np
from solver import Params


class KDispatchDP:
    def __init__(self, p: Params, K: int):
        p.validate()
        if float(p.Cf) == 0.0:
            warnings.warn("Cf = 0: this is the 3-D model of solver.py at Cf = 0, "
                          "not the 2-D switching model of the Cf = 0 note.",
                          stacklevel=2)
        self.p, self.K = p, int(K)
        self.I2v = np.arange(p.I2_min, p.I2_max + 1)
        self.b1v = np.arange(0, p.b1_max + 1)
        self.shape = (len(self.I2v), len(self.b1v))
        self.qmax = max(0, min(p.I2_max, p.b1_max))
        I2g, b1g, sh = self.I2v[:, None], self.b1v[None, :], self.shape
        cI = lambda x: np.clip(x, p.I2_min, p.I2_max) - p.I2_min
        cB = lambda x: np.clip(x, 0, p.b1_max)
        self.g, self.feas, self.ii0, self.jj0, self.jj1, self.ii2 = ([] for _ in range(6))
        for q in range(self.qmax + 1):
            I2a = np.broadcast_to(I2g - q, sh)
            b1a = np.broadcast_to(b1g - q, sh)
            g = p.cu * q + p.dt * (p.h * np.maximum(0, I2a) + p.pi1 * b1a
                                   + p.pi2 * np.maximum(0, -I2a))
            if q > 0:
                g = g + p.Cf
            self.g.append(np.asarray(g, float))
            self.feas.append(np.broadcast_to((I2g >= q) & (b1g >= q), sh)
                             if q > 0 else np.ones(sh, bool))
            self.ii0.append(cI(I2a)); self.jj0.append(cB(b1a))
            self.jj1.append(cB(b1a + 1)); self.ii2.append(cI(I2a - 1))
        self.V = None          # V[k][n] for k = 0..K
        self.pol = None        # pol[k][n] (pol[0] is all zeros)

    # ── one-step operators (same arithmetic order as solver.py) ──────
    def Q(self, q, V):
        p = self.p
        return self.g[q] + (p.p0 * V[self.ii0[q], self.jj0[q]]
                            + p.p1 * V[self.ii0[q], self.jj1[q]]
                            + p.p2 * V[self.ii2[q], self.jj0[q]])

    def best_dispatch(self, V):
        best = np.full(self.shape, np.inf)
        bq = np.zeros(self.shape, np.int16)
        for q in range(1, self.qmax + 1):
            if not self.feas[q].any():
                break
            val = np.where(self.feas[q], self.Q(q, V), np.inf)
            better = val < best
            best = np.where(better, val, best)
            bq = np.where(better, q, bq)
        return best, bq

    def terminal(self):
        p = self.p
        I2, b1 = self.I2v[:, None], self.b1v[None, :]
        return (p.c1 * b1 + p.c2 * np.maximum(0, -I2)
                - p.v2 * np.maximum(0, I2)) * np.ones(self.shape)

    # ── solve all layers ─────────────────────────────────────────────
    def solve(self, verbose=False):
        p, N = self.p, self.p.N
        V0 = np.empty((N + 1,) + self.shape)
        V0[0] = self.terminal()
        for n in range(1, N + 1):
            V0[n] = self.Q(0, V0[n - 1])
        self.V = [V0]
        self.pol = [np.zeros((N + 1,) + self.shape, np.int16)]
        for k in range(1, self.K + 1):
            prev = self.V[k - 1]
            Vk = np.empty((N + 1,) + self.shape)
            pk = np.zeros((N + 1,) + self.shape, np.int16)
            Vk[0] = self.terminal()
            for n in range(1, N + 1):
                w = self.Q(0, Vk[n - 1])
                d, bq = self.best_dispatch(prev[n - 1])
                disp = d < w
                Vk[n] = np.where(disp, d, w)
                pk[n] = np.where(disp, bq, 0)
            self.V.append(Vk); self.pol.append(pk)
            if verbose:
                print(f"  layer k = {k} solved")
        return self

    # ── queries ──────────────────────────────────────────────────────
    def _ix(self, I2, b1):
        p = self.p
        return (int(np.clip(I2, p.I2_min, p.I2_max) - p.I2_min),
                int(np.clip(b1, 0, p.b1_max)))

    def value(self, k, I2, b1, n=None):
        i, j = self._ix(I2, b1)
        return float(self.V[k][self.p.N if n is None else n][i, j])

    def n_of_tau(self, tau):
        return int(min(self.p.N, max(1, round(tau / self.p.dt))))

    def threshold_row(self, k, n, pol=None):
        """b1bar_k for I2 = 1..I2_max at period n; NaN = +infinity."""
        P = self.pol[k][n] if pol is None else pol
        sub = P[1 - self.p.I2_min:, 1:] > 0
        return np.where(sub.any(axis=1), sub.argmax(axis=1) + 1.0, np.nan)

    def threshold_table(self, k, pol_all=None):
        """b1bar_k for every period n = 1..N (rows) and I2 = 1..I2_max."""
        P = self.pol[k] if pol_all is None else pol_all
        return np.vstack([self.threshold_row(k, n, P[n])
                          for n in range(1, self.p.N + 1)])

    # ── discrete analogue of the note's one-shot rule ───────────────
    def myopic_table(self):
        """
        Dispatch iff min_q Q_q(V_0) < Q_0(V_0): 'dispatch now, never again'
        against 'never dispatch', both priced by V_0. This is the discrete
        counterpart of Vd < Vw in Sections 4-5 of the 22 July note. The exact
        layer k = 1 uses the same dispatch branch but prices waiting with V_1,
        which keeps the option of one later dispatch.
        """
        N = self.p.N
        pol = np.zeros((N + 1,) + self.shape, np.int16)
        for n in range(1, N + 1):
            w = self.Q(0, self.V[0][n - 1])
            d, bq = self.best_dispatch(self.V[0][n - 1])
            pol[n] = np.where(d < w, bq, 0)
        return pol

    # ── exact law of the dispatches from a start state ──────────────
    def path_law(self, K, I2_0, b1_0=0):
        """
        Forward propagation under the layered policy: a path that has made u
        dispatches follows the table of layer K - u. Returns the expected cost
        (equal to V_K), the law of the number of dispatches and, for each
        dispatch index j = 1..K, its probability, the law of its epoch and the
        law of its quantity.
        """
        p, N = self.p, self.p.N
        nI, nB = self.shape
        size = nI * nB
        mass = np.zeros((K + 1, size))
        i, j = self._ix(I2_0, b1_0)
        mass[0, i * nB + j] = 1.0
        when = np.zeros((K, N + 1))
        qlaw = np.zeros((K, self.qmax + 1))
        cost = 0.0
        for n in range(N, 0, -1):
            new = np.zeros_like(mass)
            for u in range(K + 1):
                mu = mass[u]
                if not mu.any():
                    continue
                A = (np.asarray(self.pol[K - u][n]).ravel() if u < K
                     else np.zeros(size, np.int16))
                for q in np.unique(A):
                    q = int(q)
                    cells = np.nonzero(A == q)[0]
                    m = mu[cells]
                    if not m.any():
                        continue
                    cost += float(np.sum(m * self.g[q].ravel()[cells]))
                    d0 = self.ii0[q].ravel()[cells] * nB + self.jj0[q].ravel()[cells]
                    d1 = self.ii0[q].ravel()[cells] * nB + self.jj1[q].ravel()[cells]
                    d2 = self.ii2[q].ravel()[cells] * nB + self.jj0[q].ravel()[cells]
                    u2 = u + 1 if q > 0 else u
                    if q > 0:
                        when[u, n] += m.sum()
                        qlaw[u, q] += m.sum()
                    new[u2] += (np.bincount(d0, m * p.p0, size)
                                + np.bincount(d1, m * p.p1, size)
                                + np.bincount(d2, m * p.p2, size))
            mass = new
        cost += float(np.sum(mass.sum(axis=0) * self.terminal().ravel()))
        t = p.T - np.arange(N + 1) * p.dt
        rows = []
        for u in range(K):
            pr = when[u].sum()
            rows.append(dict(
                dispatch=u + 1, prob=pr,
                mean_time=float(np.sum(when[u] * t)) / pr if pr > 0 else np.nan,
                mean_qty=float(np.sum(qlaw[u] * np.arange(self.qmax + 1))) / pr
                if pr > 0 else np.nan))
        return dict(cost=cost, count_pmf=mass.sum(axis=1), dispatches=rows,
                    time_law=when, t_grid=t, qty_law=qlaw)

    # ── structure checks ─────────────────────────────────────────────
    def structure_report(self, tie_free=True):
        """
        Per layer k:
          I2_viol   cells where b1bar_k increases in I2, i.e. departures
                    from a non-increasing threshold; for k = 1 this is the
                    steady rise of the threshold, not a local bump
          tau_viol  cells where b1bar_k increases as tau increases
          ret_rule  share of dispatch cells whose q equals
                    min(b1, I2 - S_k(n)), S_k(n) = retained stock at the
                    largest-backlog dispatch cell of that period
        Between layers k and k+1:
          lower     share of (n, I2) cells with b1bar_{k+1} <= b1bar_k,
                    i.e. more remaining dispatches never delays a dispatch
        """
        p, N, K = self.p, self.p.N, self.K
        tabs = [None] + [self.threshold_table(k) for k in range(1, K + 1)]
        out = []
        for k in range(1, K + 1):
            B = tabs[k]
            fin = np.isfinite(B)
            iv = int(((B[:, 1:] > B[:, :-1]) | (fin[:, :-1] & ~fin[:, 1:])).sum())
            tv = int(((B[1:, :] > B[:-1, :]) | (fin[:-1, :] & ~fin[1:, :])).sum())
            ok = tot = 0
            for n in range(1, N + 1):
                P = self.pol[k][n]
                d = np.nonzero(P > 0)
                if not d[0].size:
                    continue
                I2 = self.I2v[d[0]]; b1 = self.b1v[d[1]]; q = P[d]
                retained = I2 - q
                # retention level: what is kept when the backlog does not bind
                free = q < b1
                S = int(np.min(retained[free])) if free.any() else None
                pred = np.minimum(b1, I2 - S) if S is not None else b1
                ok += int((pred == q).sum()); tot += q.size
            row = dict(k=k, I2_violations=iv, tau_violations=tv,
                       retention_rule_share=ok / tot if tot else np.nan)
            if k < K:
                Bn = tabs[k + 1]
                a = np.where(np.isfinite(B), B, np.inf)
                b = np.where(np.isfinite(Bn), Bn, np.inf)
                row["share_b1bar_next_le_this"] = float((b <= a).mean())
            out.append(row)
        return out

    def marginal_values(self, I2_0, b1_0=0):
        """V_{k-1} - V_k at tau = T from (I2_0, b1_0), k = 1..K."""
        v = [self.value(k, I2_0, b1_0) for k in range(self.K + 1)]
        return [v[k - 1] - v[k] for k in range(1, self.K + 1)], v

    # ── verification against the unrestricted DP ────────────────────
    def verify_against_full(self):
        """
        Solves the unrestricted DP of solver.py with the same operators and
        reports max |V_K - V*| over the grid and at tau = T. When K exceeds
        the number of dispatches that can pay anywhere, both are zero.
        """
        N = self.p.N
        V = self.terminal()
        for n in range(1, N + 1):
            w = self.Q(0, V)
            d, _ = self.best_dispatch(V)
            V = np.where(d < w, d, w)
        return float(np.abs(self.V[self.K][N] - V).max()), V


if __name__ == "__main__":
    p = Params(T=5, N=400, lam1=5, lam2=3, h=1, Cf=12, cu=1, pi1=6, pi2=8,
               c1=0, c2=0, v2=0, I2_max=40, I2_min=-31, b1_max=140)
    kd = KDispatchDP(p, K=4).solve(verbose=True)
    I2_0 = 15
    dv, v = kd.marginal_values(I2_0)
    print("V_k, k = 0..4:", [round(x, 4) for x in v])
    print("value of the k-th dispatch:", [round(x, 4) for x in dv])
    for K in (1, 2, 3):
        law = kd.path_law(K, I2_0)
        print(f"K={K}: cost {law['cost']:.4f}, #dispatches {np.round(law['count_pmf'], 4)}")
        for r in law["dispatches"]:
            print(f"    dispatch {r['dispatch']}: prob {r['prob']:.4f}, "
                  f"mean time {r['mean_time']:.3f}, mean qty {r['mean_qty']:.2f}")
    for r in kd.structure_report():
        print(r)
    print("max |V_K - V*| on the grid:", kd.verify_against_full()[0])