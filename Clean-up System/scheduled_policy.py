"""
scheduled_policy.py — scheduled-epoch dispatch policy SP(K)
==========================================================

Definition
----------
Fix K and epochs 0 < t_1 < ... < t_K < T, measured from the stockout of
Retailer 1. A dispatch from Retailer 2 to Retailer 1 may be made ONLY at these
epochs. At epoch t_k the system observes (I2, b1) and chooses

    q_k in {0, 1, ..., min(I2, b1)}

as a function of the observed state. Two quantity rules are provided.

  rule = "optimal"  q_k minimises the expected cost to the end of the horizon
                    given that the only remaining opportunities are
                    t_{k+1}, ..., t_K. q_k = 0 is allowed, so SP(K) makes at
                    most K dispatches. With K = 1 this is the post-stockout
                    analogue of the single scheduled transshipment of Zhou and
                    Wang (2023), whose amount is also chosen at the epoch from
                    the observed state; here the amount also accounts for Cf
                    and for any later epochs.
  rule = "clear"    q_k = min(I2, b1) whenever positive, the quantity rule of
                    the T-policy. With equally spaced epochs t_k = k*Delta this
                    is exactly the T-policy TP(Delta).

The epochs are chosen once, at the stockout, to minimise the expected cost
from the start state (I2_0, b1_0) with tau = T. They are open loop: they do
not react to the realised demand. The quantities are closed loop.

Ordering of costs from the same start state (all exact on the same chain):

    V*  <=  V^(K)  <=  SP_optimal(K, t*)  <=  SP_clear(K, t*)
                                         <=  SP(K, equally spaced)

where V^(K) is the closed-loop optimum with at most K dispatches. Hence
SP(K) − V^(K) is the value of state-dependent timing and V^(K) − V* is the
value of allowing more than K dispatches.

Solution method
---------------
Inner problem, epochs fixed: exact backward induction on the at-most-one-
event chain of solver.py, with the action set reduced to {0} outside the
epochs (Chain.solve_schedule). Exact, no simulation.

Outer problem, choose epochs:
  K = 1   exhaustive over all N periods, using the decomposition
          cost(m) = R[m] + <P[m], min_q Q_q(V_wait[m-1])>,
          where P[m] is the law of the state at period m under waiting and
          R[m] the expected flow cost before it.
  K = 2   exhaustive over a grid of step `step` periods with the same idea
          applied for every second epoch, followed by an exact local search
          on the full period grid.
  K >= 3  coordinate descent from the best (K-1)-epoch schedule with one
          epoch inserted, each move evaluated exactly.

Scope: Cf > 0 only. At Cf = 0 this is the 3-D model of solver.py, not the
2-D switching model of the Cf = 0 note, and a warning is issued.

Units: periods are remaining-period indices n (tau = n*dt); epochs are
reported in time from the stockout, t = T - n*dt.
"""

import itertools
import time
import numpy as np
from solver import Params
from transship_core import Chain, warn_if_cf0, _dot


class ScheduledPolicy:
    def __init__(self, p: Params, rule="optimal", chain: Chain = None):
        assert rule in ("optimal", "clear")
        warn_if_cf0(p, "ScheduledPolicy")
        self.p = p
        self.rule = rule
        self.ch = chain or Chain(p)
        self._Vwait = None

    # ── helpers ─────────────────────────────────────────────────────
    def t_of_n(self, n):
        return self.p.T - n * self.p.dt

    def n_of_t(self, t):
        return self.ch.n_of_time(t)

    def _ix(self, I2, b1):
        p = self.p
        return int(np.clip(I2, p.I2_min, p.I2_max) - p.I2_min), int(np.clip(b1, 0, p.b1_max))

    def _epoch_value(self, Vprev):
        """Value at an epoch period, given next-stage values Vprev."""
        ch = self.ch
        w = ch.Q(0, Vprev)
        if self.rule == "optimal":
            d, _ = ch.best_dispatch(Vprev)
            return np.minimum(w, d)
        qc = np.clip(np.minimum(ch.I2v[:, None], ch.b1v[None, :]), 0, None)
        out = w.copy()
        for q in range(1, ch.qtop + 1):
            m = qc == q
            if m.any():
                out[m] = ch.Q(q, Vprev)[m]
        return out

    @property
    def Vwait(self):
        if self._Vwait is None:
            self._Vwait = self.ch.solve_wait_only()
        return self._Vwait

    # ── exact evaluation for given epochs ───────────────────────────
    def evaluate_n(self, epochs_n, I2_0, b1_0=0):
        V, pol = self.ch.solve_schedule(epochs_n, self.rule)
        i, j = self._ix(I2_0, b1_0)
        return float(V[self.p.N, i, j]), V, pol

    def evaluate(self, epochs_t, I2_0, b1_0=0):
        return self.evaluate_n([self.n_of_t(t) for t in epochs_t], I2_0, b1_0)[0]

    def uniform_T_policy(self, Delta, I2_0, b1_0=0):
        """TP(Delta): epochs t_k = k*Delta < T, quantity rule of this object."""
        ts = [k * Delta for k in range(1, int(np.ceil(self.p.T / Delta)) + 1)
              if k * Delta < self.p.T - 1e-12]
        return self.evaluate(ts, I2_0, b1_0), ts

    # ── K = 1, exhaustive ───────────────────────────────────────────
    def best_K1(self, I2_0, b1_0=0):
        P, R = self.ch.wait_distributions(I2_0, b1_0)
        best = (np.inf, None)
        costs = np.full(self.p.N + 1, np.nan)
        for m in range(1, self.p.N + 1):
            U = self._epoch_value(self.Vwait[m - 1])
            c = R[m] + _dot(P[m], U.ravel())
            costs[m] = c
            if c < best[0]:
                best = (c, m)
        return dict(cost=best[0], epochs_n=[best[1]],
                    epochs_t=[self.t_of_n(best[1])], cost_by_n=costs)

    # ── K = 2, exhaustive on a grid then exact local search ─────────
    def best_K2(self, I2_0, b1_0=0, step=4, verbose=False):
        p, ch = self.p, self.ch
        N = p.N
        P, R = ch.wait_distributions(I2_0, b1_0)
        grid = list(range(1, N + 1, step))
        gset = set(grid)
        best = (np.inf, None)
        t0 = time.time()
        for m2 in grid:
            W = self._epoch_value(self.Vwait[m2 - 1])        # value at n = m2
            for n in range(m2 + 1, N + 1):
                if n in gset:
                    U = self._epoch_value(W)
                    c = R[n] + _dot(P[n], U.ravel())
                    if c < best[0]:
                        best = (c, (n, m2))
                W = ch.Q(0, W)
        if verbose:
            print(f"    grid search over {len(grid)} periods: {time.time()-t0:.1f}s, "
                  f"best {best[0]:.4f} at n = {best[1]}")
        return self._local_search(list(best[1]), I2_0, b1_0, radius=step,
                                  verbose=verbose)

    # ── exact local / coordinate search ─────────────────────────────
    def _local_search(self, epochs_n, I2_0, b1_0, radius=4, verbose=False,
                      max_rounds=20):
        N = self.p.N
        cur = sorted(set(int(e) for e in epochs_n), reverse=True)
        best_c = self.evaluate_n(cur, I2_0, b1_0)[0]
        for rnd in range(max_rounds):
            improved = False
            for k in range(len(cur)):
                for d in list(range(-radius, 0)) + list(range(1, radius + 1)):
                    cand = cur.copy()
                    cand[k] = cand[k] + d
                    if cand[k] < 1 or cand[k] > N or len(set(cand)) < len(cand):
                        continue
                    c = self.evaluate_n(cand, I2_0, b1_0)[0]
                    if c < best_c - 1e-12:
                        best_c, cur, improved = c, sorted(cand, reverse=True), True
            if not improved:
                if radius > 1:
                    radius = max(1, radius // 2)
                    continue
                break
        if verbose:
            print(f"    local search: {best_c:.4f} at n = {cur}")
        return dict(cost=best_c, epochs_n=cur,
                    epochs_t=[self.t_of_n(n) for n in cur])

    def best_K(self, K, I2_0, b1_0=0, step=4, verbose=False):
        if K == 1:
            return self.best_K1(I2_0, b1_0)
        if K == 2:
            return self.best_K2(I2_0, b1_0, step=step, verbose=verbose)
        prev = self.best_K(K - 1, I2_0, b1_0, step=step, verbose=verbose)
        N = self.p.N
        cands = []
        pts = sorted([N + 1] + prev["epochs_n"] + [0], reverse=True)
        for a, b in zip(pts[:-1], pts[1:]):          # insert one epoch per gap
            if a - b >= 2:
                cands.append(sorted(prev["epochs_n"] + [(a + b) // 2], reverse=True))
        best = (np.inf, None)
        for c in cands:
            v = self.evaluate_n(c, I2_0, b1_0)[0]
            if v < best[0]:
                best = (v, c)
        return self._local_search(best[1], I2_0, b1_0, radius=max(step, 8),
                                  verbose=verbose)

    # ── reporting ───────────────────────────────────────────────────
    def dispatch_law(self, epochs_n, I2_0, b1_0=0, cmax=8):
        _, _, pol = self.evaluate_n(epochs_n, I2_0, b1_0)
        return self.ch.forward(pol, I2_0, b1_0, cmax=cmax)


def compare(p: Params, I2_0, b1_0=0, Kmax=3, step=4, verbose=True):
    """
    Exact comparison from (I2_0, b1_0) at tau = T of
      V*        optimal closed-loop policy
      V^(K)     closed loop, at most K dispatches
      SP(K)     scheduled epochs, optimal quantity, optimised epochs
      SPc(K)    scheduled epochs, clear quantity, optimised epochs
      TP(Δ*)    uniform epochs, clear quantity, best Δ on a grid
    """
    warn_if_cf0(p, "compare")
    ch = Chain(p)
    i0 = int(I2_0 - p.I2_min)
    V, polstar = ch.solve()
    Vk, _ = ch.solve_k_limited(Kmax)
    vstar = float(V[p.N, i0, b1_0])
    sp = ScheduledPolicy(p, "optimal", ch)
    spc = ScheduledPolicy(p, "clear", ch)
    rows = []
    for K in range(1, Kmax + 1):
        r1 = sp.best_K(K, I2_0, b1_0, step=step)
        r2 = spc.best_K(K, I2_0, b1_0, step=step)
        rows.append(dict(K=K, V_K=float(Vk[K][p.N, i0, b1_0]),
                         SP=r1["cost"], SP_epochs=np.round(r1["epochs_t"], 4),
                         SPc=r2["cost"], SPc_epochs=np.round(r2["epochs_t"], 4)))
        if verbose:
            print(f"  K={K}: V^(K)={rows[-1]['V_K']:.4f}  SP={r1['cost']:.4f} "
                  f"at t={np.round(r1['epochs_t'], 3)}  SPc={r2['cost']:.4f} "
                  f"at t={np.round(r2['epochs_t'], 3)}")
    best_tp = (np.inf, None)
    for Delta in np.round(np.arange(0.125, p.T, 0.125), 6):
        c, _ = spc.uniform_T_policy(float(Delta), I2_0, b1_0)
        if c < best_tp[0]:
            best_tp = (c, float(Delta))
    return dict(V_star=vstar, rows=rows, TP=best_tp[0], TP_Delta=best_tp[1])


if __name__ == "__main__":
    # two-dispatch instance: meeting parameters with Cf = 35, start (23, 0)
    p = Params(T=5, N=400, lam1=5, lam2=3, h=1, Cf=35, cu=1, pi1=6, pi2=8,
               c1=0, c2=0, v2=0, I2_max=40, I2_min=-31, b1_max=140)
    I2_0 = 23
    t0 = time.time()
    out = compare(p, I2_0, 0, Kmax=3)
    print(f"V* = {out['V_star']:.4f}, best TP = {out['TP']:.4f} "
          f"(Δ = {out['TP_Delta']}), {time.time()-t0:.0f}s")