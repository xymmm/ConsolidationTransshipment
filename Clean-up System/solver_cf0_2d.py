"""
solver_cf0_2d.py — Exact DP for the Cf = 0 optimal switching model
===================================================================

This solves the model defined in the note "Exact optimal threshold in Cf = 0",
Section 1. That model is TWO-dimensional: the value function is V(I2, tau).
It is NOT the same model as solver.py, which carries the Retailer-1 backlog as
a third state variable and gives V(I2, b1, tau).

Model (note, Section 1)
-----------------------
  State:   I2  (integer, may be negative),  tau = remaining time
  Mode:    U(t) in {0, 1};  U = 1 (transship) is available only when I2 >= 1

  Waiting mode (U = 0):
      Retailer-1 arrivals are rejected and charged pi1 * tau one-off, i.e. the
      unit stays backordered until the end of the horizon and is never cleared.
      Retailer-1 arrivals do NOT consume Retailer-2 inventory.
      I2 falls by 1 at rate lam2 only.

  Transshipment mode (U = 1):
      Retailer-1 arrivals are satisfied at unit cost cu.
      I2 falls by 1 at rate lam1 + lam2.

  Flow cost (always):   h * I2^+  +  pi2 * I2^-
  Terminal cost:        c2 * I2^-  -  v2 * I2^+
                        The closed form in Section 4.1 assumes V(I2, 0) = 0,
                        i.e. c2 = v2 = 0. Those are the defaults here.

Discretisation
--------------
At-most-one-event Poisson discretisation with step dt = T / N:
      p1 = lam1 * dt   Retailer-1 arrival
      p2 = lam2 * dt   Retailer-2 arrival
      p0 = 1 - p1 - p2 no arrival

On a Retailer-1 arrival the decision is taken per arrival:
      dispatch :  cu     + V(I2 - 1)
      wait     :  pi1*tau + V(I2)
This is exactly the comparison in eq. (20) of the note, so the DP policy is
directly comparable to the analytical threshold.

      V^n(I2) = dt*(h*I2^+ + pi2*I2^-)
                + p1 * min{ cu + V^{n-1}(I2-1),  pi1*tau + V^{n-1}(I2) }
                + p2 * V^{n-1}(I2 - 1)
                + p0 * V^{n-1}(I2)

with the min restricted to the wait branch when I2 < 1.

Usage
-----
    from solver_cf0_2d import ParamsCf0, SwitchingDPCf0, analytic_threshold
    p  = ParamsCf0(T=5.0, N=2000, lam2=5.0, h=1.0, cu=2.0, pi1=1.5, pi2=4.0)
    dp = SwitchingDPCf0(p)
    dp.solve()
    dp.threshold_at_tau(2.5)          # DP threshold  Ibar(tau)
    analytic_threshold(2.5, p)        # note's closed-form staircase
"""

import numpy as np
from dataclasses import dataclass
from math import exp, sqrt, ceil
import time as _time


# ═══════════════════════════════════════════════════════════════════════
#  Parameters
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class ParamsCf0:
    """Parameters of the Cf = 0 switching model."""
    # --- Horizon ---
    T:    float = 5.0
    N:    int   = 2000      # a 1-D state space makes large N cheap

    # --- Demand ---
    lam1: float = 3.0       # Retailer 1 arrival rate
    lam2: float = 5.0       # Retailer 2 arrival rate

    # --- Costs ---
    h:    float = 1.0       # holding cost rate
    cu:   float = 2.0       # unit transshipment cost
    pi1:  float = 1.5       # Retailer 1 backorder rate
    pi2:  float = 4.0       # Retailer 2 backorder rate

    # --- Terminal costs (0 to match the closed form of Section 4.1) ---
    c2:   float = 0.0       # cost to clear Retailer 2 backlog at T
    v2:   float = 0.0       # salvage value per unit of I2 > 0

    # --- State-space bounds ---
    I2_max: int = 60
    I2_min: int = -60

    @property
    def dt(self): return self.T / self.N

    @property
    def p1(self): return self.lam1 * self.dt

    @property
    def p2(self): return self.lam2 * self.dt

    @property
    def p0(self): return 1.0 - self.p1 - self.p2

    def validate(self):
        assert self.N >= 1
        assert self.p0 >= 0, f"dt too large: p0={self.p0:.4f} < 0. Increase N."
        assert self.I2_min < self.I2_max
        return True

    def with_auto_bounds(self, sigma: float = 6.0):
        """
        Return a copy whose state-space bounds scale with the demand over the
        horizon. The lower bound must cover lam2*T plus a buffer, otherwise the
        Retailer-2 backorder cost is capped and the threshold is distorted.
        """
        s2 = self.lam2 * self.T
        span = int(ceil(s2 + sigma * sqrt(max(s2, 1e-12))))
        new = ParamsCf0(**vars(self))
        new.I2_min = -span
        new.I2_max = max(40, span)
        return new

    def summary(self) -> str:
        return (f"T={self.T} N={self.N} lam1={self.lam1} lam2={self.lam2} "
                f"h={self.h} cu={self.cu} pi1={self.pi1} pi2={self.pi2} "
                f"c2={self.c2} v2={self.v2} I2 in [{self.I2_min}, {self.I2_max}]")


# ═══════════════════════════════════════════════════════════════════════
#  DP solver
# ═══════════════════════════════════════════════════════════════════════
class SwitchingDPCf0:
    """
    Backward-induction solver for the Cf = 0 switching model.

    After solve():
        self.policy[n, ii]  -> 1 if dispatch, 0 if wait
        self.V_final[ii]    -> V^N
        self.V_all[n, ii]   -> requires solve(store_V=True)
    """

    def __init__(self, params: ParamsCf0):
        self.p = params
        self.p.validate()
        self._nI2 = self.p.I2_max - self.p.I2_min + 1
        self._I2v = np.arange(self.p.I2_min, self.p.I2_max + 1)
        self.policy = np.zeros((self.p.N + 1, self._nI2), dtype=np.int8)
        self.V_final = None
        self.V_all = None
        self._solved = False

    # ── index helpers ────────────────────────────────────────────
    def _ii(self, I2):
        return np.clip(I2, self.p.I2_min, self.p.I2_max) - self.p.I2_min

    # ── costs ────────────────────────────────────────────────────
    def terminal(self, I2):
        """V(I2, 0) = c2 * I2^- - v2 * I2^+"""
        return self.p.c2 * np.maximum(0, -I2) - self.p.v2 * np.maximum(0, I2)

    # ── main solver ──────────────────────────────────────────────
    def solve(self, store_V: bool = False, verbose: bool = True) -> float:
        p = self.p
        I2v = self._I2v
        dt, p0, p1, p2 = p.dt, p.p0, p.p1, p.p2

        flow = dt * (p.h * np.maximum(0, I2v) + p.pi2 * np.maximum(0, -I2v))
        idx_m1 = self._ii(I2v - 1)
        can_dispatch = I2v >= 1

        V = self.terminal(I2v).astype(np.float64)
        if store_V:
            self.V_all = np.zeros((p.N + 1, self._nI2), dtype=np.float64)
            self.V_all[0] = V

        t0 = _time.time()
        for n in range(1, p.N + 1):
            tau = n * dt
            Vm1 = V[idx_m1]

            cost_dispatch = p.cu + Vm1            # satisfy the R1 arrival now
            cost_wait     = p.pi1 * tau + V       # reject, backordered to T

            dispatch = can_dispatch & (cost_dispatch <= cost_wait)
            r1_value = np.where(dispatch, cost_dispatch, cost_wait)

            V = flow + p1 * r1_value + p2 * Vm1 + p0 * V

            self.policy[n] = dispatch.astype(np.int8)
            if store_V:
                self.V_all[n] = V

            if verbose and (n % 500 == 0 or n == p.N):
                print(f"    period {n}/{p.N}")

        elapsed = _time.time() - t0
        self.V_final = V.copy()
        self._solved = True
        if verbose:
            print(f"    Solved in {elapsed:.2f}s "
                  f"({self._nI2} states, {p.N} periods)")
        return elapsed

    # ── queries ──────────────────────────────────────────────────
    def n_for_tau(self, tau: float) -> int:
        return int(min(self.p.N, max(1, round(tau / self.p.dt))))

    def get_policy(self, n: int, I2: int) -> int:
        assert self._solved, "Call solve() first."
        return int(self.policy[n, self._ii(I2)])

    def get_value(self, n: int, I2: int) -> float:
        assert self._solved, "Call solve() first."
        ii = self._ii(I2)
        if self.V_all is not None:
            return float(self.V_all[n, ii])
        if n == self.p.N:
            return float(self.V_final[ii])
        raise ValueError("V_all not stored. Re-solve with store_V=True.")

    def threshold(self, n: int) -> float:
        """Smallest I2 >= 1 at which the DP dispatches, at period n."""
        assert self._solved, "Call solve() first."
        row = self.policy[n]
        pos = np.where((self._I2v >= 1) & (row == 1))[0]
        return float(self._I2v[pos[0]]) if len(pos) else np.inf

    def threshold_at_tau(self, tau: float) -> float:
        return self.threshold(self.n_for_tau(tau))

    def threshold_curve(self, taus) -> np.ndarray:
        return np.array([self.threshold_at_tau(float(t)) for t in taus])


# ═══════════════════════════════════════════════════════════════════════
#  Analytical staircase of the note  (eq. 20-22)
# ═══════════════════════════════════════════════════════════════════════
def analytic_threshold(tau, p: ParamsCf0 = None, lam2=None, h=None, cu=None,
                       pi1=None, pi2=None, nmax=500):
    """
    Note eq. (20)-(22):
        Ibar(tau) = min{ n >= 1 : M(n, tau) >= g(tau) }
        M(n, tau) = E[min(K, n)] = sum_{j=1}^n P(K >= j),  K ~ Poisson(lam2*tau)
        g(tau)    = lam2 * (cu + (pi2 - pi1) * tau) / (h + pi2)

    Returns np.inf when no finite threshold exists. Pure Python, no scipy.
    """
    if p is not None:
        lam2, h, cu, pi1, pi2 = p.lam2, p.h, p.cu, p.pi1, p.pi2
    if tau <= 0:
        return np.inf

    hp2 = max(h + pi2, 1e-12)
    g = lam2 * (cu + (pi2 - pi1) * tau) / hp2
    mu = lam2 * tau

    pmf = exp(-mu)          # P(K = n-1), starting at n = 1
    cdf_below = 0.0         # P(K <= n-2)
    M = 0.0
    for n in range(1, nmax + 1):
        p_ge = max(1.0 - (cdf_below + pmf), 0.0)   # P(K >= n)
        M += p_ge
        if M >= g:
            return float(n)
        cdf_below += pmf
        pmf *= mu / n
    return np.inf


def analytic_curve(taus, p: ParamsCf0) -> np.ndarray:
    return np.array([analytic_threshold(float(t), p) for t in taus])


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    p = ParamsCf0(T=5.0, N=2000, lam1=3.0, lam2=5.0,
                  h=1.0, cu=2.0, pi1=1.5, pi2=4.0).with_auto_bounds()
    print(p.summary())
    dp = SwitchingDPCf0(p)
    dp.solve(verbose=True)

    print("\n  tau |  DP  | analytic")
    for tau in [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:
        print(f" {tau:4.1f} | {dp.threshold_at_tau(tau):4.0f} "
              f"| {analytic_threshold(tau, p):4.0f}")