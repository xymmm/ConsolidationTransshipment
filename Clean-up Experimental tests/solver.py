"""
solver.py — Backward-Induction DP for Two-Location Transshipment with Clean-up
===============================================================================

Model (Section 3 of the paper):
  State:   (I₂, b₁)  ∈  [I2_min, I2_max] × [0, b1_max]
  Action:  q ∈ {0, 1, …, min(I₂, b₁)}   if I₂>0 and b₁>0
           q = 0                           otherwise

  Transition (at-most-one-event Poisson discretisation):
      p₁ = λ₁ Δt   → (I₂, b₁+1)       Retailer 1 demand (backlogged)
      p₂ = λ₂ Δt   → (I₂−1, b₁)       Retailer 2 demand (served / backlogged)
      p₀ = 1−p₁−p₂ → (I₂, b₁)         no arrival

  One-period cost:
      g(I₂,b₁,q) = Cf·𝟙{q>0} + cᵤ·q + Δt·(h·I₂⁺ + π₁·b₁ + π₂·I₂⁻)

  Terminal (clean-up):
      V⁰(I₂,b₁) = c₁·b₁ + c₂·I₂⁻ − v₂·I₂⁺

  Bellman (n = 1,…,N):
      V^n(I₂,b₁) = min_q { g(I₂,b₁,q) + Σ pᵢ V^{n-1}(post-action-post-transition) }

Outputs:
  - policy[n, I₂, b₁]  →  q*          full policy tensor
  - V_final[I₂, b₁]    →  V^N         value function at start of horizon
  - V_all[n, ...]       →  optional    full value function over all periods

Usage:
    from solver import Params, TransshipmentDP
    p = Params(N=200, Cf=20, pi1=10, pi2=10, c1=5, c2=5, v2=1)
    dp = TransshipmentDP(p)
    dp.solve()
    dp.get_policy(n=200, I2=15, b1=3)   # → q*
    dp.get_value(n=200, I2=15, b1=0)    # → V^N(15,0)
"""

import numpy as np
from dataclasses import dataclass
import time as _time


# ═══════════════════════════════════════════════════════════════════════
#  Parameters
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Params:
    """All model parameters in one place."""
    # --- Horizon ---
    T:    float = 2.0       # total time between replenishments
    N:    int   = 200       # number of discrete periods

    # --- Demand ---
    lam1: float = 8.0       # Poisson rate, Retailer 1  (backlogged)
    lam2: float = 5.0       # Poisson rate, Retailer 2  (served first)

    # --- Per-period costs ---
    h:    float = 0.1       # holding cost rate  (per unit per unit time)
    Cf:   float = 20.0      # fixed transshipment cost
    cu:   float = 1.0       # unit transshipment cost
    pi1:  float = 10.0      # penalty rate for Retailer 1 backlog
    pi2:  float = 10.0      # penalty rate for Retailer 2 backlog (I₂<0)

    # --- Terminal clean-up costs ---
    c1:   float = 5.0       # unit cost to clear Retailer 1 backlog at T
    c2:   float = 5.0       # unit cost to clear Retailer 2 backlog at T (I₂<0)
    v2:   float = 1.0       # salvage value per unit of I₂>0

    # --- State-space bounds ---
    I2_max: int = 25
    I2_min: int = -15
    b1_max: int = 30

    # --- Derived quantities ---
    @property
    def dt(self):  return self.T / self.N

    @property
    def p1(self):  return self.lam1 * self.dt

    @property
    def p2(self):  return self.lam2 * self.dt

    @property
    def p0(self):  return 1.0 - self.p1 - self.p2

    def validate(self):
        assert self.p0 >= 0,    f"Δt too large: p0={self.p0:.4f}<0. Increase N."
        assert self.v2 <= self.c2, f"v2={self.v2} > c2={self.c2}, violates model assumption."
        assert self.N >= 1
        return True

    def summary(self) -> str:
        return (f"T={self.T} N={self.N} λ₁={self.lam1} λ₂={self.lam2} "
                f"Cf={self.Cf} cᵤ={self.cu} π₁={self.pi1} π₂={self.pi2} "
                f"c₁={self.c1} c₂={self.c2} v₂={self.v2}")


# ═══════════════════════════════════════════════════════════════════════
#  DP Solver
# ═══════════════════════════════════════════════════════════════════════
class TransshipmentDP:
    """
    Backward-induction solver.

    After solve(), the full policy tensor is available via
        self.policy[n, ii, jj]       (raw index access)
        self.get_policy(n, I2, b1)   (named access with clipping)

    Value function at n=N is always available; full V over all n
    requires solve(store_V=True).
    """

    def __init__(self, params: Params):
        self.p = params
        self.p.validate()

        # Dimensions
        self._nI2 = self.p.I2_max - self.p.I2_min + 1   # number of I₂ levels
        self._nb1 = self.p.b1_max + 1                     # number of b₁ levels

        # Storage
        self.policy = np.zeros((self.p.N + 1, self._nI2, self._nb1), dtype=np.int16)
        self.V_final = None      # V at n=N (always stored)
        self.V_all   = None      # V at all n (optional)

        self._solved = False

    # ── index / clipping helpers ─────────────────────────────────
    def _ii(self, I2: int) -> int:
        """I₂ value → array index."""
        return I2 - self.p.I2_min

    def _I2(self, ii: int) -> int:
        """Array index → I₂ value."""
        return ii + self.p.I2_min

    def _clip_I2(self, I2: int) -> int:
        return max(self.p.I2_min, min(self.p.I2_max, I2))

    def _clip_b1(self, b1: int) -> int:
        return max(0, min(self.p.b1_max, b1))

    # ── cost functions ───────────────────────────────────────────
    def terminal(self, I2: int, b1: int) -> float:
        """V⁰(I₂, b₁) = c₁·b₁ + c₂·(−I₂)⁺ − v₂·I₂⁺"""
        return self.p.c1 * b1 + self.p.c2 * max(0, -I2) - self.p.v2 * max(0, I2)

    def g(self, I2: int, b1: int, q: int) -> float:
        """One-period cost g(I₂, b₁, q)."""
        cost = self.p.cu * q + self.p.dt * (
            self.p.h * max(0, I2) + self.p.pi1 * b1 + self.p.pi2 * max(0, -I2)
        )
        if q > 0:
            cost += self.p.Cf
        return cost

    # ── main solver ──────────────────────────────────────────────
    def solve(self, store_V: bool = False, verbose: bool = True) -> float:
        """
        Run backward induction n = 0, 1, …, N.

        Parameters
        ----------
        store_V : bool
            If True, keep V_all[n, ii, jj] for every period (memory-heavy
            for large N and state space, but needed for time-threshold analysis).
        verbose : bool
            Print progress every 50 periods.

        Returns
        -------
        elapsed : float   (seconds)
        """
        p = self.p
        nI2, nb1 = self._nI2, self._nb1

        V = np.zeros((nI2, nb1), dtype=np.float64)
        if store_V:
            self.V_all = np.zeros((p.N + 1, nI2, nb1), dtype=np.float64)

        # ── n = 0: terminal condition ──
        for ii in range(nI2):
            I2 = self._I2(ii)
            for jj in range(nb1):
                V[ii, jj] = self.terminal(I2, jj)
        if store_V:
            self.V_all[0] = V.copy()

        # ── backward induction: n = 1, …, N ──
        t0 = _time.time()
        V_new = np.zeros_like(V)

        for n in range(1, p.N + 1):
            for ii in range(nI2):
                I2 = self._I2(ii)
                for jj in range(nb1):
                    b1 = jj

                    # Feasible action range
                    q_max = max(0, min(I2, b1)) if (I2 > 0 and b1 > 0) else 0

                    best_cost = np.inf
                    best_q    = 0

                    for q in range(0, q_max + 1):
                        I2a = I2 - q       # post-action inventory
                        b1a = b1 - q       # post-action backlog

                        # Immediate cost
                        cost_q = self.g(I2, b1, q)

                        # Expected future cost: three transitions from (I2a, b1a)
                        ii0 = self._ii(self._clip_I2(I2a))
                        jj0 = self._clip_b1(b1a)
                        ii2 = self._ii(self._clip_I2(I2a - 1))
                        jj1 = self._clip_b1(b1a + 1)

                        cost_q += (p.p0 * V[ii0, jj0]
                                 + p.p1 * V[ii0, jj1]
                                 + p.p2 * V[ii2, jj0])

                        if cost_q < best_cost:
                            best_cost = cost_q
                            best_q    = q

                    V_new[ii, jj] = best_cost
                    self.policy[n, ii, jj] = best_q

            V[:] = V_new
            if store_V:
                self.V_all[n] = V_new.copy()

            if verbose and (n % 50 == 0 or n == p.N):
                print(f"    period {n}/{p.N}")

        elapsed = _time.time() - t0
        self.V_final = V.copy()
        self._solved = True

        if verbose:
            print(f"    Solved in {elapsed:.1f}s  "
                  f"(state space {nI2}×{nb1} = {nI2*nb1} cells, {p.N} periods)")
        return elapsed

    # ── query helpers ────────────────────────────────────────────
    def get_policy(self, n: int, I2: int, b1: int) -> int:
        """Return q*(n, I₂, b₁) with automatic clipping."""
        assert self._solved, "Call solve() first."
        return int(self.policy[n, self._ii(self._clip_I2(I2)), self._clip_b1(b1)])

    def get_value(self, n: int, I2: int, b1: int) -> float:
        """Return V^n(I₂, b₁)."""
        assert self._solved, "Call solve() first."
        ii = self._ii(self._clip_I2(I2))
        jj = self._clip_b1(b1)
        if self.V_all is not None:
            return float(self.V_all[n, ii, jj])
        elif n == self.p.N:
            return float(self.V_final[ii, jj])
        else:
            raise ValueError("V_all not stored.  Re-solve with store_V=True "
                             "to access intermediate periods.")

    def get_safety_stock(self, n: int, I2: int, b1: int) -> int:
        """Retained inventory after dispatch: I₂ − q*."""
        return I2 - self.get_policy(n, I2, b1)


# ═══════════════════════════════════════════════════════════════════════
#  Quick self-test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    p = Params(N=100, Cf=20, pi1=10, pi2=10, c1=5, c2=5, v2=1)
    dp = TransshipmentDP(p)
    dp.solve(store_V=True)

    I2_0, b1_0 = 15, 0
    print(f"\nV^N({I2_0}, {b1_0}) = {dp.get_value(p.N, I2_0, b1_0):.4f}")
    print(f"q*(N, 15, 5)      = {dp.get_policy(p.N, 15, 5)}")
    print(f"safety(N, 15, 5)  = {dp.get_safety_stock(p.N, 15, 5)}")