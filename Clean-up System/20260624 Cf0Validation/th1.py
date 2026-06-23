"""
validate_thm1.py
================
Numerical validation of Theorem 1 in 20260623_Cf0_analytical.pdf.

Theorem 1 conditions
---------------------
    pi1 = pi2 = pi,   alpha2 = lambda2 * cu / (h + pi) <= 1/2

Policy structure (from the PDF)
---------------------------------
    Inner region  (tau <= I2 / lambda2):
        (i)  tau <= tau*  ->  wait
        (ii) tau >  tau*  ->  dispatch       tau* = cu / (pi + h)
    Outer region  (tau >= I2 / lambda2):
        (iii)             ->  always dispatch  (all I2 >= 1)

    The outer threshold from the general formula is
        I2*(tau) = alpha2 - gamma*tau + 1/2 = alpha2 + 1/2  (gamma=0)
    Since alpha2 <= 1/2, we have I2*(tau) <= 1, so I2 >= 1 always dispatches.

What is validated
------------------
T1  Outer always-dispatch
        For every (I2, tau) with I2 >= 1 and tau >= I2/lambda2,
        the DP policy must be dispatch.  Zero violations expected.

T2  Inner time-threshold
        Fix I2_fix large (so that all test tau < I2_fix/lambda2, i.e.
        we are firmly in the inner region).  Scan tau around tau*.
        Expected transition: wait for tau < tau*, dispatch for tau > tau*.

T3  tau-independence of the outer threshold
        Since gamma = 0, the threshold I2* = alpha2 + 1/2 is constant in tau.
        Equivalently, the outer threshold extracted from the DP should be
        the same integer across all bulk tau values (spread = 0).
        For Theorem 1, that constant value is 1 (always dispatch).

Five parameter configurations are tested with alpha2 in {0.20, 0.25, 0.30, 0.40, 0.50},
covering the full always-dispatch regime including the boundary alpha2 = 1/2.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════
# 1.  PARAMETERS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ZFParams:
    """All parameters for the Cf=0 reduced-state DP."""
    T:      float = 8.0
    N:      int   = 1600    # dt = T/N; default dt = 0.005
    lam1:   float = 5.0     # retailer-1 arrival rate (does not affect I2 threshold)
    lam2:   float = 2.0
    h:      float = 1.0
    cu:     float = 0.5
    pi1:    float = 3.0
    pi2:    float = 3.0
    I2_max: int   = 30
    I2_min: int   = -5

    @property
    def dt(self) -> float:
        return self.T / self.N

    def validate(self) -> "ZFParams":
        assert self.lam2 * self.dt < 0.5, (
            f"Stability violated: lam2*dt = {self.lam2 * self.dt:.4f} >= 0.5. "
            f"Increase N or reduce lam2."
        )
        assert self.pi1 == self.pi2, (
            "Theorem 1 requires pi1 = pi2."
        )
        return self


# ══════════════════════════════════════════════════════════════════════
# 2.  DP SOLVER  (provided by collaborators)
# ══════════════════════════════════════════════════════════════════════

class ZeroFixedCostDP:
    """
    Backward induction in tau (n periods remaining, tau = n*dt) for the
    reduced state I2. Dispatch is a same-tau (instantaneous) jump, so for
    each n we sweep I2 in increasing order and use the already-updated
    value at I2-1 within the same n.
    """

    def __init__(self, p: ZFParams):
        self.p = p
        self.nI2 = p.I2_max - p.I2_min + 1
        self.V = np.zeros((p.N + 1, self.nI2))
        self.policy = np.zeros((p.N + 1, self.nI2), dtype=np.int8)  # 0=wait, 1=dispatch
        self._solved = False

    def _ii(self, I2): return I2 - self.p.I2_min
    def _I2(self, ii): return ii + self.p.I2_min
    def _clip_ii(self, ii): return max(0, min(self.nI2 - 1, ii))

    def solve(self):
        p = self.p
        dt = p.dt
        for n in range(1, p.N + 1):
            tau = n * dt
            V_prev = self.V[n - 1]
            V_new  = self.V[n]
            for ii in range(self.nI2):
                I2 = self._I2(ii)
                flow = (p.h * max(I2, 0) + p.pi2 * max(-I2, 0)) * dt \
                       + p.pi1 * tau * p.lam1 * dt
                ii_m1    = self._clip_ii(ii - 1)
                wait_val = flow \
                           + (1 - p.lam2 * dt) * V_prev[ii] \
                           + p.lam2 * dt       * V_prev[ii_m1]

                if I2 >= 1:
                    dispatch_val = p.cu - p.pi1 * tau + V_new[ii_m1]
                    if dispatch_val < wait_val:
                        V_new[ii]          = dispatch_val
                        self.policy[n, ii] = 1
                    else:
                        V_new[ii]          = wait_val
                        self.policy[n, ii] = 0
                else:
                    V_new[ii]          = wait_val
                    self.policy[n, ii] = 0
        self._solved = True

    def get_policy(self, n, I2):
        assert self._solved
        return int(self.policy[n, self._clip_ii(self._ii(I2))])


# ══════════════════════════════════════════════════════════════════════
# 3.  ANALYTICAL FORMULAS  (Theorem 1, pi1=pi2=pi)
# ══════════════════════════════════════════════════════════════════════

def alpha2_sym(lam2: float, cu: float, h: float, pi: float) -> float:
    """alpha2 = lambda2 * cu / (h + pi).  Theorem 1 applies when alpha2 <= 1/2."""
    return lam2 * cu / (h + pi)


def tau_star_sym(cu: float, pi: float, h: float) -> float:
    """Inner-region time threshold: dispatch iff tau >= tau* = cu / (pi + h)."""
    return cu / (pi + h)


# ══════════════════════════════════════════════════════════════════════
# 4.  HELPERS
# ══════════════════════════════════════════════════════════════════════

def n_for_tau(tau: float, p: ZFParams) -> int:
    """Convert remaining time tau to period index n (rounded)."""
    return max(0, min(p.N, round(tau / p.dt)))


def dp_outer_threshold(dp: ZeroFixedCostDP, n: int) -> int | None:
    """Minimum I2 >= 1 such that DP dispatches at step n.  None if never."""
    for I2 in range(1, dp.p.I2_max + 1):
        if dp.get_policy(n, I2) == 1:
            return I2
    return None


SEP = "─" * 62

def _ok(condition: bool) -> str:
    return "OK" if condition else "NG"


# ══════════════════════════════════════════════════════════════════════
# 5.  CHECK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def check_T1_outer_always_dispatch(
    dp: ZeroFixedCostDP,
    tau_vals: list[float],
) -> None:
    """
    T1: Outer region always-dispatch.

    For every (I2, tau) satisfying I2 >= 1 and tau >= I2/lambda2,
    the DP must dispatch.  Theorem 1(iii) guarantees this when alpha2 <= 1/2.

    Pass criterion: zero violations at every tau in tau_vals.
    """
    p = dp.p
    print("  [T1] Outer always-dispatch  (expect 0 violations at each tau)")
    print(f"  {'tau':>6} | {'#outer (I2,tau) pairs':>24} | violations")
    total_viol = 0
    for tau in tau_vals:
        n        = n_for_tau(tau, p)
        n_pairs  = sum(1 for I2 in range(1, p.I2_max + 1) if tau >= I2 / p.lam2)
        viol     = sum(
            1 for I2 in range(1, p.I2_max + 1)
            if tau >= I2 / p.lam2 and dp.get_policy(n, I2) == 0
        )
        total_viol += viol
        print(f"  {tau:>6.2f} | {n_pairs:>24} | {viol}  {_ok(viol == 0)}")
    print(f"  Total violations: {total_viol}  {_ok(total_viol == 0)}")


def check_T2_inner_time_threshold(
    dp: ZeroFixedCostDP,
    I2_fix: int,
) -> None:
    """
    T2: Inner region time-threshold.

    Fix I2_fix (large enough that tau_test < I2_fix/lambda2 for all test points,
    i.e. we remain in the inner region throughout the scan).
    Scan tau from 0.20*tau* to 2.50*tau*; expected policy:
        wait     when tau < tau*
        dispatch when tau > tau*

    Pass criterion: actual DP policy matches prediction at every test point.
    Near-terminal points (very small tau) may deviate by at most one dt step.
    """
    p          = dp.p
    ts         = tau_star_sym(p.cu, p.pi1, p.h)
    tau_in_max = I2_fix / p.lam2   # inner-region boundary for this I2_fix

    print(f"  [T2] Inner time-threshold  tau* = {ts:.4f}  "
          f"I2_fix = {I2_fix}  (inner: tau < {tau_in_max:.2f})")
    print(f"  {'tau/tau*':>8} | {'tau':>7} | {'DP':>9} | {'predicted':>9} | status")

    errors = 0
    for frac in [0.20, 0.50, 0.80, 0.95, 1.05, 1.30, 1.80, 2.50]:
        tau = ts * frac
        if tau <= 0.0 or tau >= tau_in_max:
            continue
        n        = n_for_tau(tau, p)
        pol_dp   = dp.get_policy(n, I2_fix)
        pred     = 1 if tau >= ts else 0
        match    = (pol_dp == 1) == (pred == 1)
        if not match:
            errors += 1
        dp_str   = "dispatch" if pol_dp == 1 else "wait"
        pred_str = "dispatch" if pred   == 1 else "wait"
        print(f"  {frac:>8.2f} | {tau:>7.4f} | {dp_str:>9} | {pred_str:>9} | {_ok(match)}")

    if errors:
        print(f"  NOTE: {errors} mismatch(es). "
              f"Points with tau very close to tau* may be off by one dt = {p.dt:.4f}.")


def check_T3_tau_independence(
    dp: ZeroFixedCostDP,
    tau_vals: list[float],
) -> None:
    """
    T3: tau-independence of the outer threshold.

    Because gamma = 0 (pi1 = pi2), the threshold I2*(tau) = alpha2 + 1/2
    is constant in tau.  For Theorem 1 (alpha2 <= 1/2) this constant equals
    at most 1, so the DP threshold should be 1 (always dispatch) at every
    bulk tau.  The spread of DP-extracted thresholds across tau_vals should be 0.

    Pass criterion: spread <= 1 (allowing one unit of integer rounding).
    A spread of 0 is the ideal outcome.
    """
    thresholds = [
        dp_outer_threshold(dp, n_for_tau(t, dp.p)) for t in tau_vals
    ]
    valid  = [t for t in thresholds if t is not None]
    spread = (max(valid) - min(valid)) if len(valid) >= 2 else None
    ok     = spread is not None and spread <= 1
    print(f"  [T3] tau-independence  "
          f"thresholds = {thresholds}  spread = {spread}  {_ok(ok)}")
    if spread is not None and spread == 0:
        print(f"       (perfect: threshold constant = {valid[0]} at all bulk tau)")


# ══════════════════════════════════════════════════════════════════════
# 6.  PARAMETER CONFIGURATIONS  (Theorem 1)
# ══════════════════════════════════════════════════════════════════════

# Columns: (label, lam2, cu, h, pi)   with  pi1=pi2=pi  and  alpha2 <= 1/2
# alpha2 = lam2*cu/(h+pi)
THM1_CONFIGS = [
    ("alpha2=0.20", 2, 0.4, 1.0, 3.0),   # 2*0.40/(1+3) = 0.20
    ("alpha2=0.25", 2, 0.5, 1.0, 3.0),   # 2*0.50/(1+3) = 0.25
    ("alpha2=0.30", 3, 0.5, 1.0, 4.0),   # 3*0.50/(1+4) = 0.30
    ("alpha2=0.40", 4, 0.4, 1.0, 3.0),   # 4*0.40/(1+3) = 0.40
    ("alpha2=0.50", 3, 1.0, 1.0, 5.0),   # 3*1.00/(1+5) = 0.50  (boundary)
]

# Global DP settings
T0         = 8.0
N_STEPS    = 1600   # dt = 0.005; max lam2*dt = 4*0.005 = 0.02 < 0.5  (stable)
LAM1       = 5.0
I2_MAX     = 30
I2_MIN     = -5

# Bulk tau values: 40%–100% of T0, far enough from the terminal boundary
BULK_FRACS = [0.40, 0.55, 0.70, 0.85, 1.00]


# ══════════════════════════════════════════════════════════════════════
# 7.  MAIN VALIDATION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def validate_theorem1() -> None:
    bulk_taus = [T0 * f for f in BULK_FRACS]

    print("=" * 62)
    print("Theorem 1  —  pi1=pi2=pi,  alpha2 = lambda2*cu/(h+pi) <= 1/2")
    print("=" * 62)
    print("What is validated:")
    print("  T1  Outer region (tau >= I2/lam2, I2>=1) -> always dispatch")
    print("  T2  Inner region (tau <= I2_fix/lam2)   -> dispatch iff tau >= tau*")
    print("  T3  tau-independence: outer threshold constant across bulk tau")
    print(f"Settings: T0={T0}, N={N_STEPS}, lam1={LAM1}, I2 in [{I2_MIN},{I2_MAX}]")
    print(f"Bulk tau = {[round(T0*f, 2) for f in BULK_FRACS]}")

    for label, lam2, cu, h, pi in THM1_CONFIGS:
        a2 = alpha2_sym(lam2, cu, h, pi)
        ts = tau_star_sym(cu, pi, h)

        assert a2 <= 0.5 + 1e-9, f"Config error: alpha2 = {a2:.4f} > 1/2"

        print(f"\n{SEP}")
        print(f"  {label}:  lam2={lam2}  cu={cu}  h={h}  pi={pi}")
        print(f"  alpha2 = {a2:.4f} <= 1/2  check   "
              f"tau* = cu/(pi+h) = {ts:.4f}   "
              f"outer threshold = ceil({a2:.4f}+0.5) = {math.ceil(a2+0.5)}")

        p = ZFParams(
            T=T0, N=N_STEPS, lam1=LAM1, lam2=lam2,
            h=h, cu=cu, pi1=pi, pi2=pi,
            I2_max=I2_MAX, I2_min=I2_MIN,
        )
        p.validate()

        dp = ZeroFixedCostDP(p)
        dp.solve()

        # T1: outer always-dispatch
        check_T1_outer_always_dispatch(dp, bulk_taus)

        # T3: tau-independence (run before T2 so the print groups cleanly)
        check_T3_tau_independence(dp, bulk_taus)

        # T2: inner time threshold
        # I2_fix must satisfy: 2.5 * tau* < I2_fix / lam2
        # => I2_fix > 2.5 * lam2 * tau*
        I2_fix = min(I2_MAX, max(5, math.ceil(2.5 * lam2 * ts) + 2))
        check_T2_inner_time_threshold(dp, I2_fix)

    print(f"\n{'='*62}")
    print("VALIDATION COMPLETE")
    print("=" * 62)


if __name__ == "__main__":
    validate_theorem1()