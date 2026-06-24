"""
validate_thm2.py
================
Validation of Theorem 2 (Cf=0, pi1=pi2=pi, alpha2 > 1/2).

Reuses from validate_thm1.py:
    ZFParams, ZeroFixedCostDP,
    alpha2_sym, tau_star_sym,
    n_for_tau, dp_outer_threshold,
    check_T2_inner_time_threshold, check_T3_tau_independence,
    SEP, _ok, T0, N_STEPS, LAM1, I2_MAX, I2_MIN, BULK_FRACS

Theorem 2 policy structure
---------------------------
    Inner  (tau <= I2/lam2):  dispatch iff tau >= tau* = cu/(pi+h)   [same as Thm 1]
    Outer  (tau >= I2/lam2):  dispatch iff I2 >= ceil(alpha2 + 1/2)  [tau-independent, gamma=0]

What is validated
------------------
T1  Outer threshold value: DP threshold at bulk tau matches ceil(alpha2 + 1/2), |diff| <= 1.
T3  tau-independence: spread of DP threshold across bulk tau is 0.
T2  Inner time threshold: same as Theorem 1 (dispatch iff tau >= tau*).
"""

import math
from th1 import (
    ZFParams, ZeroFixedCostDP,
    alpha2_sym, tau_star_sym,
    n_for_tau, dp_outer_threshold,
    check_T2_inner_time_threshold, check_T3_tau_independence,
    SEP, _ok,
    T0, N_STEPS, LAM1, I2_MAX, I2_MIN, BULK_FRACS,
)


# ── parameter configurations: pi1=pi2=pi, alpha2 > 1/2 ───────────────
# (label, lam2, cu, h, pi)    alpha2 = lam2*cu/(h+pi)
THM2_CONFIGS = [
    ("alpha2=1.00", 4, 1.0, 1.0, 3.0),   # 4/4   = 1.00  -> ceil(1.5) = 2
    ("alpha2=1.50", 3, 1.0, 1.0, 1.0),   # 3/2   = 1.50  -> ceil(2.0) = 2
    ("alpha2=2.50", 5, 1.0, 1.0, 1.0),   # 5/2   = 2.50  -> ceil(3.0) = 3
    ("alpha2=3.00", 6, 2.0, 1.0, 3.0),   # 12/4  = 3.00  -> ceil(3.5) = 4
]


# ── new check: outer threshold value + tau-independence ───────────────

def check_T1_outer_threshold(dp: ZeroFixedCostDP,
                              analytic_threshold: int,
                              tau_vals: list[float]) -> None:
    """
    T1: Outer region threshold = ceil(alpha2 + 1/2), tau-independent.
    Extracts DP threshold at each bulk tau and compares to the analytical value.
    Pass criterion: |dp_threshold - analytic_threshold| <= 1 at every tau,
                    spread across tau_vals <= 1.
    """
    print(f"  [T1] Outer threshold  analytical = ceil(alpha2+1/2) = {analytic_threshold}")
    print(f"  {'tau':>6} | {'DP thr':>7} | {'analytic':>9} | {'diff':>5} | status")
    diffs = []
    for tau in tau_vals:
        n     = n_for_tau(tau, dp.p)
        th_dp = dp_outer_threshold(dp, n)
        d     = (th_dp - analytic_threshold) if th_dp is not None else None
        ok    = d is not None and abs(d) <= 1
        if d is not None:
            diffs.append(abs(d))
        print(f"  {tau:>6.2f} | {str(th_dp):>7} | {analytic_threshold:>9} | {str(d):>5} | {_ok(ok)}")
    valid = [dp_outer_threshold(dp, n_for_tau(t, dp.p)) for t in tau_vals]
    valid = [v for v in valid if v is not None]
    spread = (max(valid) - min(valid)) if len(valid) >= 2 else None
    print(f"  spread = {spread}  {_ok(spread is not None and spread <= 1)}"
          f"   MAE = {sum(diffs)/len(diffs):.3f}" if diffs else "")


# ── main ──────────────────────────────────────────────────────────────

def validate_theorem2() -> None:
    bulk_taus = [T0 * f for f in BULK_FRACS]

    print("=" * 62)
    print("Theorem 2  —  pi1=pi2=pi,  alpha2 > 1/2")
    print("  T1  outer threshold = ceil(alpha2 + 1/2), tau-independent")
    print("  T3  tau-independence: spread of DP threshold across bulk tau")
    print("  T2  inner: dispatch iff tau >= tau* = cu/(pi+h)")
    print("=" * 62)

    for label, lam2, cu, h, pi in THM2_CONFIGS:
        a2  = alpha2_sym(lam2, cu, h, pi)
        ts  = tau_star_sym(cu, pi, h)
        thr = math.ceil(a2 + 0.5)
        assert a2 > 0.5, f"Config error: alpha2 = {a2:.4f} <= 1/2"

        print(f"\n{SEP}")
        print(f"  {label}:  lam2={lam2}  cu={cu}  h={h}  pi={pi}")
        print(f"  alpha2 = {a2:.4f} > 1/2  check   "
              f"thr = ceil({a2:.4f}+0.5) = {thr}   tau* = {ts:.4f}")

        p = ZFParams(T=T0, N=N_STEPS, lam1=LAM1, lam2=lam2,
                     h=h, cu=cu, pi1=pi, pi2=pi,
                     I2_max=I2_MAX, I2_min=I2_MIN)
        p.validate()
        dp = ZeroFixedCostDP(p)
        dp.solve()

        check_T1_outer_threshold(dp, thr, bulk_taus)
        check_T3_tau_independence(dp, bulk_taus)

        I2_fix = min(I2_MAX, max(5, math.ceil(2.5 * lam2 * ts) + 2))
        check_T2_inner_time_threshold(dp, I2_fix)

    print(f"\n{'='*62}")
    print("VALIDATION COMPLETE")
    print("=" * 62)


if __name__ == "__main__":
    validate_theorem2()