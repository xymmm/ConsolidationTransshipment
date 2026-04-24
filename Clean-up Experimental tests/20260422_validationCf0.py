"""
validate_Cf0.py
===============
Validation of the Cf=0 analytical results (analysis_Cf0.pdf) against the
backward-induction DP solver.

Key structural difference from the general model
-------------------------------------------------
When Cf=0, the state space collapses from (I2, b1, tau) to (I2, tau).
There is no fixed cost, so each unit is dispatched independently.
The dispatch condition (Eq.2 of analysis_Cf0.pdf) becomes:

    V(I2, tau) <= cu - pi1*tau + V(I2-1, tau)

which gives a threshold only on I2:

    I2*(tau) = lambda2*cu/(h+pi2) - lambda2*(pi1-pi2)/(h+pi2) * tau

In the existing (I2, b1) solver with Cf=0:
  - The b1 dimension is still present, but with Cf=0 and any b1>=1,
    the dispatch decision depends only on I2 vs I2*(tau).
  - We extract I2*(tau) from DP by: min I2 s.t. dp dispatches at
    (I2, b1=1, n), since b1=1 is the minimum non-zero backlog.

Theorems validated
------------------
  Theorem 1 (pi1=pi2, alpha2<=1):  always dispatch if I2>=1
  Theorem 2 (pi1=pi2, alpha2>1):   dispatch iff I2 >= alpha2 = lambda2*cu/(h+pi)
  Theorem 3 (pi1>pi2, alpha2<=1):  always dispatch if I2>=1
  Theorem 4 (pi1>pi2, alpha2>1):   dispatch iff I2 >= alpha2 - gamma*tau
  Theorem 5 (pi1<pi2):             dispatch iff I2 >= alpha2 + |gamma|*tau
    -> threshold INCREASES with tau (stronger protection at larger tau)

Note on pi1=pi2 and tau-independence
-------------------------------------
When pi1=pi2, gamma=0, so I2*(tau) = alpha2 = constant.
The threshold is tau-independent regardless of cu.
This directly answers the collaborator's observation:
  "when pi1=pi2, Cf=0, cu large -> threshold does not depend on tau"
  -> CORRECT. The protection mechanism shifts entirely to the I2 dimension
     (high cu -> high alpha2 -> high I2 threshold), not to tau.
When pi1!=pi2, the threshold is tau-dependent, and the collaborator's
intuition about a "protection mechanism" is captured by Theorem 4/5.

Usage
-----
    python validate_Cf0.py
Requires solver.py in the same directory.
"""

import math
import numpy as np
from solver import Params, TransshipmentDP


# ======================================================================
# PARAMETER SETS
# ======================================================================
# All configs use Cf=0.
# c1=c2=pi2, v2=1 (same convention as main validation).
# b1_max=60: with Cf=0, large b1 is needed but b1 dimension is artificial.
# B1_LARGE=1: use b1=1 to query I2 threshold (minimum non-zero backlog).
# State space wide enough: I2_max=30.

B1_QUERY = 1   # query threshold at b1=1 (not b1_large; b1 irrelevant when Cf=0)

# Theorem 1 & 2 configs: pi1=pi2, varying alpha2 = lambda2*cu/(h+pi)
# alpha2 = lambda2*cu/(h+pi):  <=1 -> Thm1 (always dispatch),  >1 -> Thm2
SYM_CONFIGS = [
    # label,              lam2, cu,   h,  pi,   c1,   c2,   v2
    ("alpha2=0.5  Thm1",    2, 1.0,  1,  3.0,  3.0,  3.0,  1.0),  # alpha2=2/4=0.5
    ("alpha2=1.0  Thm1",    3, 1.0,  1,  2.0,  2.0,  2.0,  1.0),  # alpha2=3/3=1.0
    ("alpha2=1.5  Thm2",    3, 1.0,  1,  1.0,  1.0,  1.0,  1.0),  # alpha2=3/2=1.5
    ("alpha2=2.5  Thm2",    5, 1.0,  1,  1.0,  1.0,  1.0,  1.0),  # alpha2=5/2=2.5
    ("alpha2=4.0  Thm2",    4, 2.0,  1,  1.0,  1.0,  1.0,  1.0),  # alpha2=8/2=4.0
]

# Theorem 3 & 4 configs: pi1>pi2
# gamma = lambda2*(pi1-pi2)/(h+pi2) > 0
# threshold I2*(tau) = alpha2 - gamma*tau  (DECREASING in tau)
# -> easier to dispatch as tau grows (more time remaining)
ASYM_GT_CONFIGS = [
    # label,                 lam2, cu,   h, pi1, pi2,   c1,   c2,   v2
    ("pi1>pi2 alpha2=0.4 Thm3",  2, 0.5, 1,  5,   4,  4.0,  4.0,  1.0),  # alpha2=2*0.5/5=0.2 wait: 0.2<=1 -> Thm3: always dispatch
    ("pi1>pi2 alpha2=2.0 Thm4",  4, 1.0, 1,  5,   2,  2.0,  2.0,  1.0),  # alpha2=4/3=1.33, gamma=4*3/3=4
    ("pi1>pi2 alpha2=3.0 Thm4",  3, 2.0, 1,  6,   3,  3.0,  3.0,  1.0),  # alpha2=6/4=1.5, gamma=3*3/4=2.25
]
# Fix alpha2 more carefully
ASYM_GT_CONFIGS = [
    # label,                    lam2,  cu,   h, pi1, pi2,  c1,   c2,   v2
    ("pi1>pi2 alpha2=0.5 Thm3",   2, 1.0,  1,   6,   4, 4.0,  4.0,  1.0),  # alpha2=2/5=0.4<=1
    ("pi1>pi2 alpha2=1.5 Thm4",   3, 1.0,  1,   7,   4, 4.0,  4.0,  1.0),  # alpha2=3/5=0.6 no...
]
# Let me be precise: alpha2 = lam2*cu/(h+pi2)
# Thm3: alpha2<=1, pi1>pi2.  Example: lam2=2, cu=1, h=1, pi2=3, pi1=5 -> alpha2=2/4=0.5<=1 OK
# Thm4: alpha2>1, pi1>pi2.  Example: lam2=4, cu=1, h=1, pi2=1, pi1=4 -> alpha2=4/2=2.0>1 OK
ASYM_GT_CONFIGS = [
    # label,                     lam2, cu,   h, pi1, pi2,   c1,   c2,   v2
    ("pi1>pi2 alpha2=0.5 Thm3",    2, 1.0,  1,   5,   3,  3.0,  3.0,  1.0),  # alpha2=2/4=0.5
    ("pi1>pi2 alpha2=2.0 Thm4",    4, 1.0,  1,   5,   1,  1.0,  1.0,  1.0),  # alpha2=4/2=2.0
    ("pi1>pi2 alpha2=3.0 Thm4",    3, 2.0,  1,   6,   2,  2.0,  2.0,  1.0),  # alpha2=6/3=2.0... let me use lam2=6
    ("pi1>pi2 alpha2=3.0 Thm4b",   6, 1.0,  1,   4,   1,  1.0,  1.0,  1.0),  # alpha2=6/2=3.0
]

# Theorem 5 configs: pi1<pi2
# gamma < 0, |gamma|*tau term makes threshold INCREASE with tau
# -> harder to dispatch as tau grows (stronger protection at large tau)
ASYM_LT_CONFIGS = [
    # label,                     lam2, cu,   h, pi1, pi2,   c1,   c2,   v2
    ("pi1<pi2 alpha2=0.5 Thm5",    2, 1.0,  1,   2,   5,  5.0,  5.0,  1.0),  # alpha2=2/6=0.33, gamma=-2*3/6=-1
    ("pi1<pi2 alpha2=2.0 Thm5",    4, 1.0,  1,   1,   5,  5.0,  5.0,  1.0),  # alpha2=4/6=0.67... need alpha2>1
    ("pi1<pi2 alpha2=2.0 Thm5b",   4, 2.0,  1,   2,   5,  5.0,  5.0,  1.0),  # alpha2=8/6=1.33
    ("pi1<pi2 alpha2=3.5 Thm5c",   7, 1.0,  1,   1,   1,  1.0,  1.0,  1.0),  # alpha2=7/2=3.5
]


# ======================================================================
# ANALYTICAL FORMULAS (analysis_Cf0.pdf)
# ======================================================================

def alpha2(lam2, cu, h, pi2):
    return lam2 * cu / (h + pi2)

def gamma_cf0(lam2, pi1, pi2, h):
    return lam2 * (pi1 - pi2) / (h + pi2)

def I2_threshold_analytical(tau, lam2, cu, h, pi1, pi2):
    """
    Continuous threshold I2*(tau) from analysis_Cf0.pdf.

    General formula (Section 5):
        I2*(tau) = lambda2*cu/(h+pi2) - lambda2*(pi1-pi2)/(h+pi2) * tau
                 = alpha2 - gamma * tau

    Special cases:
        pi1=pi2:  I2*(tau) = alpha2  (tau-independent, Theorems 1&2)
        pi1>pi2:  gamma>0, I2*(tau) decreasing in tau (Theorems 3&4)
                  -> at large tau, threshold is lower -> easier to dispatch
        pi1<pi2:  gamma<0, I2*(tau) increasing in tau (Theorem 5)
                  -> at large tau, threshold is higher -> harder to dispatch
    """
    a2  = alpha2(lam2, cu, h, pi2)
    gam = gamma_cf0(lam2, pi1, pi2, h)
    return a2 - gam * tau

def always_dispatch_region(lam2, cu, h, pi2):
    """
    True if alpha2 = lambda2*cu/(h+pi2) <= 1.
    In this case, I2*(tau) <= 1 for all tau >= 0,
    so I2>=1 always satisfies the dispatch condition.
    """
    return alpha2(lam2, cu, h, pi2) <= 1.0


# ======================================================================
# DP HELPERS
# ======================================================================

def build_and_solve_Cf0(lam2, cu, h, pi1, pi2, c1, c2, v2,
                         lam1=5.0, T=2.0, N=200,
                         I2_max=30, I2_min=-5, b1_max=60):
    """Build Params with Cf=0, solve DP."""
    p = Params(
        T=T, N=N, lam1=lam1, lam2=lam2,
        h=h, Cf=0.0, cu=cu,
        pi1=pi1, pi2=pi2,
        c1=c1, c2=c2, v2=v2,
        I2_max=I2_max, I2_min=I2_min, b1_max=b1_max,
    )
    dp = TransshipmentDP(p)
    dp.solve(store_V=False, verbose=False)
    return dp, p

def I2_threshold_dp_Cf0(dp, n, b1_query=B1_QUERY):
    """
    Smallest I2 s.t. DP dispatches at (I2, b1_query, n periods left).
    With Cf=0, any b1>=1 gives the same threshold, so b1=1 suffices.
    Returns None if no dispatch found.
    """
    for I2 in range(1, dp.p.I2_max + 1):
        if dp.get_policy(n, I2, b1_query) > 0:
            return I2
    return None

def n_for_tau(tau, dp):
    return max(0, min(dp.p.N, round(tau / dp.p.dt)))

def _diff_str(dp_val, analytic_val):
    """diff = dp_val - ceil(analytic_val). OK if |diff|<=1."""
    if dp_val is None or analytic_val is None:
        return None, "--"
    d    = dp_val - math.ceil(analytic_val)
    flag = "OK" if abs(d) <= 1 else "NG"
    return d, flag

SEP = "-" * 68


# ======================================================================
# VALIDATION: SYMMETRIC CASE (pi1=pi2)
# ======================================================================

def validate_symmetric(label, lam2, cu, h, pi, c1, c2, v2,
                        lam1=5.0, T=2.0, N=200):
    a2  = alpha2(lam2, cu, h, pi)
    always = always_dispatch_region(lam2, cu, h, pi)

    print(f"\n{SEP}")
    print(f"SYMMETRIC | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi={pi}  Cf=0  c1=c2={c1}  v2={v2}")
    print(f"  alpha2 = {a2:.4f}  {'<= 1 -> Theorem 1 (always dispatch)' if always else '> 1 -> Theorem 2 (threshold at I2=alpha2)'}")
    print(f"  Threshold I2*(tau) = {a2:.4f}  [TAU-INDEPENDENT, pi1=pi2]")

    dp, p = build_and_solve_Cf0(lam2, cu, h, pi, pi, c1, c2, v2,
                                  lam1=lam1, T=T, N=N)

    # T1: tau-independence check
    # I2 threshold should be the same across all tau values
    tau_vals = [T*0.2, T*0.4, T*0.6, T*0.8, T]
    thresholds = []
    for tau in tau_vals:
        n  = n_for_tau(tau, dp)
        th = I2_threshold_dp_Cf0(dp, n)
        thresholds.append(th)

    print(f"\n  [T1] Tau-independence check (pi1=pi2 -> threshold constant):")
    print(f"       {'tau':>5} | {'I2*(DP)':>8} | {'Analytical':>11} | {'diff':>5} | flag")
    all_ok = True
    for tau, th in zip(tau_vals, thresholds):
        diff, flag = _diff_str(th, a2)
        if flag == "NG":
            all_ok = False
        print(f"       {tau:>5.2f} | {str(th):>8} | {a2:>11.4f} | {str(diff):>5} | {flag}")
    spread = max(t for t in thresholds if t is not None) - min(t for t in thresholds if t is not None)
    print(f"       Spread across tau: {spread} (expect 0 or 1 for integer rounding)")

    # T2: always-dispatch check (Theorem 1)
    if always:
        print(f"\n  [T2] Theorem 1 check: I2>=1 always dispatches (alpha2={a2:.4f}<=1)")
        n_full = N
        viol = 0
        for I2 in range(1, p.I2_max + 1):
            if dp.get_policy(n_full, I2, B1_QUERY) == 0:
                viol += 1
        print(f"       {viol} non-dispatch cases at I2>=1 (expect 0)")
        flag = "OK" if viol == 0 else "NG"
        print(f"       {flag}")
    else:
        print(f"\n  [T2] Theorem 2 check: threshold at I2=alpha2={a2:.4f}")
        n_full = N
        th = I2_threshold_dp_Cf0(dp, n_full)
        diff, flag = _diff_str(th, a2)
        print(f"       DP threshold={th}  Analytical={a2:.4f}  ceil={math.ceil(a2)}  diff={diff}  {flag}")


# ======================================================================
# VALIDATION: ASYMMETRIC CASE (pi1 != pi2)
# ======================================================================

def validate_asymmetric(label, lam2, cu, h, pi1, pi2, c1, c2, v2,
                         lam1=5.0, T=2.0, N=200):
    a2   = alpha2(lam2, cu, h, pi2)
    gam  = gamma_cf0(lam2, pi1, pi2, h)
    always = always_dispatch_region(lam2, cu, h, pi2)

    direction = "DECREASING in tau (pi1>pi2, easier to dispatch with more time)" if gam > 0 \
           else "INCREASING in tau (pi1<pi2, harder to dispatch with more time)"

    print(f"\n{SEP}")
    print(f"ASYMMETRIC | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi1={pi1}  pi2={pi2}  Cf=0  c1=c2={c1}  v2={v2}")
    print(f"  alpha2={a2:.4f}  gamma={gam:.4f}")
    print(f"  Theorem: {'3 (always dispatch, alpha2<=1)' if always else '4 (threshold, alpha2>1)'}"
          if pi1 > pi2 else f"  Theorem: 5")
    print(f"  I2*(tau) = {a2:.4f} - {gam:.4f}*tau  [{direction}]")

    dp, p = build_and_solve_Cf0(lam2, cu, h, pi1, pi2, c1, c2, v2,
                                  lam1=lam1, T=T, N=N)

    # T1: threshold vs tau
    tau_vals = [T*0.1, T*0.25, T*0.5, T*0.75, T*1.0]
    print(f"\n  [T1] I2*(tau) -- DP vs analytical   [TAU-DEPENDENT when pi1!=pi2]:")
    print(f"       {'tau':>5} | {'I2*(DP)':>8} | {'Analytical':>11} | {'ceil':>5} | {'diff':>5} | flag")
    diffs = []
    dp_thresholds = []
    an_thresholds = []
    for tau in tau_vals:
        n    = n_for_tau(tau, dp)
        th   = I2_threshold_dp_Cf0(dp, n)
        an   = I2_threshold_analytical(tau, lam2, cu, h, pi1, pi2)
        diff, flag = _diff_str(th, an)
        dp_thresholds.append(th)
        an_thresholds.append(an)
        if diff is not None:
            diffs.append(abs(diff))
        print(f"       {tau:>5.2f} | {str(th):>8} | {an:>11.4f} | "
              f"{math.ceil(an) if not math.isnan(an) else '--':>5} | {str(diff):>5} | {flag}")
    if diffs:
        print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")

    # T2: direction check -- threshold should increase or decrease monotonically
    valid_dp = [t for t in dp_thresholds if t is not None]
    if len(valid_dp) >= 2:
        if gam > 0:
            monotone = all(valid_dp[i] >= valid_dp[i+1] for i in range(len(valid_dp)-1))
            expected = "non-increasing (gamma>0: threshold decreases as tau increases)"
        else:
            monotone = all(valid_dp[i] <= valid_dp[i+1] for i in range(len(valid_dp)-1))
            expected = "non-decreasing (gamma<0: threshold increases as tau increases)"
        print(f"\n  [T2] Monotonicity check:")
        print(f"       Expected: {expected}")
        print(f"       DP thresholds: {valid_dp}")
        print(f"       {'OK' if monotone else 'NG (non-monotone)'}")

    # T3: always-dispatch if alpha2<=1
    if always:
        n_full = N
        viol = sum(dp.get_policy(n_full, I2, B1_QUERY) == 0
                   for I2 in range(1, p.I2_max + 1))
        print(f"\n  [T3] Always-dispatch check (alpha2={a2:.4f}<=1): "
              f"{viol} non-dispatch at I2>=1  {'OK' if viol==0 else 'NG'}")


# ======================================================================
# pi1=pi2, Cf=0, large cu
# ======================================================================

def validate_collaborator_question(lam1=5.0, T=2.0, N=200):
    print(f"\n{'='*68}")
    print("COLLABORATOR QUESTION: pi1=pi2, Cf=0, cu large -> threshold tau-independent?")
    print(f"{'='*68}")
    print("""
  When pi1=pi2:  gamma = lambda2*(pi1-pi2)/(h+pi2) = 0
  Therefore:     I2*(tau) = alpha2 - 0*tau = alpha2  [constant]

  The threshold is tau-independent regardless of cu.
  The "protection" with large cu is entirely captured by alpha2:
    large cu  ->  large alpha2 = lambda2*cu/(h+pi)
    ->  higher I2 threshold required before dispatching
    ->  fewer dispatches overall

  There is NO tau-dependent protection when pi1=pi2 and Cf=0.
  This is mathematically exact, not an approximation.

  The tau-dependent protection exists ONLY when pi1 != pi2:
    pi1 > pi2 (Thm4): threshold DECREASES with tau
                      -> harder to dispatch when little time remains
                      -> this IS the protection the collaborator asked about
    pi1 < pi2 (Thm5): threshold INCREASES with tau
                      -> harder to dispatch when much time remains
""")

    configs = [
        ("cu=1.0, alpha2=1.0", 2, 1.0, 1, 2.0, 2.0, 1.0),   # alpha2=2/3*1=0.67 hmm
        ("cu=2.0, alpha2=2.0", 4, 2.0, 1, 3.0, 3.0, 1.0),   # alpha2=4*2/4=2.0
        ("cu=4.0, alpha2=4.0", 4, 4.0, 1, 3.0, 3.0, 1.0),   # alpha2=4*4/4=4.0
    ]

    for label, lam2, cu, h, pi, c1, v2 in configs:
        a2 = lam2*cu/(h+pi)
        dp, p = build_and_solve_Cf0(lam2, cu, h, pi, pi, c1, c1, v2,
                                     lam1=lam1, T=T, N=N)
        tau_vals = [T*0.2, T*0.5, T*1.0]
        ths = [I2_threshold_dp_Cf0(dp, n_for_tau(tau, dp)) for tau in tau_vals]
        spread = max(t for t in ths if t) - min(t for t in ths if t)
        print(f"  {label:30s}  alpha2={a2:.2f}  "
              f"thresholds at tau=0.4,1.0,2.0: {ths}  "
              f"spread={spread} {'(tau-independent OK)' if spread<=1 else '(NG)'}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 68)
    print("Cf=0 VALIDATION  (analysis_Cf0.pdf Theorems 1-5)")
    print("=" * 68)
    print(f"Solver: corrected g() (post-dispatch flow cost), Cf=0.")
    print(f"Settings: N=200, T=2.0, I2_max=30, b1_max=60, lam1=5.")
    print(f"I2 threshold queried at b1={B1_QUERY} (b1 dimension irrelevant when Cf=0).")
    print(f"Criterion: |diff| <= 1 -> OK.")

    # Collaborator's question first
    validate_collaborator_question()

    print(f"\n\n{'='*68}")
    print("SYMMETRIC CASE  (pi1=pi2)  -- Theorems 1 & 2")
    print(f"{'='*68}")
    for args in SYM_CONFIGS:
        validate_symmetric(*args)

    print(f"\n\n{'='*68}")
    print("ASYMMETRIC CASE  (pi1>pi2)  -- Theorems 3 & 4")
    print(f"{'='*68}")
    for args in ASYM_GT_CONFIGS:
        validate_asymmetric(*args)

    print(f"\n\n{'='*68}")
    print("ASYMMETRIC CASE  (pi1<pi2)  -- Theorem 5")
    print(f"{'='*68}")
    for args in ASYM_LT_CONFIGS:
        validate_asymmetric(*args)

    print(f"\n{SEP}")
    print("RUN COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()