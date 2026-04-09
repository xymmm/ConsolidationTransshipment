"""
validate_thresholds.py
======================
Validation of analytical threshold formulas (Theorems 3-5) against the
backward-induction DP solver.  Covers Cases A1, A2, B1a and explicitly
surfaces the three paper errors identified in the analysis.

Assumes solver.py (corrected version) is in the same directory.  Run as:
    python validate_thresholds.py

Output is structured as one section per case, each section printing:
  (i)   parameter summary and dimensionless group values
  (ii)  q* structure check  (policy quantity)
  (iii) b1*(I2) threshold table -- DP vs analytical
  (iv)  I2bar threshold -- DP vs correct vs paper formula(s)
  (v)   summary error statistics

Notation
--------
  alpha   = lam2 * cu / (h + pi)         [symmetric case]
  alpha2  = lam2 * cu / (h + pi2)        [asymmetric case]
  Phi     = 2 * lam2 * Cf / (h + pi)
  Phi2    = 2 * lam2 * Cf / (h + pi2)
  gamma   = lam2 * (pi1 - pi2) / (h + pi2)
  beta    = round(alpha - 0.5)            [A2 hold-back quantity]

Time mapping
------------
  Period n remaining  <->  tau = n * dt  time remaining
  n = N  <->  tau = T   (full horizon)
  n = k  <->  tau = k * T / N

Threshold extraction from DP
-----------------------------
  b1_DP(I2, n)   = min b1 in {1,...,I2} s.t. dp dispatches at (I2, b1, n)
  I2bar_DP(n)    = min I2 in {1,...,I2_max} s.t. dp dispatches at (I2, B1_LARGE, n)
                   B1_LARGE is fixed (=45), not b1_max, to avoid clipping distortion.

Error metric
------------
  diff = t_DP - ceil(t_analytical)
  |diff| <= 1  ->  OK  (integrality gap only)
  |diff| >  1  ->  NG  (formula error)

Clipping note
-------------
  b1_max=80, B1_LARGE=45, lam1*T=10.
  Buffer = 80 - 45 - 10 = 25 > 0, so future transitions from B1_LARGE
  never hit the boundary, eliminating clipping distortion in I2bar_DP.

Solver fix note
---------------
  The corrected solver.py computes flow cost on the POST-dispatch state
  (I2-q, b1-q).  The original version used the pre-dispatch state (I2, b1),
  overcharging by dt*(h+pi1)*q per unit dispatched, which artificially
  raised all thresholds by ~2-3 units.
"""

import math
import numpy as np
from solver import Params, TransshipmentDP


# ======================================================================
# SECTION 0:  PARAMETER SETS
# ======================================================================
#
# Design rationale:
#   A1: alpha in {0.18, 0.33, 0.45}  --  low / mid / near-boundary
#   A2: alpha in {0.75, 1.50, 2.00}  --  beta in {0, 1, 2}
#       beta=0: Error 2 vanishes (baseline); beta=1,2: expose Error 2
#   B1a: alpha2 in {0.30, 0.25}, gamma in {2.0, 2.5}
#       tau* = (2*alpha2-1)/(2*gamma) < 0 in both configs (B1a region OK)
#
# Terminal costs: c1=c2=pi (flow penalty), v2=1.0 < c2.
#   Non-zero c1,c2 activate the correct Vw boundary condition.
#   v2>0 gives DP the salvage incentive for partial dispatch in A2.
#   For B1a: paper assumes c1=c2=c, so c1=c2=pi2.
#   Threshold formulas are independent of c1=c2 by Eq.(19) -- only c1-c2
#   appears in Vd-Vw=0, which is zero for symmetric c1=c2.
#
# State space: I2_max=35, I2_min=-10, b1_max=80.
# Solver: N=200, T=2.0, dt=0.01.  Stability: p0=1-(lam1+lam2)*dt.
#   A1/A2 worst case: lam1+lam2=5+5=10, p0=1-0.1=0.9 OK.
#   B1a worst case:   lam1+lam2=5+3=8,  p0=1-0.08=0.92 OK.
# lam1=5 throughout (does not affect threshold formulas).

A1_CONFIGS = [
    # (label,          lam2, cu,   h,   pi,   Cf,   c1,   c2,   v2)
    ("alpha=0.18",       2, 0.5,  1,  4.5,   5,  4.5,  4.5,  1.0),
    ("alpha=0.33",       3, 1.0,  1,  8.0,  10,  8.0,  8.0,  1.0),
    ("alpha=0.45",       5, 0.9,  1,  9.0,  15,  9.0,  9.0,  1.0),
]

A2_CONFIGS = [
    # (label,          lam2, cu,   h,   pi,   Cf,   c1,   c2,   v2)
    ("alpha=0.75 beta=0",  3, 1.0,  1,  3.0,   8,  3.0,  3.0,  1.0),
    ("alpha=1.50 beta=1",  3, 2.0,  1,  3.0,  10,  3.0,  3.0,  1.0),
    ("alpha=2.00 beta=2",  4, 2.0,  1,  3.0,  12,  3.0,  3.0,  1.0),
]

B1A_CONFIGS = [
    # (label,             lam2, cu,   h, pi1, pi2,  Cf,   c1,   c2,   v2)
    ("alpha2=0.30 gamma=2.0",  2, 0.75, 1,  9,   4,   8,  4.0,  4.0,  1.0),
    ("alpha2=0.25 gamma=2.5",  3, 0.50, 1, 10,   5,  10,  5.0,  5.0,  1.0),
]

# Fixed b1 used when querying I2bar (must be > I2_max=35 for Case-2 regime,
# and b1_max - B1_LARGE - lam1*T = 80 - 45 - 10 = 25 > 0 for no clipping).
B1_LARGE = 45


# ======================================================================
# SECTION 1:  ANALYTICAL FORMULAS
# ======================================================================

def _sqrt(x):
    return math.sqrt(max(0.0, x))


# -- A1 (Theorem 3) ----------------------------------------------------

def A1_alpha(lam2, cu, h, pi):
    return lam2 * cu / (h + pi)

def A1_Phi(lam2, Cf, h, pi):
    return 2 * lam2 * Cf / (h + pi)

def A1_b1star(I2, lam2, cu, h, pi, Cf):
    """Case 1 threshold b1*(I2) -- paper-correct. None if Delta<0."""
    a   = A1_alpha(lam2, cu, h, pi)
    phi = A1_Phi(lam2, Cf, h, pi)
    A   = 2 * I2 + 1 - 2 * a
    D   = A**2 - 4 * phi
    if D < 0:
        return None
    return 0.5 * (A - _sqrt(D))

def A1_I2bar(lam2, cu, h, pi, Cf):
    """Case 2 threshold I2bar -- paper-correct."""
    a   = A1_alpha(lam2, cu, h, pi)
    phi = A1_Phi(lam2, Cf, h, pi)
    xi  = 1 - 2 * a
    return 0.5 * (_sqrt(xi**2 + 4 * phi) - xi)


# -- A2 (Theorem 4) ----------------------------------------------------

def A2_beta(lam2, cu, h, pi):
    return round(A1_alpha(lam2, cu, h, pi) - 0.5)   # banker's rounding

def A2_b1star(I2, lam2, cu, h, pi, Cf):
    """Case 1 threshold -- same formula as A1, paper-correct."""
    return A1_b1star(I2, lam2, cu, h, pi, Cf)

def A2_I2bar_correct(lam2, cu, h, pi, Cf):
    """Case 2 threshold -- CORRECTED (Error 2 fixed).
    kappa = Phi - beta*(2*(alpha-0.5) - beta)
    """
    a    = A1_alpha(lam2, cu, h, pi)
    phi  = A1_Phi(lam2, Cf, h, pi)
    beta = A2_beta(lam2, cu, h, pi)
    kap  = phi - beta * (2 * (a - 0.5) - beta)
    Abar = 2 * a - 1
    return 0.5 * (Abar + _sqrt(Abar**2 + 4 * kap))

def A2_I2bar_paper(lam2, cu, h, pi, Cf):
    """Case 2 threshold -- PAPER formula, contains Error 2 (extra -beta).
    kappa_paper = Phi - beta*(2*(alpha-0.5)-beta) - beta
    """
    a    = A1_alpha(lam2, cu, h, pi)
    phi  = A1_Phi(lam2, Cf, h, pi)
    beta = A2_beta(lam2, cu, h, pi)
    kap  = phi - beta * (2 * (a - 0.5) - beta) - beta
    Abar = 2 * a - 1
    return 0.5 * (Abar + _sqrt(Abar**2 + 4 * kap))


# -- B1a (Theorem 5) ---------------------------------------------------

def B1a_alpha2(lam2, cu, h, pi2):
    return lam2 * cu / (h + pi2)

def B1a_Phi2(lam2, Cf, h, pi2):
    return 2 * lam2 * Cf / (h + pi2)

def B1a_gamma(lam2, pi1, pi2, h):
    return lam2 * (pi1 - pi2) / (h + pi2)

def B1a_b1star(I2, tau, lam2, cu, h, pi1, pi2, Cf):
    """Case 1 threshold b1*(I2,tau) -- paper-correct (no error in Case 1).
    A(tau) = 2*I2 + 1 - 2*alpha2 + 2*gamma*tau
    """
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    A    = 2 * I2 + 1 - 2 * a2 + 2 * gam * tau
    D    = A**2 - 4 * phi2
    if D < 0:
        return None
    return 0.5 * (A - _sqrt(D))

def B1a_I2bar_correct(tau, lam2, cu, h, pi1, pi2, Cf):
    """Case 2 threshold I2bar(tau) -- CORRECTED.
    xi(tau) = 1 - 2*alpha2 + 2*gamma*tau   (fixed from paper's eta)
    Phi2    = 2*lam2*Cf / (h+pi2)          (fixed from paper's h+pi1)
    """
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    xi   = 1 - 2 * a2 + 2 * gam * tau
    return 0.5 * (_sqrt(xi**2 + 4 * phi2) - xi)

def B1a_I2bar_err1_only(tau, lam2, cu, h, pi1, pi2, Cf):
    """Case 2: Error 1 only -- wrong xi (eta), correct Phi2.
    Paper writes eta = 1 + 2*alpha2 - 2*gamma*tau (sign of alpha2 and gamma*tau swapped).
    """
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    eta  = 1 + 2 * a2 - 2 * gam * tau
    return 0.5 * (_sqrt(eta**2 + 4 * phi2) - eta)

def B1a_I2bar_err3_only(tau, lam2, cu, h, pi1, pi2, Cf):
    """Case 2: Error 3 only -- correct xi, wrong Phi denominator.
    Theorem 5 writes 8*lam2*Cf/(h+pi) using pi1 instead of pi2.
    """
    a2      = B1a_alpha2(lam2, cu, h, pi2)
    gam     = B1a_gamma(lam2, pi1, pi2, h)
    phi_err = 2 * lam2 * Cf / (h + pi1)
    xi      = 1 - 2 * a2 + 2 * gam * tau
    return 0.5 * (_sqrt(xi**2 + 4 * phi_err) - xi)

def B1a_I2bar_both_errors(tau, lam2, cu, h, pi1, pi2, Cf):
    """Case 2: both Error 1 and Error 3 as the paper actually states."""
    a2      = B1a_alpha2(lam2, cu, h, pi2)
    gam     = B1a_gamma(lam2, pi1, pi2, h)
    phi_err = 2 * lam2 * Cf / (h + pi1)
    eta     = 1 + 2 * a2 - 2 * gam * tau
    return 0.5 * (_sqrt(eta**2 + 4 * phi_err) - eta)


# ======================================================================
# SECTION 2:  DP HELPERS
# ======================================================================

def build_and_solve(lam2, cu, h, pi1, pi2, Cf, lam1=5.0,
                    T=2.0, N=200, I2_max=35, I2_min=-10, b1_max=80,
                    c1=0.0, c2=0.0, v2=0.0):
    """Build Params, run DP, return (dp, params)."""
    p = Params(
        T=T, N=N, lam1=lam1, lam2=lam2,
        h=h, Cf=Cf, cu=cu,
        pi1=pi1, pi2=pi2,
        c1=c1, c2=c2, v2=v2,
        I2_max=I2_max, I2_min=I2_min, b1_max=b1_max,
    )
    dp = TransshipmentDP(p)
    dp.solve(store_V=False, verbose=False)
    return dp, p


def b1_threshold_dp(dp, n, I2):
    """Smallest b1 in {1,...,I2} s.t. dp dispatches at (I2, b1, n periods left).
    Returns None if no dispatch found (always wait for this I2).
    """
    for b1 in range(1, min(I2, dp.p.b1_max) + 1):
        if dp.get_policy(n, I2, b1) > 0:
            return b1
    return None


def I2_threshold_dp(dp, n, b1_large=B1_LARGE):
    """Smallest I2 in {1,...,I2_max} s.t. dp dispatches when b1=b1_large.
    Uses fixed b1_large (not b1_max) to avoid boundary clipping distortion.
    Returns None if no dispatch found.
    """
    for I2 in range(1, dp.p.I2_max + 1):
        if dp.get_policy(n, I2, b1_large) > 0:
            return I2
    return None


def periods_for_tau(tau_fraction, N):
    """Convert tau as fraction of horizon to number of periods remaining."""
    return max(1, round(tau_fraction * N))


# ======================================================================
# SECTION 3:  VALIDATION RUNNERS
# ======================================================================

SEP = "-" * 72

def _diff_str(dp_val, analytic_val):
    """diff = dp_val - ceil(analytic_val).  OK if |diff|<=1, NG otherwise."""
    if dp_val is None or analytic_val is None:
        return None, "--"
    d    = dp_val - math.ceil(analytic_val)
    flag = "OK" if abs(d) <= 1 else "NG"
    return d, flag


# -- Case A1 -----------------------------------------------------------

def validate_A1(label, lam2, cu, h, pi, Cf, c1, c2, v2,
                lam1=5.0, N=200, T=2.0):
    a   = A1_alpha(lam2, cu, h, pi)
    phi = A1_Phi(lam2, Cf, h, pi)

    print(f"\n{SEP}")
    print(f"CASE A1 | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi={pi}  Cf={Cf}"
          f"  c1=c2={c1}  v2={v2}  lam1={lam1}")
    print(f"  alpha={a:.4f}  Phi={phi:.4f}"
          f"  {'(alpha<0.5 OK)' if a < 0.5 else '(alpha>=0.5 WRONG CASE)'}")

    dp, p = build_and_solve(lam2, cu, h, pi, pi, Cf, lam1, T, N,
                            c1=c1, c2=c2, v2=v2)
    n_full = N

    # T1: q* structure check
    viol = []
    for I2 in range(1, p.I2_max + 1):
        for b1 in range(1, p.b1_max + 1):
            q = dp.get_policy(n_full, I2, b1)
            if q > 0 and q != min(I2, b1):
                viol.append((I2, b1, q))
    print(f"\n  [T1] q* = min(I2,b1) whenever dispatching:")
    print(f"       {len(viol)} violation(s)  {'(expect 0 OK)' if not viol else 'NG'}")
    for v in viol[:5]:
        print(f"       I2={v[0]} b1={v[1]} q*={v[2]} expected={min(v[0],v[1])}")

    # T2: b1* -- Case 1 threshold
    print(f"\n  [T2] b1*(I2) threshold -- Case 1 (b1<=I2), tau=T:")
    print(f"       {'I2':>4} | {'DP':>5} | {'Analytical':>11}"
          f" | {'ceil':>5} | {'diff':>5} | flag")
    diffs = []
    for I2 in range(2, min(20, p.I2_max) + 1):
        b1_dp = b1_threshold_dp(dp, n_full, I2)
        b1_an = A1_b1star(I2, lam2, cu, h, pi, Cf)
        diff, flag = _diff_str(b1_dp, b1_an)
        if b1_an is None:
            print(f"       {I2:>4} | {'--':>5} | {'Delta<0':>11}"
                  f" | {'--':>5} | {'--':>5} | --")
            continue
        print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>11.4f}"
              f" | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
        if diff is not None:
            diffs.append(abs(diff))
    if diffs:
        print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")

    # T3: I2bar -- Case 2 threshold
    I2_dp  = I2_threshold_dp(dp, n_full)
    I2_bar = A1_I2bar(lam2, cu, h, pi, Cf)
    diff, flag = _diff_str(I2_dp, I2_bar)
    print(f"\n  [T3] I2bar -- Case 2 (b1>=I2+1), tau=T, query b1={B1_LARGE}:")
    print(f"       DP={I2_dp}  Analytical={I2_bar:.4f}"
          f"  ceil={math.ceil(I2_bar)}  diff={diff}  {flag}")


# -- Case A2 -----------------------------------------------------------

def validate_A2(label, lam2, cu, h, pi, Cf, c1, c2, v2,
                lam1=5.0, N=200, T=2.0):
    a     = A1_alpha(lam2, cu, h, pi)
    phi   = A1_Phi(lam2, Cf, h, pi)
    beta  = A2_beta(lam2, cu, h, pi)
    I2c   = A2_I2bar_correct(lam2, cu, h, pi, Cf)
    I2p   = A2_I2bar_paper(lam2, cu, h, pi, Cf)
    kap_c = phi - beta * (2 * (a - 0.5) - beta)
    kap_p = kap_c - beta

    print(f"\n{SEP}")
    print(f"CASE A2 | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi={pi}  Cf={Cf}"
          f"  c1=c2={c1}  v2={v2}")
    print(f"  alpha={a:.4f}  Phi={phi:.4f}  beta={beta}"
          f"  {'(alpha>0.5 OK)' if a > 0.5 else '(alpha<=0.5 WRONG)'}")
    print(f"  kappa_correct={kap_c:.4f}  kappa_paper={kap_p:.4f}"
          f"  Delta_kappa=beta={beta}")
    print(f"  I2bar_correct={I2c:.4f}  I2bar_paper={I2p:.4f}"
          f"  Delta_I2bar={I2c - I2p:.4f}")

    dp, p = build_and_solve(lam2, cu, h, pi, pi, Cf, lam1, T, N,
                            c1=c1, c2=c2, v2=v2)
    n_full   = N
    I2_floor = int(math.floor(a))

    # T1: pre-check I2 <= alpha -> always wait
    print(f"\n  [T1] Pre-check: I2 <= alpha={a:.2f}  ->  always wait")
    for I2 in range(1, I2_floor + 1):
        nd   = sum(dp.get_policy(n_full, I2, b1) > 0
                   for b1 in range(1, p.b1_max + 1))
        flag = "OK" if nd == 0 else f"NG ({nd} dispatches)"
        print(f"       I2={I2}: {flag}")

    # T2: q* quantity -- partial dispatch check
    print(f"\n  [T2] q* quantity: b1>=I2-beta+1 regime, expect q*=I2-beta")
    q_viol = []
    for I2 in range(I2_floor + 2, min(20, p.I2_max) + 1):
        for b1 in range(max(1, I2 - beta + 1), min(I2 + 15, p.b1_max) + 1):
            q = dp.get_policy(n_full, I2, b1)
            if q > 0 and q != I2 - beta:
                q_viol.append((I2, b1, q, I2 - beta))
    print(f"       {len(q_viol)} violation(s)"
          f"  {'(expect 0 OK)' if not q_viol else 'NG'}")
    for v in q_viol[:5]:
        print(f"       I2={v[0]} b1={v[1]} q*={v[2]} expected={v[3]}")

    # T3: I2bar -- Error 2 comparison
    I2_dp = I2_threshold_dp(dp, n_full)
    dc, fc  = _diff_str(I2_dp, I2c)
    dp2, fp = _diff_str(I2_dp, I2p)
    print(f"\n  [T3] I2bar -- Case 2, tau=T, query b1={B1_LARGE}  (*** Error 2 ***)")
    print(f"       DP result         = {I2_dp}")
    print(f"       Corrected formula = {I2c:.4f}"
          f"  -> ceil={math.ceil(I2c)}  diff={dc}  {fc}")
    print(f"       Paper formula     = {I2p:.4f}"
          f"  -> ceil={math.ceil(I2p)}  diff={dp2}  {fp}")
    print(f"       Error 2: kappa_paper = kappa_correct - beta"
          f" = {kap_c:.4f} - {beta} = {kap_p:.4f}")
    print(f"       I2bar_paper underestimates by {I2c - I2p:.4f} units")

    # T4: b1* -- Case 1 (no error, baseline check)
    print(f"\n  [T4] b1*(I2) -- Case 1 (b1<=I2-beta), no error expected:")
    print(f"       {'I2':>4} | {'DP':>5} | {'Analytical':>11}"
          f" | {'ceil':>5} | {'diff':>5} | flag")
    diffs = []
    for I2 in range(I2_floor + 2, min(18, p.I2_max) + 1):
        b1_dp = b1_threshold_dp(dp, n_full, I2)
        b1_an = A2_b1star(I2, lam2, cu, h, pi, Cf)
        diff, flag = _diff_str(b1_dp, b1_an)
        if b1_an is None:
            print(f"       {I2:>4} | {'--':>5} | {'Delta<0':>11}"
                  f" | {'--':>5} | {'--':>5} | --")
            continue
        print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>11.4f}"
              f" | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
        if diff is not None:
            diffs.append(abs(diff))
    if diffs:
        print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")


# -- Case B1a ----------------------------------------------------------

def validate_B1a(label, lam2, cu, h, pi1, pi2, Cf, c1, c2, v2,
                 lam1=5.0, N=200, T=2.0):
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    tau_star = (2 * a2 - 1) / (2 * gam) if gam != 0 else float('inf')

    print(f"\n{SEP}")
    print(f"CASE B1a | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi1={pi1}  pi2={pi2}"
          f"  Cf={Cf}  c1=c2={c1}  v2={v2}")
    print(f"  alpha2={a2:.4f}  Phi2={phi2:.4f}  gamma={gam:.4f}")
    print(f"  {'(alpha2<0.5 OK, gamma>0 OK)' if a2 < 0.5 and gam > 0 else 'WARNING: wrong region'}")
    print(f"  tau* (hypothetical) = {tau_star:.4f}"
          f"  {'(tau*<0 -> no crossing in [0,T] OK)' if tau_star < 0 else '(tau*>0 -> verify B1a region)'}")

    dp, p = build_and_solve(lam2, cu, h, pi1, pi2, Cf, lam1, T, N,
                            c1=c1, c2=c2, v2=v2)
    n_full = N

    # T1: q* structure -- full dispatch
    viol = []
    for I2 in range(1, p.I2_max + 1):
        for b1 in range(1, p.b1_max + 1):
            q = dp.get_policy(n_full, I2, b1)
            if q > 0 and q != min(I2, b1):
                viol.append((I2, b1, q))
    print(f"\n  [T1] q* = min(I2,b1) whenever dispatching (full dispatch):")
    print(f"       {len(viol)} violation(s)"
          f"  {'(expect 0 OK)' if not viol else 'NG'}")
    for v in viol[:5]:
        print(f"       I2={v[0]} b1={v[1]} q*={v[2]} expected={min(v[0],v[1])}")

    # T2: b1*(I2, tau) -- Case 1, across tau
    tau_fracs = [0.25, 0.50, 0.75, 1.00]
    print(f"\n  [T2] b1*(I2, tau) -- Case 1, across tau and I2:")
    for tf in tau_fracs:
        tau = tf * T
        n   = periods_for_tau(tf, N)
        print(f"\n       tau={tau:.2f}  (n={n})")
        print(f"       {'I2':>4} | {'DP':>5} | {'Analytical':>11}"
              f" | {'ceil':>5} | {'diff':>5} | flag")
        diffs = []
        for I2 in range(2, min(18, p.I2_max) + 1):
            b1_dp = b1_threshold_dp(dp, n, I2)
            b1_an = B1a_b1star(I2, tau, lam2, cu, h, pi1, pi2, Cf)
            diff, flag = _diff_str(b1_dp, b1_an)
            if b1_an is None:
                print(f"       {I2:>4} | {'--':>5} | {'Delta<0':>11}"
                      f" | {'--':>5} | {'--':>5} | --")
                continue
            print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>11.4f}"
                  f" | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
            if diff is not None:
                diffs.append(abs(diff))
        if diffs:
            print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")

    # T3: I2bar(tau) -- Error 1 & 3 comparison
    print(f"\n  [T3] I2bar(tau) -- Case 2, query b1={B1_LARGE}"
          f"  (*** Errors 1 & 3 ***)")
    hdr = (f"  {'tau':>5} | {'DP':>4} | {'Correct':>8} | {'Err1':>8}"
           f" | {'Err3':>8} | {'Both':>8} | dC | d1 | d3 | dB")
    print(f"\n  {hdr}")
    print(f"  {'-' * 90}")
    for tf in [0.10, 0.25, 0.50, 0.75, 1.00]:
        tau   = tf * T
        n     = periods_for_tau(tf, N)
        I2_dp = I2_threshold_dp(dp, n)
        I2_c  = B1a_I2bar_correct(tau, lam2, cu, h, pi1, pi2, Cf)
        I2_e1 = B1a_I2bar_err1_only(tau, lam2, cu, h, pi1, pi2, Cf)
        I2_e3 = B1a_I2bar_err3_only(tau, lam2, cu, h, pi1, pi2, Cf)
        I2_b  = B1a_I2bar_both_errors(tau, lam2, cu, h, pi1, pi2, Cf)
        dc, _ = _diff_str(I2_dp, I2_c)
        d1, _ = _diff_str(I2_dp, I2_e1)
        d3, _ = _diff_str(I2_dp, I2_e3)
        db, _ = _diff_str(I2_dp, I2_b)
        print(f"  {tau:>5.2f} | {str(I2_dp):>4} | {I2_c:>8.3f} | {I2_e1:>8.3f}"
              f" | {I2_e3:>8.3f} | {I2_b:>8.3f}"
              f" | {str(dc):>2} | {str(d1):>2} | {str(d3):>2} | {str(db):>2}")

    # T4: xi vs eta trajectory -- Error 1 magnitude
    print(f"\n  [T4] xi vs eta trajectory (Error 1 magnitude over tau):")
    print(f"       {'tau':>5} | {'xi_correct':>11} | {'eta_paper':>10}"
          f" | {'xi-eta':>8}")
    print(f"       {'-' * 46}")
    for tf in [0.10, 0.25, 0.50, 0.75, 1.00]:
        tau = tf * T
        xi  = 1 - 2 * a2 + 2 * gam * tau
        eta = 1 + 2 * a2 - 2 * gam * tau
        print(f"       {tau:>5.2f} | {xi:>11.4f} | {eta:>10.4f}"
              f" | {xi - eta:>8.4f}")
    print(f"       (xi-eta) = 4*(gamma*tau - alpha2)"
          f"  -> sign change at tau=alpha2/gamma={a2/gam:.4f}")
    phi_err3 = 2 * lam2 * Cf / (h + pi1)
    print(f"       Error 3: Phi2={phi2:.4f}  Phi_err3={phi_err3:.4f}"
          f"  DeltaPhi={phi2 - phi_err3:.4f}")


# ======================================================================
# SECTION 4:  DIAGRAM VALIDITY CHECK
# ======================================================================

def check_diagram():
    print(f"\n{SEP}")
    print("DIAGRAM VALIDITY CHECK")
    print(SEP)
    print("""
The interactive diagram classifies (alpha2, gamma*T0) into 8 regions:
  (1) Vertical line:  alpha2 = 1/2
  (2) Diagonal line:  gamma*T0 = alpha2 - 1/2

Boundaries come from q^axis(tau) >= I2, i.e. gamma*tau >= alpha2 - 1/2.
Errors 1, 2, 3 affect threshold FORMULAS (Vd=Vw), not the boundaries.
The 8-region partition is correct as drawn.

Formulas to correct in click-through panels:
  B1a Case 2: use xi(tau) = 1 - 2*alpha2 + 2*gamma*tau  (not paper's eta)
              use Phi2 = 2*lam2*Cf/(h+pi2)              (not h+pi1)
  A2  Case 2: use kappa = Phi - beta*(2*(alpha-0.5)-beta)  (drop extra -beta)
""")


# ======================================================================
# SECTION 5:  MAIN
# ======================================================================

def main():
    print("=" * 72)
    print("THRESHOLD VALIDATION -- Theorems 3-5 vs DP Solver")
    print("=" * 72)
    print("Solver fix: g() uses POST-dispatch flow cost (I2-q, b1-q).")
    print(f"Settings:   N=200, T=2.0, b1_max=80, I2_max=35, I2_min=-10.")
    print(f"I2bar query: fixed b1={B1_LARGE} (not b1_max; avoids clipping).")
    print("Terminal:   c1=c2=pi, v2=1.0 per config (v2 < c2 OK).")
    print("Criterion:  |diff| <= 1 -> OK (integrality only).")

    print("\n\n" + "=" * 72)
    print("CASE A1  (Theorem 3)   pi1=pi2, alpha <= 0.5")
    print("=" * 72)
    for args in A1_CONFIGS:
        validate_A1(*args)

    print("\n\n" + "=" * 72)
    print("CASE A2  (Theorem 4)   pi1=pi2, alpha > 0.5   [Error 2]")
    print("=" * 72)
    for args in A2_CONFIGS:
        validate_A2(*args)

    print("\n\n" + "=" * 72)
    print("CASE B1a (Theorem 5)   pi1>pi2, alpha2<=0.5   [Errors 1 & 3]")
    print("=" * 72)
    for args in B1A_CONFIGS:
        validate_B1a(*args)

    check_diagram()

    print(f"\n{SEP}")
    print("RUN COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()