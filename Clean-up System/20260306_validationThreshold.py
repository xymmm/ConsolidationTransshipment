"""
validate_thresholds.py
======================
Validation of analytical threshold formulas (Theorems 3-5) against the
backward-induction DP solver.  Covers Cases A1, A2, B1a.

One confirmed error in the manuscript:
  Error 1 (B1a, Case 2): paper writes
      eta = 1 - 2*lambda2*((pi1-pi2)*tau - cu) / (h+pi2)
    which expands to  1 + 2*alpha2 - 2*gamma*tau.
    The correct coefficient should be  1 - 2*alpha2 + 2*gamma*tau.

Assumes solver.py is in the same directory.  Run as:
    python validate_thresholds.py
"""

import math
import numpy as np
from solver import Params, TransshipmentDP


# ======================================================================
# PARAMETER SETS
# ======================================================================

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

B1_LARGE = 45


# ======================================================================
# ANALYTICAL FORMULAS
# ======================================================================

def _sqrt(x):
    return math.sqrt(max(0.0, x))


# -- A1 (Theorem 3) ----------------------------------------------------

def A1_alpha(lam2, cu, h, pi):
    return lam2 * cu / (h + pi)

def A1_Phi(lam2, Cf, h, pi):
    return 2 * lam2 * Cf / (h + pi)

def A1_b1star(I2, lam2, cu, h, pi, Cf):
    """Case 1 threshold b1*(I2). Returns None if discriminant < 0."""
    a   = A1_alpha(lam2, cu, h, pi)
    phi = A1_Phi(lam2, Cf, h, pi)
    A   = 2 * I2 + 1 - 2 * a
    D   = A**2 - 4 * phi
    if D < 0:
        return None
    return 0.5 * (A - _sqrt(D))

def A1_I2bar(lam2, cu, h, pi, Cf):
    """Case 2 threshold I2bar (paper formula, correct)."""
    a   = A1_alpha(lam2, cu, h, pi)
    phi = A1_Phi(lam2, Cf, h, pi)
    xi  = 1 - 2 * a
    return 0.5 * (_sqrt(xi**2 + 4 * phi) - xi)


# -- A2 (Theorem 4) ----------------------------------------------------

def A2_beta(lam2, cu, h, pi):
    return round(A1_alpha(lam2, cu, h, pi) - 0.5)

def A2_b1star(I2, lam2, cu, h, pi, Cf):
    """Case 1 threshold (same formula as A1)."""
    return A1_b1star(I2, lam2, cu, h, pi, Cf)

def A2_I2bar(lam2, cu, h, pi, Cf):
    """Case 2 threshold (paper formula, Theorem 4).
    kappa = Phi - [lam2*cu/(h+pi) - 1/2] * (2*(lam2*cu/(h+pi) - 1/2)
                                              - [lam2*cu/(h+pi) - 1/2])
    """
    a    = A1_alpha(lam2, cu, h, pi)
    phi  = A1_Phi(lam2, Cf, h, pi)
    beta = A2_beta(lam2, cu, h, pi)
    kap  = phi - beta * (2 * (a - 0.5) - beta)
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
    """Case 1 threshold b1*(I2, tau) -- paper formula, no error in Case 1."""
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    A    = 2 * I2 + 1 - 2 * a2 + 2 * gam * tau
    D    = A**2 - 4 * phi2
    if D < 0:
        return None
    return 0.5 * (A - _sqrt(D))

def B1a_I2bar_corrected(tau, lam2, cu, h, pi1, pi2, Cf):
    """Case 2 threshold -- Error 1 corrected.
    Uses correct coefficient: 1 - 2*alpha2 + 2*gamma*tau
    (paper writes 1 + 2*alpha2 - 2*gamma*tau, signs of alpha2 and gamma*tau swapped).
    """
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    xi   = 1 - 2 * a2 + 2 * gam * tau          # corrected
    return 0.5 * (_sqrt(xi**2 + 4 * phi2) - xi)

def B1a_I2bar_paper(tau, lam2, cu, h, pi1, pi2, Cf):
    """Case 2 threshold -- paper formula as written in Theorem 5.
    Uses paper's coefficient: 1 - 2*lambda2*((pi1-pi2)*tau - cu)/(h+pi2)
    which expands to 1 + 2*alpha2 - 2*gamma*tau  (Error 1).
    """
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    eta  = 1 + 2 * a2 - 2 * gam * tau          # paper's version
    return 0.5 * (_sqrt(eta**2 + 4 * phi2) - eta)


# ======================================================================
# DP HELPERS
# ======================================================================

def build_and_solve(lam2, cu, h, pi1, pi2, Cf, lam1=5.0,
                    T=2.0, N=200, I2_max=35, I2_min=-10, b1_max=80,
                    c1=0.0, c2=0.0, v2=0.0):
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
    for b1 in range(1, min(I2, dp.p.b1_max) + 1):
        if dp.get_policy(n, I2, b1) > 0:
            return b1
    return None


def I2_threshold_dp(dp, n, b1_large=B1_LARGE):
    for I2 in range(1, dp.p.I2_max + 1):
        if dp.get_policy(n, I2, b1_large) > 0:
            return I2
    return None


def periods_for_tau(tau_fraction, N):
    return max(1, round(tau_fraction * N))


# ======================================================================
# VALIDATION RUNNERS
# ======================================================================

SEP = "-" * 72

def _diff_str(dp_val, analytic_val):
    if dp_val is None or analytic_val is None:
        return None, "--"
    d    = dp_val - math.ceil(analytic_val)
    flag = "OK" if abs(d) <= 1 else "NG"
    return d, flag


# -- A1 ----------------------------------------------------------------

def validate_A1(label, lam2, cu, h, pi, Cf, c1, c2, v2, lam1=5.0, N=200, T=2.0):
    a   = A1_alpha(lam2, cu, h, pi)
    phi = A1_Phi(lam2, Cf, h, pi)

    print(f"\n{SEP}")
    print(f"CASE A1 | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi={pi}  Cf={Cf}  c1=c2={c1}  v2={v2}")
    print(f"  alpha={a:.4f}  Phi={phi:.4f}")

    dp, p = build_and_solve(lam2, cu, h, pi, pi, Cf, lam1, T, N, c1=c1, c2=c2, v2=v2)
    n_full = N

    # T1: q* structure
    viol = [(I2, b1, dp.get_policy(n_full, I2, b1))
            for I2 in range(1, p.I2_max+1)
            for b1 in range(1, p.b1_max+1)
            if dp.get_policy(n_full, I2, b1) > 0
            and dp.get_policy(n_full, I2, b1) != min(I2, b1)]
    print(f"\n  [T1] q* = min(I2,b1): {len(viol)} violation(s)"
          f"  {'OK' if not viol else 'NG'}")

    # T2: b1* threshold
    print(f"\n  [T2] b1*(I2) -- Case 1, tau=T:")
    print(f"       {'I2':>4} | {'DP':>5} | {'Analytical':>11} | {'ceil':>5} | {'diff':>5} | flag")
    diffs = []
    for I2 in range(2, min(20, p.I2_max)+1):
        b1_dp = b1_threshold_dp(dp, n_full, I2)
        b1_an = A1_b1star(I2, lam2, cu, h, pi, Cf)
        diff, flag = _diff_str(b1_dp, b1_an)
        if b1_an is None:
            print(f"       {I2:>4} | {'--':>5} | {'Delta<0':>11} | {'--':>5} | {'--':>5} | --")
            continue
        print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>11.4f}"
              f" | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
        if diff is not None:
            diffs.append(abs(diff))
    if diffs:
        print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")

    # T3: I2bar
    I2_dp  = I2_threshold_dp(dp, n_full)
    I2_bar = A1_I2bar(lam2, cu, h, pi, Cf)
    diff, flag = _diff_str(I2_dp, I2_bar)
    print(f"\n  [T3] I2bar -- Case 2, tau=T, b1={B1_LARGE}:")
    print(f"       DP={I2_dp}  Analytical={I2_bar:.4f}"
          f"  ceil={math.ceil(I2_bar)}  diff={diff}  {flag}")


# -- A2 ----------------------------------------------------------------

def validate_A2(label, lam2, cu, h, pi, Cf, c1, c2, v2, lam1=5.0, N=200, T=2.0):
    a    = A1_alpha(lam2, cu, h, pi)
    phi  = A1_Phi(lam2, Cf, h, pi)
    beta = A2_beta(lam2, cu, h, pi)
    I2b  = A2_I2bar(lam2, cu, h, pi, Cf)
    kap  = phi - beta * (2 * (a - 0.5) - beta)

    print(f"\n{SEP}")
    print(f"CASE A2 | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi={pi}  Cf={Cf}  c1=c2={c1}  v2={v2}")
    print(f"  alpha={a:.4f}  Phi={phi:.4f}  beta={beta}  kappa={kap:.4f}")
    print(f"  I2bar={I2b:.4f}  ceil={math.ceil(I2b)}")

    dp, p = build_and_solve(lam2, cu, h, pi, pi, Cf, lam1, T, N, c1=c1, c2=c2, v2=v2)
    n_full   = N
    I2_floor = int(math.floor(a))

    # T1: always wait for I2 <= alpha
    print(f"\n  [T1] I2 <= alpha={a:.2f} -> always wait:")
    for I2 in range(1, I2_floor+1):
        nd   = sum(dp.get_policy(n_full, I2, b1) > 0 for b1 in range(1, p.b1_max+1))
        print(f"       I2={I2}: {'OK' if nd==0 else f'NG ({nd} dispatches)'}")

    # T2: q* = I2 - beta
    print(f"\n  [T2] q* = I2 - beta = I2 - {beta}:")
    q_viol = [(I2, b1, dp.get_policy(n_full, I2, b1), I2-beta)
              for I2 in range(I2_floor+2, min(20, p.I2_max)+1)
              for b1 in range(max(1, I2-beta+1), min(I2+15, p.b1_max)+1)
              if dp.get_policy(n_full, I2, b1) > 0
              and dp.get_policy(n_full, I2, b1) != I2-beta]
    print(f"       {len(q_viol)} violation(s)  {'OK' if not q_viol else 'NG'}")
    for v in q_viol[:5]:
        print(f"       I2={v[0]} b1={v[1]} q*={v[2]} expected={v[3]}")

    # T3: I2bar (Theorem 4 paper formula)
    I2_dp = I2_threshold_dp(dp, n_full)
    diff, flag = _diff_str(I2_dp, I2b)
    print(f"\n  [T3] I2bar -- Case 2, tau=T, b1={B1_LARGE}:")
    print(f"       DP={I2_dp}  Analytical={I2b:.4f}"
          f"  ceil={math.ceil(I2b)}  diff={diff}  {flag}")

    # T4: b1* baseline
    print(f"\n  [T4] b1*(I2) -- Case 1, tau=T:")
    print(f"       {'I2':>4} | {'DP':>5} | {'Analytical':>11} | {'ceil':>5} | {'diff':>5} | flag")
    diffs = []
    for I2 in range(I2_floor+2, min(18, p.I2_max)+1):
        b1_dp = b1_threshold_dp(dp, n_full, I2)
        b1_an = A2_b1star(I2, lam2, cu, h, pi, Cf)
        diff, flag = _diff_str(b1_dp, b1_an)
        if b1_an is None:
            print(f"       {I2:>4} | {'--':>5} | {'Delta<0':>11} | {'--':>5} | {'--':>5} | --")
            continue
        print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>11.4f}"
              f" | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
        if diff is not None:
            diffs.append(abs(diff))
    if diffs:
        print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")


# -- B1a ---------------------------------------------------------------

def validate_B1a(label, lam2, cu, h, pi1, pi2, Cf, c1, c2, v2,
                 lam1=5.0, N=200, T=2.0):
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    tau_star = (2*a2-1)/(2*gam) if gam != 0 else float('inf')

    print(f"\n{SEP}")
    print(f"CASE B1a | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi1={pi1}  pi2={pi2}  Cf={Cf}")
    print(f"  alpha2={a2:.4f}  Phi2={phi2:.4f}  gamma={gam:.4f}")
    print(f"  tau*={tau_star:.4f}  {'(tau*<0 -> B1a region confirmed)' if tau_star < 0 else '(check region)'}")

    dp, p = build_and_solve(lam2, cu, h, pi1, pi2, Cf, lam1, T, N,
                            c1=c1, c2=c2, v2=v2)
    n_full = N

    # T1: full dispatch
    viol = [(I2, b1, dp.get_policy(n_full, I2, b1))
            for I2 in range(1, p.I2_max+1)
            for b1 in range(1, p.b1_max+1)
            if dp.get_policy(n_full, I2, b1) > 0
            and dp.get_policy(n_full, I2, b1) != min(I2, b1)]
    print(f"\n  [T1] q* = min(I2,b1): {len(viol)} violation(s)"
          f"  {'OK' if not viol else 'NG'}")

    # T2: b1*(I2, tau)
    tau_fracs = [0.25, 0.50, 0.75, 1.00]
    print(f"\n  [T2] b1*(I2, tau) -- Case 1:")
    for tf in tau_fracs:
        tau = tf * T
        n   = periods_for_tau(tf, N)
        print(f"\n       tau={tau:.2f}:")
        print(f"       {'I2':>4} | {'DP':>5} | {'Analytical':>11} | {'ceil':>5} | {'diff':>5} | flag")
        diffs = []
        for I2 in range(2, min(18, p.I2_max)+1):
            b1_dp = b1_threshold_dp(dp, n, I2)
            b1_an = B1a_b1star(I2, tau, lam2, cu, h, pi1, pi2, Cf)
            diff, flag = _diff_str(b1_dp, b1_an)
            if b1_an is None:
                print(f"       {I2:>4} | {'--':>5} | {'Delta<0':>11} | {'--':>5} | {'--':>5} | --")
                continue
            print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>11.4f}"
                  f" | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
            if diff is not None:
                diffs.append(abs(diff))
        if diffs:
            print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")

    # T3: I2bar(tau) -- corrected vs paper (Error 1 only)
    print(f"\n  [T3] I2bar(tau) -- Case 2, b1={B1_LARGE}  [Error 1]")
    print(f"       Corrected: coefficient = 1 - 2*alpha2 + 2*gamma*tau")
    print(f"       Paper:     coefficient = 1 + 2*alpha2 - 2*gamma*tau  (signs swapped)")
    print(f"\n       {'tau':>5} | {'DP':>4} | {'Corrected':>10} | {'dC':>4}"
          f" | {'Paper':>10} | {'dP':>4} | {'fC':>4} | {'fP':>4}")
    print(f"       {'-'*72}")
    for tf in [0.10, 0.25, 0.50, 0.75, 1.00]:
        tau   = tf * T
        n     = periods_for_tau(tf, N)
        I2_dp = I2_threshold_dp(dp, n)
        I2_c  = B1a_I2bar_corrected(tau, lam2, cu, h, pi1, pi2, Cf)
        I2_p  = B1a_I2bar_paper(tau, lam2, cu, h, pi1, pi2, Cf)
        dc, fc = _diff_str(I2_dp, I2_c)
        dp_, fp = _diff_str(I2_dp, I2_p)
        print(f"       {tau:>5.2f} | {str(I2_dp):>4} | {I2_c:>10.3f} | {str(dc):>4}"
              f" | {I2_p:>10.3f} | {str(dp_):>4} | {fc:>4} | {fp:>4}")

    # coefficient trajectory
    print(f"\n  [T3 detail] Coefficient trajectory:")
    print(f"       {'tau':>5} | {'Corrected (1-2a2+2g*tau)':>25} | {'Paper (1+2a2-2g*tau)':>22}")
    print(f"       {'-'*58}")
    for tf in [0.10, 0.25, 0.50, 0.75, 1.00]:
        tau = tf * T
        xi  = 1 - 2*a2 + 2*gam*tau
        eta = 1 + 2*a2 - 2*gam*tau
        print(f"       {tau:>5.2f} | {xi:>25.4f} | {eta:>22.4f}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 72)
    print("THRESHOLD VALIDATION -- Theorems 3-5 vs DP Solver")
    print("=" * 72)
    print("Confirmed error: Error 1 (B1a Case 2, sign of coefficient).")
    print("Criterion: |diff| <= 1 -> OK (integrality rounding only).")
    print(f"I2bar query: fixed b1={B1_LARGE}.")

    print("\n\n" + "=" * 72)
    print("CASE A1  (Theorem 3)   pi1=pi2, alpha <= 0.5")
    print("=" * 72)
    for args in A1_CONFIGS:
        validate_A1(*args)

    print("\n\n" + "=" * 72)
    print("CASE A2  (Theorem 4)   pi1=pi2, alpha > 0.5")
    print("=" * 72)
    for args in A2_CONFIGS:
        validate_A2(*args)

    print("\n\n" + "=" * 72)
    print("CASE B1a (Theorem 5)   pi1>pi2, alpha2<=0.5   [Error 1]")
    print("=" * 72)
    for args in B1A_CONFIGS:
        validate_B1a(*args)

    print(f"\n{SEP}")
    print("RUN COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()