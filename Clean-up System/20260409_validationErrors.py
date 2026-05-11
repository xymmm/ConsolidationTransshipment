"""
validate_paper_original.py
==========================
Validation of the analytical formulas as stated in the paper (analytics.pdf),
WITHOUT any corrections.  Uses the paper's formulas verbatim:

  Theorem 3 (A1):  b1*(I2) and I2bar -- as stated, no errors in A1.
  Theorem 4 (A2):  kappa_paper = Phi - beta*(2*(alpha-0.5)-beta) - beta
                   (the extra -beta that we identified as Error 2 is KEPT).
  Theorem 5 (B1a): eta(tau) = 1 + 2*alpha2 - 2*gamma*tau  (Error 1 kept)
                   Phi_paper = 2*lam2*Cf/(h+pi1)          (Error 3 kept)

Logic is identical to validate_thresholds.py.  Solver, parameter sets,
threshold extraction, and error metric are unchanged.

Purpose: provide a clean record of how the paper's own formulas compare
against the DP ground truth, without mixing in corrections.
"""

import math
import numpy as np
from solver import Params, TransshipmentDP


# ======================================================================
# SECTION 0:  PARAMETER SETS  (identical to validate_thresholds.py)
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
# SECTION 1:  PAPER FORMULAS (verbatim, no corrections)
# ======================================================================

def _sqrt(x):
    return math.sqrt(max(0.0, x))


# -- A1 (Theorem 3) -- no errors in A1 ---------------------------------

def paper_A1_alpha(lam2, cu, h, pi):
    return lam2 * cu / (h + pi)

def paper_A1_Phi(lam2, Cf, h, pi):
    return 2 * lam2 * Cf / (h + pi)

def paper_A1_b1star(I2, lam2, cu, h, pi, Cf):
    """
    Paper Theorem 3, Case 1: b1*(I2).
    A = 2*I2 + 1 - 2*alpha
    Delta = A^2 - 4*Phi
    b1* = (A - sqrt(Delta)) / 2   if Delta >= 0, else None (always wait).
    """
    a   = paper_A1_alpha(lam2, cu, h, pi)
    phi = paper_A1_Phi(lam2, Cf, h, pi)
    A   = 2 * I2 + 1 - 2 * a
    D   = A**2 - 4 * phi
    if D < 0:
        return None
    return 0.5 * (A - _sqrt(D))

def paper_A1_I2bar(lam2, cu, h, pi, Cf):
    """
    Paper Theorem 3, Case 2: I2bar.
    xi = 1 - 2*alpha
    I2bar = (sqrt(xi^2 + 4*Phi) - xi) / 2
    """
    a   = paper_A1_alpha(lam2, cu, h, pi)
    phi = paper_A1_Phi(lam2, Cf, h, pi)
    xi  = 1 - 2 * a
    return 0.5 * (_sqrt(xi**2 + 4 * phi) - xi)


# -- A2 (Theorem 4) -- Error 2 kept ------------------------------------

def paper_A2_beta(lam2, cu, h, pi):
    return round(paper_A1_alpha(lam2, cu, h, pi) - 0.5)

def paper_A2_b1star(I2, lam2, cu, h, pi, Cf):
    """Paper Theorem 4, Case 1: same formula as A1 (no error in Case 1)."""
    return paper_A1_b1star(I2, lam2, cu, h, pi, Cf)

def paper_A2_I2bar(lam2, cu, h, pi, Cf):
    """
    Paper Theorem 4, Case 2: I2bar -- AS STATED IN PAPER (Error 2 kept).
    kappa = Phi - beta*(2*(alpha-0.5) - beta) - beta
                                               ^^^^^^ extra -beta (Error 2)
    Abar = 2*alpha - 1
    I2bar = (Abar + sqrt(Abar^2 + 4*kappa)) / 2
    """
    a    = paper_A1_alpha(lam2, cu, h, pi)
    phi  = paper_A1_Phi(lam2, Cf, h, pi)
    beta = paper_A2_beta(lam2, cu, h, pi)
    kap  = phi - beta * (2 * (a - 0.5) - beta) - beta   # Error 2 kept
    Abar = 2 * a - 1
    return 0.5 * (Abar + _sqrt(Abar**2 + 4 * kap))


# -- B1a (Theorem 5) -- Error 1 and Error 3 kept -----------------------

def paper_B1a_alpha2(lam2, cu, h, pi2):
    return lam2 * cu / (h + pi2)

def paper_B1a_Phi2(lam2, Cf, h, pi2):
    return 2 * lam2 * Cf / (h + pi2)

def paper_B1a_gamma(lam2, pi1, pi2, h):
    return lam2 * (pi1 - pi2) / (h + pi2)

def paper_B1a_b1star(I2, tau, lam2, cu, h, pi1, pi2, Cf):
    """
    Paper Theorem 5, Case 1: b1*(I2, tau) -- no error in Case 1.
    A(tau) = 2*I2 + 1 - 2*alpha2 + 2*gamma*tau
    Delta(tau) = A(tau)^2 - 4*Phi2
    b1* = (A - sqrt(Delta)) / 2
    """
    a2   = paper_B1a_alpha2(lam2, cu, h, pi2)
    phi2 = paper_B1a_Phi2(lam2, Cf, h, pi2)
    gam  = paper_B1a_gamma(lam2, pi1, pi2, h)
    A    = 2 * I2 + 1 - 2 * a2 + 2 * gam * tau
    D    = A**2 - 4 * phi2
    if D < 0:
        return None
    return 0.5 * (A - _sqrt(D))

def paper_B1a_I2bar(tau, lam2, cu, h, pi1, pi2, Cf):
    """
    Paper Theorem 5, Case 2: I2bar(tau) -- AS STATED IN PAPER.

    Error 1 kept: paper writes eta = 1 - 2*lam2*((pi1-pi2)*tau - cu)/(h+pi2)
                  which expands to  eta = 1 + 2*alpha2 - 2*gamma*tau
                  (sign of alpha2 and gamma*tau are both WRONG vs derivation)

    Error 3 kept: paper writes 8*lam2*Cf/(h+pi) in the sqrt term.
                  We interpret h+pi as h+pi1 (the non-subscripted pi in
                  Theorem 5 which uses the asymmetric setup where pi2 != pi1).
                  Phi_paper = 2*lam2*Cf/(h+pi1)

    I2bar = (sqrt(eta^2 + 4*Phi_paper) - eta) / 2
    """
    a2        = paper_B1a_alpha2(lam2, cu, h, pi2)
    gam       = paper_B1a_gamma(lam2, pi1, pi2, h)
    eta       = 1 + 2 * a2 - 2 * gam * tau              # Error 1
    phi_paper = 2 * lam2 * Cf / (h + pi1)               # Error 3
    return 0.5 * (_sqrt(eta**2 + 4 * phi_paper) - eta)


# ======================================================================
# SECTION 2:  DP HELPERS  (identical to validate_thresholds.py)
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
# SECTION 3:  VALIDATION RUNNERS
# ======================================================================

SEP = "-" * 72

def _diff_str(dp_val, analytic_val):
    if dp_val is None or analytic_val is None:
        return None, "--"
    d    = dp_val - math.ceil(analytic_val)
    flag = "OK" if abs(d) <= 1 else "NG"
    return d, flag


# -- Case A1 -----------------------------------------------------------

def validate_A1(label, lam2, cu, h, pi, Cf, c1, c2, v2,
                lam1=5.0, N=200, T=2.0):
    a   = paper_A1_alpha(lam2, cu, h, pi)
    phi = paper_A1_Phi(lam2, Cf, h, pi)

    print(f"\n{SEP}")
    print(f"CASE A1 | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi={pi}  Cf={Cf}"
          f"  c1=c2={c1}  v2={v2}  lam1={lam1}")
    print(f"  alpha={a:.4f}  Phi={phi:.4f}"
          f"  {'(alpha<0.5 OK)' if a < 0.5 else '(alpha>=0.5 WRONG CASE)'}")

    dp, p = build_and_solve(lam2, cu, h, pi, pi, Cf, lam1, T, N,
                            c1=c1, c2=c2, v2=v2)
    n_full = N

    # T1: q* structure
    viol = []
    for I2 in range(1, p.I2_max + 1):
        for b1 in range(1, p.b1_max + 1):
            q = dp.get_policy(n_full, I2, b1)
            if q > 0 and q != min(I2, b1):
                viol.append((I2, b1, q))
    print(f"\n  [T1] q* = min(I2,b1) whenever dispatching:")
    print(f"       {len(viol)} violation(s)"
          f"  {'(expect 0 OK)' if not viol else 'NG'}")
    for v in viol[:5]:
        print(f"       I2={v[0]} b1={v[1]} q*={v[2]} expected={min(v[0],v[1])}")

    # T2: b1* -- Case 1
    print(f"\n  [T2] b1*(I2) -- Case 1 (b1<=I2), tau=T   [paper Theorem 3]:")
    print(f"       {'I2':>4} | {'DP':>5} | {'Paper b1*':>10}"
          f" | {'ceil':>5} | {'diff':>5} | flag")
    diffs = []
    for I2 in range(2, min(20, p.I2_max) + 1):
        b1_dp = b1_threshold_dp(dp, n_full, I2)
        b1_an = paper_A1_b1star(I2, lam2, cu, h, pi, Cf)
        diff, flag = _diff_str(b1_dp, b1_an)
        if b1_an is None:
            print(f"       {I2:>4} | {'--':>5} | {'Delta<0':>10}"
                  f" | {'--':>5} | {'--':>5} | --")
            continue
        print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>10.4f}"
              f" | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
        if diff is not None:
            diffs.append(abs(diff))
    if diffs:
        print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")

    # T3: I2bar -- Case 2
    I2_dp  = I2_threshold_dp(dp, n_full)
    I2_bar = paper_A1_I2bar(lam2, cu, h, pi, Cf)
    diff, flag = _diff_str(I2_dp, I2_bar)
    print(f"\n  [T3] I2bar -- Case 2 (b1>=I2+1), tau=T, query b1={B1_LARGE}"
          f"   [paper Theorem 3]:")
    print(f"       DP={I2_dp}  Paper={I2_bar:.4f}"
          f"  ceil={math.ceil(I2_bar)}  diff={diff}  {flag}")


# -- Case A2 -----------------------------------------------------------

def validate_A2(label, lam2, cu, h, pi, Cf, c1, c2, v2,
                lam1=5.0, N=200, T=2.0):
    a    = paper_A1_alpha(lam2, cu, h, pi)
    phi  = paper_A1_Phi(lam2, Cf, h, pi)
    beta = paper_A2_beta(lam2, cu, h, pi)
    I2_paper = paper_A2_I2bar(lam2, cu, h, pi, Cf)
    kap_paper = phi - beta * (2 * (a - 0.5) - beta) - beta

    print(f"\n{SEP}")
    print(f"CASE A2 | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi={pi}  Cf={Cf}"
          f"  c1=c2={c1}  v2={v2}")
    print(f"  alpha={a:.4f}  Phi={phi:.4f}  beta={beta}"
          f"  {'(alpha>0.5 OK)' if a > 0.5 else '(alpha<=0.5 WRONG)'}")
    print(f"  kappa_paper={kap_paper:.4f}  I2bar_paper={I2_paper:.4f}")

    dp, p = build_and_solve(lam2, cu, h, pi, pi, Cf, lam1, T, N,
                            c1=c1, c2=c2, v2=v2)
    n_full   = N
    I2_floor = int(math.floor(a))

    # T1: pre-check
    print(f"\n  [T1] Pre-check: I2 <= alpha={a:.2f}  ->  always wait")
    for I2 in range(1, I2_floor + 1):
        nd   = sum(dp.get_policy(n_full, I2, b1) > 0
                   for b1 in range(1, p.b1_max + 1))
        flag = "OK" if nd == 0 else f"NG ({nd} dispatches)"
        print(f"       I2={I2}: {flag}")

    # T2: q* quantity -- partial dispatch
    print(f"\n  [T2] q* quantity: b1>=I2-beta+1 regime, expect q*=I2-beta={{}}"
          f"   [paper Theorem 4]:")
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

    # T3: I2bar -- paper formula (Error 2 kept)
    I2_dp = I2_threshold_dp(dp, n_full)
    diff, flag = _diff_str(I2_dp, I2_paper)
    print(f"\n  [T3] I2bar -- Case 2, tau=T, query b1={B1_LARGE}"
          f"   [paper Theorem 4, kappa includes extra -beta]:")
    print(f"       DP={I2_dp}  Paper={I2_paper:.4f}"
          f"  ceil={math.ceil(I2_paper)}  diff={diff}  {flag}")

    # T4: b1* -- Case 1
    print(f"\n  [T4] b1*(I2) -- Case 1 (b1<=I2-beta)   [paper Theorem 4]:")
    print(f"       {'I2':>4} | {'DP':>5} | {'Paper b1*':>10}"
          f" | {'ceil':>5} | {'diff':>5} | flag")
    diffs = []
    for I2 in range(I2_floor + 2, min(18, p.I2_max) + 1):
        b1_dp = b1_threshold_dp(dp, n_full, I2)
        b1_an = paper_A2_b1star(I2, lam2, cu, h, pi, Cf)
        diff, flag = _diff_str(b1_dp, b1_an)
        if b1_an is None:
            print(f"       {I2:>4} | {'--':>5} | {'Delta<0':>10}"
                  f" | {'--':>5} | {'--':>5} | --")
            continue
        print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>10.4f}"
              f" | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
        if diff is not None:
            diffs.append(abs(diff))
    if diffs:
        print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")


# -- Case B1a ----------------------------------------------------------

def validate_B1a(label, lam2, cu, h, pi1, pi2, Cf, c1, c2, v2,
                 lam1=5.0, N=200, T=2.0):
    a2       = paper_B1a_alpha2(lam2, cu, h, pi2)
    phi2     = paper_B1a_Phi2(lam2, Cf, h, pi2)
    gam      = paper_B1a_gamma(lam2, pi1, pi2, h)
    phi_paper = 2 * lam2 * Cf / (h + pi1)
    tau_star = (2 * a2 - 1) / (2 * gam) if gam != 0 else float('inf')

    print(f"\n{SEP}")
    print(f"CASE B1a | {label}")
    print(f"  lam2={lam2}  cu={cu}  h={h}  pi1={pi1}  pi2={pi2}"
          f"  Cf={Cf}  c1=c2={c1}  v2={v2}")
    print(f"  alpha2={a2:.4f}  Phi2(correct)={phi2:.4f}"
          f"  Phi(paper/pi1)={phi_paper:.4f}  gamma={gam:.4f}")
    print(f"  {'(alpha2<0.5 OK, gamma>0 OK)' if a2 < 0.5 and gam > 0 else 'WARNING'}")
    print(f"  tau* = {tau_star:.4f}"
          f"  {'(tau*<0 -> B1a region OK)' if tau_star < 0 else '(tau*>0 -> check region)'}")

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

    # T2: b1*(I2, tau) -- Case 1, no error in Case 1
    tau_fracs = [0.25, 0.50, 0.75, 1.00]
    print(f"\n  [T2] b1*(I2, tau) -- Case 1   [paper Theorem 5, no error in Case 1]:")
    for tf in tau_fracs:
        tau = tf * T
        n   = periods_for_tau(tf, N)
        print(f"\n       tau={tau:.2f}  (n={n})")
        print(f"       {'I2':>4} | {'DP':>5} | {'Paper b1*':>10}"
              f" | {'ceil':>5} | {'diff':>5} | flag")
        diffs = []
        for I2 in range(2, min(18, p.I2_max) + 1):
            b1_dp = b1_threshold_dp(dp, n, I2)
            b1_an = paper_B1a_b1star(I2, tau, lam2, cu, h, pi1, pi2, Cf)
            diff, flag = _diff_str(b1_dp, b1_an)
            if b1_an is None:
                print(f"       {I2:>4} | {'--':>5} | {'Delta<0':>10}"
                      f" | {'--':>5} | {'--':>5} | --")
                continue
            print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>10.4f}"
                  f" | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
            if diff is not None:
                diffs.append(abs(diff))
        if diffs:
            print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")

    # T3: I2bar(tau) -- paper formula (Error 1 + Error 3 kept)
    print(f"\n  [T3] I2bar(tau) -- Case 2, query b1={B1_LARGE}"
          f"   [paper Theorem 5, Error 1 + Error 3 kept]:")
    print(f"\n  {'tau':>5} | {'DP':>4} | {'Paper I2bar':>12}"
          f" | {'ceil':>5} | {'diff':>5} | flag")
    print(f"  {'-' * 52}")
    for tf in [0.10, 0.25, 0.50, 0.75, 1.00]:
        tau   = tf * T
        n     = periods_for_tau(tf, N)
        I2_dp = I2_threshold_dp(dp, n)
        I2_an = paper_B1a_I2bar(tau, lam2, cu, h, pi1, pi2, Cf)
        diff, flag = _diff_str(I2_dp, I2_an)
        print(f"  {tau:>5.2f} | {str(I2_dp):>4} | {I2_an:>12.4f}"
              f" | {math.ceil(I2_an):>5} | {str(diff):>5} | {flag}")

    # T4: eta trajectory -- to show how paper's formula behaves
    print(f"\n  [T4] Paper's eta(tau) = 1 + 2*alpha2 - 2*gamma*tau"
          f"   [Error 1 in full view]:")
    print(f"       {'tau':>5} | {'eta_paper':>10} | {'Phi_paper':>10}"
          f" | {'I2bar_paper':>12}")
    print(f"       {'-' * 48}")
    for tf in [0.10, 0.25, 0.50, 0.75, 1.00]:
        tau = tf * T
        eta = 1 + 2 * a2 - 2 * gam * tau
        I2_an = paper_B1a_I2bar(tau, lam2, cu, h, pi1, pi2, Cf)
        print(f"       {tau:>5.2f} | {eta:>10.4f} | {phi_paper:>10.4f}"
              f" | {I2_an:>12.4f}")
    print(f"       Note: eta goes negative for tau > alpha2/gamma"
          f" = {a2/gam:.4f}")
    print(f"       When eta < 0 and large: I2bar_paper = "
          f"(sqrt(eta^2+4*Phi)-eta)/2 grows rapidly")


# ======================================================================
# SECTION 4:  MAIN
# ======================================================================

def main():
    print("=" * 72)
    print("PAPER ORIGINAL VALIDATION -- Theorems 3-5 as stated in paper")
    print("=" * 72)
    print("Using paper formulas VERBATIM -- no corrections applied.")
    print("  A1: no errors -> same as corrected version")
    print("  A2: kappa = Phi - beta*(2*(alpha-0.5)-beta) - beta  [Error 2 kept]")
    print("  B1a Case 2: eta = 1+2*alpha2-2*gamma*tau            [Error 1 kept]")
    print("              Phi = 2*lam2*Cf/(h+pi1)                 [Error 3 kept]")
    print(f"Solver: corrected g() (post-dispatch flow cost).")
    print(f"Settings: N=200, T=2.0, b1_max=80, I2_max=35, I2_min=-10.")
    print(f"I2bar query: fixed b1={B1_LARGE}.")
    print("Criterion: |diff| <= 1 -> OK.")

    print("\n\n" + "=" * 72)
    print("CASE A1  (Theorem 3)   pi1=pi2, alpha<=0.5   [no errors]")
    print("=" * 72)
    for args in A1_CONFIGS:
        validate_A1(*args)

    print("\n\n" + "=" * 72)
    print("CASE A2  (Theorem 4)   pi1=pi2, alpha>0.5"
          "   [Error 2 in I2bar]")
    print("=" * 72)
    for args in A2_CONFIGS:
        validate_A2(*args)

    print("\n\n" + "=" * 72)
    print("CASE B1a (Theorem 5)   pi1>pi2, alpha2<=0.5"
          "   [Errors 1&3 in I2bar]")
    print("=" * 72)
    for args in B1A_CONFIGS:
        validate_B1a(*args)

    print(f"\n{SEP}")
    print("RUN COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()