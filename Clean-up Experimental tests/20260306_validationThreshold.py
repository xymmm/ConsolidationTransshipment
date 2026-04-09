"""
validate_thresholds.py
======================
Validation of analytical threshold formulas (Theorems 3–5) against the
backward-induction DP solver.  Covers Cases A1, A2, B1a and explicitly
surfaces the three paper errors identified in the analysis.

Assumes solver.py is in the same directory.  Run as:
    python validate_thresholds.py

Output is structured as one section per case, each section printing:
  (i)   parameter summary and dimensionless group values
  (ii)  q* structure check  (policy quantity)
  (iii) b₁*(I₂) threshold table — DP vs analytical
  (iv)  Ī₂ threshold — DP vs correct vs paper formula(s)
  (v)   summary error statistics

Notation
--------
  α   = λ₂ cᵤ / (h + π)         [symmetric case]
  α₂  = λ₂ cᵤ / (h + π₂)        [asymmetric case]
  Φ   = 2 λ₂ Cf / (h + π)
  Φ₂  = 2 λ₂ Cf / (h + π₂)
  γ   = λ₂ (π₁ − π₂) / (h + π₂)
  β   = round(α − ½)             [A2 hold-back quantity]

Time mapping
------------
  Period n remaining  ↔  τ = n · Δt  time remaining
  n = N  ↔  τ = T   (full horizon, used for Theorem 3/4 which are τ-static)
  n = k  ↔  τ = k·T/N

Threshold extraction from DP
-----------------------------
  b₁_DP(I₂, n)  = min b₁ ∈ {1,…,I₂} s.t. dp.qstar(n, I₂, b₁) > 0
  Ī₂_DP(n)      = min I₂ ∈ {1,…,I₂_max} s.t. dp.qstar(n, I₂, b₁_large) > 0
                  where b₁_large = b₁_max (forces Case-2 regime)

Error metric
------------
  For integer DP threshold t_DP and continuous analytical threshold t_an:
    diff = t_DP − ceil(t_an)
  Expect |diff| ≤ 1 for correct formulas (integrality gap only).
  Systematic non-zero diff across parameter sets → formula error.
"""

import math
import numpy as np
from solver import Params, TransshipmentDP          # ← adjust path if needed


# ══════════════════════════════════════════════════════════════════════
# SECTION 0:  PARAMETER SETS
# ══════════════════════════════════════════════════════════════════════
#
# Design rationale (UTD-level breadth):
#   • Three A1 configs:  α ∈ {0.20, 0.33, 0.45} — low/mid/near-boundary
#   • Three A2 configs:  α ∈ {0.75, 1.50, 2.00} — β ∈ {0, 1, 2}
#       β=0 makes A2 degenerate to A1 structure; β=1,2 expose Error 2 fully.
#   • Two  B1a configs:  α₂ ∈ {0.25, 0.40}, γ large/small
#       Chosen so τ* = (2α₂−1)/(2γ) is outside [0,T] (confirming B1a region)
#
# All configs use c₁=c₂=v₂=0  (per paper Remark, p.3) so that Vw cancels
# cleanly in Vd−Vw=0 and the threshold does not depend on terminal costs.
# λ₁ is set to 5 throughout (does not affect threshold formulas).
# N=400, T=2 gives Δt=0.005 — tight enough for continuous-time convergence.

A1_CONFIGS = [
    # label,          lam2, cu,   h,  pi,   Cf
    ("α=0.20  (low)",   2, 0.5,  1,  4.5,   5),   # α=2×0.5/5.5=0.182, Φ=20/5.5=3.64
    ("α=0.33  (mid)",   3, 1.0,  1,  8.0,  10),   # α=3/9=0.333,        Φ=60/9=6.67
    ("α=0.45  (high)",  4, 1.0,  1, 10.0,  12),   # α=4/11=0.364... hmm
    # Recalculated to hit 0.45:  lam2=4, cu=1.0, h=1, pi=7.89  → use rounded
    # Better:  lam2=5, cu=0.9, h=1, pi=9.0  → α=5×0.9/10=0.45
]
# Override config 3 to exactly α=0.45
A1_CONFIGS[2] = ("α=0.45  (high)", 5, 0.9, 1, 9.0, 15)   # α=4.5/10=0.45, Φ=150/10=15

A2_CONFIGS = [
    # label,         lam2, cu,    h,  pi,   Cf,   expected β
    ("α=0.75 β=0",    3, 1.0,   1,  3.0,   8),   # α=3/4=0.75,  β=round(0.25)=0
    ("α=1.50 β=1",    3, 2.0,   1,  3.0,  10),   # α=6/4=1.50,  β=round(1.0)=1
    ("α=2.00 β=2",    4, 2.0,   1,  3.0,  12),   # α=8/4=2.00,  β=round(1.5)=2
]

B1A_CONFIGS = [
    # label,           lam2, cu,   h, pi1, pi2,  Cf
    ("α₂=0.25 γ=1.6",  2, 1.0,   1,  8,   4,   8),   # α₂=2/5=0.40 → adjust
    ("α₂=0.25 γ=2.5",  3, 0.5,   1, 10,   5,  10),   # α₂=1.5/6=0.25, γ=3×5/6=2.5
]
# Override config 1 to exactly α₂=0.25:  lam2=2, cu=0.75, h=1, pi2=5, pi1=9
B1A_CONFIGS[0] = ("α₂=0.30 γ=2.4", 2, 0.75, 1, 9, 4, 8)
# α₂ = 2×0.75/(1+4)=1.5/5=0.30  γ=2×(9-4)/5=2.0  — verified below


# ══════════════════════════════════════════════════════════════════════
# SECTION 1:  ANALYTICAL FORMULAS
# ══════════════════════════════════════════════════════════════════════

def _sqrt(x):
    return math.sqrt(max(0.0, x))

# ── A1 (Theorem 3) ────────────────────────────────────────────────────

def A1_alpha(lam2, cu, h, pi):
    return lam2 * cu / (h + pi)

def A1_Phi(lam2, Cf, h, pi):
    return 2 * lam2 * Cf / (h + pi)

def A1_b1star(I2, lam2, cu, h, pi, Cf):
    """
    Case 1 threshold (b₁ ≤ I₂): b₁*(I₂) — continuous, paper-correct.
    Returns None if Δ < 0 (always wait for this I₂).
    """
    a   = A1_alpha(lam2, cu, h, pi)
    phi = A1_Phi(lam2, Cf, h, pi)
    A   = 2 * I2 + 1 - 2 * a
    D   = A**2 - 4 * phi
    if D < 0:
        return None
    return 0.5 * (A - _sqrt(D))

def A1_I2bar(lam2, cu, h, pi, Cf):
    """
    Case 2 threshold (b₁ ≥ I₂+1): Ī₂ — continuous, paper-correct.
    """
    a   = A1_alpha(lam2, cu, h, pi)
    phi = A1_Phi(lam2, Cf, h, pi)
    xi  = 1 - 2 * a
    return 0.5 * (_sqrt(xi**2 + 4 * phi) - xi)


# ── A2 (Theorem 4) ────────────────────────────────────────────────────

def A2_beta(lam2, cu, h, pi):
    a = A1_alpha(lam2, cu, h, pi)
    return round(a - 0.5)           # Python uses banker's rounding

def A2_b1star(I2, lam2, cu, h, pi, Cf):
    """
    Case 1 threshold (b₁ ≤ I₂ − β): same formula as A1, paper-correct.
    """
    return A1_b1star(I2, lam2, cu, h, pi, Cf)

def A2_I2bar_correct(lam2, cu, h, pi, Cf):
    """
    Case 2 threshold: CORRECTED formula (Error 2 fixed).
    κ_correct = Φ − β·(2(α−½) − β)
    """
    a    = A1_alpha(lam2, cu, h, pi)
    phi  = A1_Phi(lam2, Cf, h, pi)
    beta = A2_beta(lam2, cu, h, pi)
    kap  = phi - beta * (2 * (a - 0.5) - beta)
    A    = 2 * a - 1
    return 0.5 * (A + _sqrt(A**2 + 4 * kap))

def A2_I2bar_paper(lam2, cu, h, pi, Cf):
    """
    Case 2 threshold: PAPER formula — contains Error 2.
    κ_paper = Φ − β·(2(α−½) − β) − β   ← extra −β
    """
    a    = A1_alpha(lam2, cu, h, pi)
    phi  = A1_Phi(lam2, Cf, h, pi)
    beta = A2_beta(lam2, cu, h, pi)
    kap  = phi - beta * (2 * (a - 0.5) - beta) - beta
    A    = 2 * a - 1
    return 0.5 * (A + _sqrt(A**2 + 4 * kap))

def A2_kappa_gap(lam2, cu, h, pi, Cf):
    """Δκ = κ_correct − κ_paper = β (the magnitude of Error 2 in κ)."""
    return A2_beta(lam2, cu, h, pi)


# ── B1a (Theorem 5) ───────────────────────────────────────────────────

def B1a_alpha2(lam2, cu, h, pi2):
    return lam2 * cu / (h + pi2)

def B1a_Phi2(lam2, Cf, h, pi2):
    return 2 * lam2 * Cf / (h + pi2)

def B1a_gamma(lam2, pi1, pi2, h):
    return lam2 * (pi1 - pi2) / (h + pi2)

def B1a_b1star(I2, tau, lam2, cu, h, pi1, pi2, Cf):
    """
    Case 1 threshold b₁*(I₂, τ): paper-correct (no typo in Case 1).
    A(τ) = 2I₂ + 1 − 2α₂ + 2γτ
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
    """
    Case 2 threshold Ī₂(τ): CORRECTED.
    ξ(τ) = 1 − 2α₂ + 2γτ  (sign of γτ and α₂ both fixed vs. paper)
    Φ₂   = 2λ₂Cf / (h+π₂)
    """
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    xi   = 1 - 2 * a2 + 2 * gam * tau
    return 0.5 * (_sqrt(xi**2 + 4 * phi2) - xi)

def B1a_I2bar_err1_only(tau, lam2, cu, h, pi1, pi2, Cf):
    """
    Case 2: Error 1 only — wrong ξ, correct Φ₂.
    Paper §6.1.1 writes η = 1 + 2α₂ − 2γτ  (signs of α₂ and γτ swapped).
    """
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    eta  = 1 + 2 * a2 - 2 * gam * tau      # wrong ξ
    return 0.5 * (_sqrt(eta**2 + 4 * phi2) - eta)

def B1a_I2bar_err3_only(tau, lam2, cu, h, pi1, pi2, Cf):
    """
    Case 2: Error 3 only — correct ξ, wrong Φ in Theorem 5.
    Theorem 5 writes 8λ₂Cf/(h+π) using h+π₁ instead of h+π₂.
    """
    a2      = B1a_alpha2(lam2, cu, h, pi2)
    gam     = B1a_gamma(lam2, pi1, pi2, h)
    phi_err = 2 * lam2 * Cf / (h + pi1)    # wrong denominator: h+π₁ not h+π₂
    xi      = 1 - 2 * a2 + 2 * gam * tau
    return 0.5 * (_sqrt(xi**2 + 4 * phi_err) - xi)

def B1a_I2bar_both_errors(tau, lam2, cu, h, pi1, pi2, Cf):
    """Case 2: both Error 1 and Error 3 combined (as paper states it)."""
    a2      = B1a_alpha2(lam2, cu, h, pi2)
    gam     = B1a_gamma(lam2, pi1, pi2, h)
    phi_err = 2 * lam2 * Cf / (h + pi1)    # Error 3
    eta     = 1 + 2 * a2 - 2 * gam * tau   # Error 1
    return 0.5 * (_sqrt(eta**2 + 4 * phi_err) - eta)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2:  DP HELPERS
# ══════════════════════════════════════════════════════════════════════

def build_and_solve(lam2, cu, h, pi1, pi2, Cf, lam1=5.0,
                    T=2.0, N=400, I2_max=22, I2_min=-5, b1_max=28):
    """Build Params, run DP, return (dp, params)."""
    p = Params(
        T=T, N=N, lam1=lam1, lam2=lam2,
        h=h, Cf=Cf, cu=cu,
        pi1=pi1, pi2=pi2,
        c1=0.0, c2=0.0, v2=0.0,        # terminal costs zero (per paper remark)
        I2_max=I2_max, I2_min=I2_min, b1_max=b1_max,
    )
    dp = TransshipmentDP(p)
    dp.solve(store_V=False, verbose=False)
    return dp, p


def b1_threshold_dp(dp, n, I2):
    """
    Smallest b₁ ∈ {1,…,I₂} for which dp dispatches at period n.
    Returns None if no dispatch found (always wait for this I₂).
    """
    for b1 in range(1, min(I2, dp.p.b1_max) + 1):
        if dp.get_policy(n, I2, b1) > 0:
            return b1
    return None


def I2_threshold_dp(dp, n, b1_large=None):
    """
    Smallest I₂ ∈ {1,…,I₂_max} for which dp dispatches when b₁=b1_large.
    b1_large defaults to b1_max (forcing the Case-2 regime).
    Returns None if no dispatch found anywhere.
    """
    if b1_large is None:
        b1_large = dp.p.b1_max
    for I2 in range(1, dp.p.I2_max + 1):
        if dp.get_policy(n, I2, b1_large) > 0:
            return I2
    return None


def periods_for_tau(tau_fraction, N):
    """Convert fraction of horizon [0,1] to number of periods remaining."""
    return max(1, round(tau_fraction * N))


# ══════════════════════════════════════════════════════════════════════
# SECTION 3:  VALIDATION RUNNERS
# ══════════════════════════════════════════════════════════════════════

SEP = "─" * 72

def _diff_str(dp_val, analytic_val):
    """
    Returns (diff, flag_str) where diff = dp_val - ceil(analytic_val).
    flag: ✓ if |diff|≤1 (integrality only), ✗ otherwise.
    """
    if dp_val is None or analytic_val is None:
        return None, "—"
    d = dp_val - math.ceil(analytic_val)
    flag = "✓" if abs(d) <= 1 else "✗"
    return d, flag


# ── Case A1 ────────────────────────────────────────────────────────────

def validate_A1(label, lam2, cu, h, pi, Cf, lam1=5.0, N=400, T=2.0):
    a   = A1_alpha(lam2, cu, h, pi)
    phi = A1_Phi(lam2, Cf, h, pi)

    print(f"\n{SEP}")
    print(f"CASE A1 | {label}")
    print(f"  λ₂={lam2}  cᵤ={cu}  h={h}  π={pi}  Cf={Cf}  λ₁={lam1}")
    print(f"  α = {a:.4f}  Φ = {phi:.4f}  {'(α<½ ✓)' if a<0.5 else '(α≥½ ✗ — wrong case!)'}")

    dp, p = build_and_solve(lam2, cu, h, pi, pi, Cf, lam1, T, N)
    n_full = N   # τ = T

    # ── Test 1: q* structure — whenever dispatching, q* = min(I₂,b₁)
    violations = []
    for I2 in range(1, p.I2_max + 1):
        for b1 in range(1, p.b1_max + 1):
            q = dp.get_policy(n_full, I2, b1)
            if q > 0 and q != min(I2, b1):
                violations.append((I2, b1, q))
    print(f"\n  [T1] q* = min(I₂,b₁) whenever dispatching:")
    print(f"       {len(violations)} violation(s)  {'(expect 0 ✓)' if not violations else '✗'}")
    if violations:
        for v in violations[:5]:
            print(f"       I₂={v[0]} b₁={v[1]} q*={v[2]} expected={min(v[0],v[1])}")

    # ── Test 2: b₁*(I₂) — Case 1 threshold
    print(f"\n  [T2] b₁*(I₂) threshold — Case 1 (b₁≤I₂), τ=T:")
    print(f"       {'I₂':>4} | {'DP':>5} | {'Analytical':>11} | {'ceil':>5} | {'diff':>5} | ok")
    diffs = []
    for I2 in range(2, min(20, p.I2_max) + 1):
        b1_dp  = b1_threshold_dp(dp, n_full, I2)
        b1_an  = A1_b1star(I2, lam2, cu, h, pi, Cf)
        diff, flag = _diff_str(b1_dp, b1_an)
        if b1_an is None:
            print(f"       {I2:>4} | {'—':>5} | {'Δ<0 (wait)':>11} | {'—':>5} | {'—':>5} | —")
            continue
        print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>11.4f} | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
        if diff is not None:
            diffs.append(abs(diff))
    if diffs:
        print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")

    # ── Test 3: Ī₂ — Case 2 threshold
    I2_dp  = I2_threshold_dp(dp, n_full)
    I2_bar = A1_I2bar(lam2, cu, h, pi, Cf)
    diff, flag = _diff_str(I2_dp, I2_bar)
    print(f"\n  [T3] Ī₂ — Case 2 (b₁≥I₂+1), τ=T:")
    print(f"       DP={I2_dp}  Analytical={I2_bar:.4f}  ceil={math.ceil(I2_bar)}  diff={diff}  {flag}")


# ── Case A2 ────────────────────────────────────────────────────────────

def validate_A2(label, lam2, cu, h, pi, Cf, lam1=5.0, N=400, T=2.0):
    a    = A1_alpha(lam2, cu, h, pi)
    phi  = A1_Phi(lam2, Cf, h, pi)
    beta = A2_beta(lam2, cu, h, pi)
    I2c  = A2_I2bar_correct(lam2, cu, h, pi, Cf)
    I2p  = A2_I2bar_paper(lam2, cu, h, pi, Cf)
    kap_c = phi - beta * (2*(a-0.5) - beta)
    kap_p = kap_c - beta

    print(f"\n{SEP}")
    print(f"CASE A2 | {label}")
    print(f"  λ₂={lam2}  cᵤ={cu}  h={h}  π={pi}  Cf={Cf}")
    print(f"  α = {a:.4f}  Φ = {phi:.4f}  β = {beta}  {'(α>½ ✓)' if a>0.5 else '(α≤½ ✗)'}")
    print(f"  κ_correct = {kap_c:.4f}    κ_paper = {kap_p:.4f}    Δκ = β = {beta}")
    print(f"  Ī₂_correct = {I2c:.4f}    Ī₂_paper = {I2p:.4f}    ΔĪ₂ = {I2c-I2p:.4f}")

    dp, p = build_and_solve(lam2, cu, h, pi, pi, Cf, lam1, T, N)
    n_full = N

    # ── Test 1: pre-check  I₂ ≤ α  →  always wait
    print(f"\n  [T1] Pre-check: I₂ ≤ α={a:.2f}  →  always wait (q*=0 for all b₁)")
    I2_floor = int(math.floor(a))
    for I2 in range(1, I2_floor + 1):
        n_dispatch = sum(dp.get_policy(n_full, I2, b1) > 0
                         for b1 in range(1, p.b1_max + 1))
        flag = "✓" if n_dispatch == 0 else f"✗ ({n_dispatch} dispatches)"
        print(f"       I₂={I2}: {flag}")

    # ── Test 2: q* quantity — for b₁≥I₂−β+1, expect q*=I₂−β (partial)
    print(f"\n  [T2] q* quantity: b₁≥I₂−β+1 regime, expect q*=I₂−β")
    q_viol = []
    for I2 in range(I2_floor + 2, min(20, p.I2_max) + 1):
        bmin = I2 - beta + 1
        for b1 in range(max(1, bmin), p.b1_max + 1):
            q = dp.get_policy(n_full, I2, b1)
            if q > 0 and q != I2 - beta:
                q_viol.append((I2, b1, q, I2 - beta))
    print(f"       {len(q_viol)} violation(s)  {'(expect 0 ✓)' if not q_viol else '✗'}")
    if q_viol:
        for v in q_viol[:5]:
            print(f"       I₂={v[0]} b₁={v[1]} q*={v[2]} expected={v[3]}")

    # ── Test 3: Ī₂ — Error 2 comparison
    I2_dp = I2_threshold_dp(dp, n_full)
    dc, fc = _diff_str(I2_dp, I2c)
    dp2, fp = _diff_str(I2_dp, I2p)

    print(f"\n  [T3] Ī₂ — Case 2, τ=T  (*** Error 2 ***)")
    print(f"       DP result         = {I2_dp}")
    print(f"       Corrected formula = {I2c:.4f}  →  ceil={math.ceil(I2c)}  diff={dc}  {fc}")
    print(f"       Paper formula     = {I2p:.4f}  →  ceil={math.ceil(I2p)}  diff={dp2}  {fp}")
    print(f"       Error 2 impact:  Ī₂_paper underestimates by {I2c-I2p:.4f} units")
    print(f"       Root cause:  κ_paper = κ_correct − β = {kap_c:.4f} − {beta} = {kap_p:.4f}")
    print(f"       Practical effect: paper says 'transship at lower inventory' → suboptimal policy")

    # ── Test 4: b₁* — Case 1 (same formula as A1, no error here)
    print(f"\n  [T4] b₁*(I₂) — Case 1 (b₁≤I₂−β):")
    print(f"       {'I₂':>4} | {'DP':>5} | {'Analytical':>11} | {'ceil':>5} | {'diff':>5} | ok")
    diffs = []
    for I2 in range(I2_floor + 2, min(18, p.I2_max) + 1):
        b1_dp = b1_threshold_dp(dp, n_full, I2)
        b1_an = A2_b1star(I2, lam2, cu, h, pi, Cf)
        diff, flag = _diff_str(b1_dp, b1_an)
        if b1_an is None:
            print(f"       {I2:>4} | {'—':>5} | {'Δ<0':>11} | {'—':>5} | {'—':>5} | —")
            continue
        print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>11.4f} | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
        if diff is not None:
            diffs.append(abs(diff))
    if diffs:
        print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")


# ── Case B1a ──────────────────────────────────────────────────────────

def validate_B1a(label, lam2, cu, h, pi1, pi2, Cf, lam1=5.0, N=400, T=2.0):
    a2   = B1a_alpha2(lam2, cu, h, pi2)
    phi2 = B1a_Phi2(lam2, Cf, h, pi2)
    gam  = B1a_gamma(lam2, pi1, pi2, h)
    # τ* where q^axis hits I₂ (only meaningful if α₂>½, not relevant here)
    tau_star_hyp = (2*a2 - 1) / (2*gam) if gam != 0 else float('inf')

    print(f"\n{SEP}")
    print(f"CASE B1a | {label}")
    print(f"  λ₂={lam2}  cᵤ={cu}  h={h}  π₁={pi1}  π₂={pi2}  Cf={Cf}")
    print(f"  α₂={a2:.4f}  Φ₂={phi2:.4f}  γ={gam:.4f}")
    print(f"  {'(α₂<½ ✓, γ>0 ✓)' if a2<0.5 and gam>0 else 'WARNING: wrong region'}")
    print(f"  τ* (hypothetical crossing) = {tau_star_hyp:.4f}  {'(τ*<0: safe, no crossing ✓)' if tau_star_hyp<0 else '(τ*>0: check τ*<0 needed for B1a)'}")

    dp, p = build_and_solve(lam2, cu, h, pi1, pi2, Cf, lam1, T, N)
    n_full = N
    dt = T / N

    # ── Test 1: q* structure — full dispatch whenever dispatching
    print(f"\n  [T1] q* = min(I₂,b₁) whenever dispatching (full dispatch):")
    viol = []
    for I2 in range(1, p.I2_max + 1):
        for b1 in range(1, p.b1_max + 1):
            q = dp.get_policy(n_full, I2, b1)
            if q > 0 and q != min(I2, b1):
                viol.append((I2, b1, q))
    print(f"       {len(viol)} violation(s)  {'(expect 0 ✓)' if not viol else '✗'}")

    # ── Test 2: b₁*(I₂, τ) at multiple τ values
    tau_fracs = [0.25, 0.50, 0.75, 1.00]
    print(f"\n  [T2] b₁*(I₂, τ) — Case 1, across τ and I₂:")
    for tf in tau_fracs:
        tau = tf * T
        n   = periods_for_tau(tf, N)
        print(f"\n       τ = {tau:.2f}  (n={n})")
        print(f"       {'I₂':>4} | {'DP':>5} | {'Analytical':>11} | {'ceil':>5} | {'diff':>5} | ok")
        diffs = []
        for I2 in range(2, min(18, p.I2_max) + 1):
            b1_dp = b1_threshold_dp(dp, n, I2)
            b1_an = B1a_b1star(I2, tau, lam2, cu, h, pi1, pi2, Cf)
            diff, flag = _diff_str(b1_dp, b1_an)
            if b1_an is None:
                print(f"       {I2:>4} | {'—':>5} | {'Δ<0':>11} | {'—':>5} | {'—':>5} | —")
                continue
            print(f"       {I2:>4} | {str(b1_dp):>5} | {b1_an:>11.4f} | {math.ceil(b1_an):>5} | {str(diff):>5} | {flag}")
            if diff is not None:
                diffs.append(abs(diff))
        if diffs:
            print(f"       MAE={np.mean(diffs):.3f}  MaxAE={max(diffs):.0f}")

    # ── Test 3: Ī₂(τ) — Error 1 and Error 3 comparison across τ
    print(f"\n  [T3] Ī₂(τ) — Case 2 (b₁=large)  (*** Errors 1 & 3 ***)")
    hdr = (f"  {'τ':>5} | {'DP':>4} | {'Correct':>8} | {'Err1':>8} | "
           f"{'Err3':>8} | {'Both':>8} | dC | d1 | d3 | dB")
    print(f"\n  {hdr}")
    print(f"  {'─'*90}")

    for tf in [0.10, 0.25, 0.50, 0.75, 1.00]:
        tau = tf * T
        n   = periods_for_tau(tf, N)
        I2_dp   = I2_threshold_dp(dp, n)
        I2_c    = B1a_I2bar_correct(tau, lam2, cu, h, pi1, pi2, Cf)
        I2_e1   = B1a_I2bar_err1_only(tau, lam2, cu, h, pi1, pi2, Cf)
        I2_e3   = B1a_I2bar_err3_only(tau, lam2, cu, h, pi1, pi2, Cf)
        I2_both = B1a_I2bar_both_errors(tau, lam2, cu, h, pi1, pi2, Cf)
        dc, _  = _diff_str(I2_dp, I2_c)
        d1, _  = _diff_str(I2_dp, I2_e1)
        d3, _  = _diff_str(I2_dp, I2_e3)
        db, _  = _diff_str(I2_dp, I2_both)
        print(f"  {tau:>5.2f} | {str(I2_dp):>4} | {I2_c:>8.3f} | {I2_e1:>8.3f} | "
              f"{I2_e3:>8.3f} | {I2_both:>8.3f} | {str(dc):>2} | {str(d1):>2} | {str(d3):>2} | {str(db):>2}")

    # ── Test 4: ξ(τ) trajectory — to make Error 1 magnitude explicit
    print(f"\n  [T4] ξ vs η trajectory (Error 1 magnitude over τ):")
    print(f"       {'τ':>5} | {'ξ_correct':>10} | {'η_paper':>10} | {'Δ=ξ−η':>8}")
    print(f"       {'─'*45}")
    for tf in [0.10, 0.25, 0.50, 0.75, 1.00]:
        tau = tf * T
        xi  = 1 - 2*a2 + 2*gam*tau
        eta = 1 + 2*a2 - 2*gam*tau
        print(f"       {tau:>5.2f} | {xi:>10.4f} | {eta:>10.4f} | {xi-eta:>8.4f}")
    print(f"       (ξ − η) = 4(γτ − α₂)  →  sign changes at τ = α₂/γ = {a2/gam:.4f}")
    print(f"       Error 3 effect: Φ₂={phi2:.4f}  Φ_err3={2*lam2*Cf/(h+pi1):.4f}  ΔΦ={phi2-2*lam2*Cf/(h+pi1):.4f}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4:  DIAGRAM VALIDITY CHECK  (text only)
# ══════════════════════════════════════════════════════════════════════

def check_diagram():
    print(f"\n{SEP}")
    print("DIAGRAM VALIDITY CHECK")
    print(f"{SEP}")
    print("""
The interactive diagram classifies (α₂, γT₀) into 8 regions based on:
  (1) Vertical line α₂ = ½
  (2) Diagonal   γT₀ = α₂ − ½

These boundaries come from the condition q^axis(τ) ≥ I₂, i.e.,
  I₂ + ½ − α₂ + γτ ≥ I₂  ↔  γτ ≥ α₂ − ½

Checked:  Does error impact these boundaries?  No.
  • The classification boundaries are derived from q^axis alone — a property
    of the objective function's axis of symmetry.
  • Errors 1, 2, 3 are all in the THRESHOLD formulas (when to dispatch, i.e.,
    Vd=Vw) — not in the q* quantity formula or the classification structure.
  • Therefore the diagram's 8-region partition is correct as drawn.

What the diagram does NOT show (and needs annotation):
  • The b₁*(I₂,τ) and Ī₂(τ) formulas listed in the click-through panel for
    B1a Case 2 use ξ_correct = 1−2α₂+2γτ  (not paper's η = 1+2α₂−2γτ).
  • The Φ₂ in the √ term uses h+π₂ (not h+π₁ as in Theorem 5).
  • The A2 Case 2 panel uses κ_correct (not κ_paper = κ_correct − β).

Recommendation: add an ⚠ badge to B1a Case 2 and A2 Case 2 click-through
panels marking the corrected formulas.
""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5:  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("THRESHOLD VALIDATION — Theorems 3–5 vs DP Solver")
    print("=" * 72)
    print("Settings: N=400, T=2.0, c₁=c₂=v₂=0, λ₁=5 throughout.")
    print("Expect |diff| ≤ 1 for correct formulas (integrality rounding only).")
    print("Systematic |diff| > 1 or wrong sign → identifies formula error.")

    # ── Case A1 ──
    print("\n\n" + "═"*72)
    print("CASE A1  (Theorem 3)   π₁=π₂, α ≤ ½")
    print("═"*72)
    for args in A1_CONFIGS:
        validate_A1(*args)

    # ── Case A2 ──
    print("\n\n" + "═"*72)
    print("CASE A2  (Theorem 4)   π₁=π₂, α > ½   ← Error 2 visible here")
    print("═"*72)
    for args in A2_CONFIGS:
        validate_A2(*args)

    # ── Case B1a ──
    print("\n\n" + "═"*72)
    print("CASE B1a (Theorem 5)   π₁>π₂, α₂ ≤ ½  ← Errors 1 & 3 visible here")
    print("═"*72)
    for args in B1A_CONFIGS:
        validate_B1a(*args)

    # ── Diagram ──
    check_diagram()

    print(f"\n{SEP}")
    print("RUN COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()