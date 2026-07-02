"""
validate_pdf.py
===============================================================================
Numerical validation of "Dispatch and transshipment finite horizon"
(note dated 2026-07-01) against the backward-induction DP in solver.py.

NOT executed here by design. Intended usage:

    python validate_pdf.py            # full run (pure-python DP: slow, ~tens of min)
    FAST=1 python validate_pdf.py     # coarse smoke test

Tests (one per result in the note):
  TEST 1  Closed-form V(I2<=0, b1, tau)                 [Sec 2 -- EXACT result]
  TEST 2  Boundary V(0, b1, tau), Eq.(15)               [Sec 3 -- EXACT result]
  TEST 3  Vw solves PDE (16) + value-matching at I2=lam2*tau   [Sec 3, Eq.(19)]
  TEST 4  Fluid gap: Vw vs exact no-dispatch DP         [Sec 3 -- approximation]
  TEST 5  Proposition 1 lower-bound property            [Sec 2.1]
  TEST 6  Theorem 2 threshold structure of q*           [Sec 2.2]
  TEST 7  Theorem 3 policy vs DP policy                 [Sec 5]
          + 7a probe: Region 1 branch-crossing flaw (safety stock retained?)
          + 7b probe: Region 1 ignores Cf in the dispatch/wait test
  TEST 8  solver.py <-> note consistency unit checks

Conventions:
  tau = n * dt (time-to-go), so DP value V^n corresponds to V(., ., n*dt).
  For all Theorem 3 / Section 5 tests we set c1 = c2 = v2 = 0 (the note's
  assumption).  solver.py defaults are c1=c2=5, v2=1 -- must be overridden.

Truncation control (per user note):
  b1_max is set VERY high:  b1_max >= b1_probe + lam1*T + 8*sqrt(lam1*T)
  I2_min is set VERY low:   I2_min <= -(lam2*T + 8*sqrt(lam2*T))
  Both matter: clipping at either boundary contaminates interior values.
===============================================================================
"""

import math
import os
import numpy as np

from solver import Params, TransshipmentDP

FAST = bool(int(os.environ.get("FAST", "0")))
N_DEFAULT = 100 if FAST else 400
TOPK = 8          # how many worst discrepancies to print per test


# ═══════════════════════════════════════════════════════════════════════
#  Helpers: parameter construction with generous, truncation-safe bounds
# ═══════════════════════════════════════════════════════════════════════
def make_params(d, N=N_DEFAULT, b1_probe_max=0, I2_probe_max=25):
    """Build Params with bounds wide enough that clipping is negligible."""
    lamT1 = d["lam1"] * d["T"]
    lamT2 = d["lam2"] * d["T"]
    b1_max = int(math.ceil(b1_probe_max + lamT1 + 8.0 * math.sqrt(max(lamT1, 1.0))))
    I2_min = -int(math.ceil(lamT2 + 8.0 * math.sqrt(max(lamT2, 1.0))))
    return Params(
        T=d["T"], N=N,
        lam1=d["lam1"], lam2=d["lam2"],
        h=d["h"], Cf=d["Cf"], cu=d["cu"],
        pi1=d["pi1"], pi2=d["pi2"],
        c1=d.get("c1", 0.0), c2=d.get("c2", 0.0), v2=d.get("v2", 0.0),
        I2_max=I2_probe_max, I2_min=I2_min, b1_max=b1_max,
    )


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report_rows(rows, topk=TOPK, key=lambda r: -abs(r[-1])):
    """Print the worst `topk` discrepancy rows (last field = error)."""
    for r in sorted(rows, key=key)[:topk]:
        print("   " + " | ".join(str(x) for x in r))


def verdict(name, max_err, tol):
    tag = "PASS" if max_err <= tol else "FAIL"
    print(f"  -> [{tag}] {name}: max|err| = {max_err:.6g} (tol = {tol:.3g})")
    return tag == "PASS"


# ═══════════════════════════════════════════════════════════════════════
#  Closed forms from the note
# ═══════════════════════════════════════════════════════════════════════
def V_closed_nonpos(I2, b1, tau, p):
    """Sec 2 closed form, valid for I2 <= 0 (EXACT for the stochastic model)."""
    return (-p.pi2 * I2 * tau + p.pi1 * b1 * tau
            + p.pi1 * p.lam1 * tau**2 / 2.0
            + p.pi2 * p.lam2 * tau**2 / 2.0
            + p.c1 * (b1 + p.lam1 * tau)
            + p.c2 * (p.lam2 * tau - I2))


def V_eq15(b1, tau, p):
    """Eq.(15): boundary at I2 = 0 (EXACT; equals V_closed_nonpos at I2=0)."""
    return (p.pi1 * b1 * tau
            + p.pi1 * p.lam1 * tau**2 / 2.0
            + p.pi2 * p.lam2 * tau**2 / 2.0
            + p.c1 * (b1 + p.lam1 * tau)
            + p.c2 * p.lam2 * tau)


def Vw_fluid(I2, b1, tau, p):
    """Eq.(19): fluid waiting-value, c1=c2=v2=0, I2 >= 0."""
    if I2 >= p.lam2 * tau:
        return (p.pi1 * b1 * tau + p.h * I2 * tau
                + (p.pi1 * p.lam1 - p.h * p.lam2) / 2.0 * tau**2)
    return ((p.h + p.pi2) / (2.0 * p.lam2) * I2**2
            - p.pi2 * I2 * tau + p.pi1 * b1 * tau
            + (p.pi1 * p.lam1 + p.pi2 * p.lam2) / 2.0 * tau**2)


def f_candidate(I2, b1, tau, p):
    """Candidate f for Proposition 1: Vw on I2>=0, exact closed form on I2<0
    (with c1=c2=v2=0).  Continuous at I2=0 by construction."""
    if I2 >= 0:
        return Vw_fluid(I2, b1, tau, p)
    return (-p.pi2 * I2 * tau + p.pi1 * b1 * tau
            + p.pi1 * p.lam1 * tau**2 / 2.0
            + p.pi2 * p.lam2 * tau**2 / 2.0)


# ═══════════════════════════════════════════════════════════════════════
#  Theorem 3 policy, implemented LITERALLY as stated in the note
# ═══════════════════════════════════════════════════════════════════════
def theorem3_policy(I2, b1, tau, p):
    """Return q* per Theorem 3 (0 == wait).  Assumes pi1==pi2, c=0."""
    pi = p.pi1
    if I2 <= 0 or b1 <= 0:
        return 0
    ratio = p.lam2 * p.cu / (p.h + pi)          # lam2*cu/(h+pi)
    m = int(round(ratio))                        # [lam2*cu/(h+pi)], nearest int

    if I2 >= p.lam2 * tau:                       # ── Region 1
        if tau <= p.cu / (pi + p.h):
            return 0
        return min(I2, b1)                       # <- note's claim (suspect)

    # ── Region 2: 1 <= I2 <= lam2*tau
    if I2 <= ratio + 0.5:                        # (i)
        return 0
    if b1 <= I2 - m:                             # (ii) case 1
        X = I2 - ratio
        disc = X * X - 2.0 * p.lam2 * p.Cf / (p.h + pi)
        if disc < 0:
            return 0
        b1_bar = X - math.sqrt(disc)
        return b1 if b1 >= b1_bar else 0
    # (ii) case 2: b1 >= I2 - m + 1
    K = 2.0 * p.lam2 * p.Cf / (p.h + pi) - m * (2.0 * ratio - m)
    I2_bar = ratio + math.sqrt(max(ratio * ratio + K, 0.0))
    if I2 < I2_bar:
        return 0
    return I2 - m


def theorem3_region1_corrected(I2, b1, tau, p):
    """Conjectured fix for Region 1: piecewise-convex objective in q.
    Interior optimum retains ~lam2*cu/(h+pi) units; dispatch only if the
    total variable-cost saving beats Cf."""
    pi = p.pi1
    ratio = p.lam2 * p.cu / (p.h + pi)
    q_v = I2 - ratio                             # unconstrained minimiser
    q = int(min(max(round(q_v), 0), min(I2, b1)))
    if q <= 0:
        return 0
    # exact fluid saving of dispatching q vs waiting (both branches handled
    # by evaluating Vw at pre/post states):
    saving = Vw_fluid(I2, b1, tau, p) - (p.cu * q + Vw_fluid(I2 - q, b1 - q, tau, p))
    return q if saving > p.Cf else 0


# ═══════════════════════════════════════════════════════════════════════
#  Exact no-dispatch DP (vectorised) -- reference for TEST 4 / TEST 5
# ═══════════════════════════════════════════════════════════════════════
def no_dispatch_dp(p):
    """Value of the 'wait forever' policy, all periods.  Shape (N+1, nI2, nb1)."""
    nI2 = p.I2_max - p.I2_min + 1
    nb1 = p.b1_max + 1
    I2v = np.arange(p.I2_min, p.I2_max + 1, dtype=float).reshape(-1, 1)
    b1v = np.arange(nb1, dtype=float).reshape(1, -1)
    flow = (p.h * np.maximum(I2v, 0.0)
            + p.pi2 * np.maximum(-I2v, 0.0)
            + p.pi1 * b1v)
    V = (p.c1 * b1v + p.c2 * np.maximum(-I2v, 0.0)
         - p.v2 * np.maximum(I2v, 0.0)) * np.ones((nI2, nb1))
    out = np.empty((p.N + 1, nI2, nb1))
    out[0] = V
    for n in range(1, p.N + 1):
        Vb = np.concatenate([V[:, 1:], V[:, -1:]], axis=1)   # b1 -> b1+1 (clip)
        Vi = np.vstack([V[0:1, :], V[:-1, :]])               # I2 -> I2-1 (clip)
        V = p.dt * flow + p.p0 * V + p.p1 * Vb + p.p2 * Vi
        out[n] = V
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Parameter sets  (>= 5 per tested result)
# ═══════════════════════════════════════════════════════════════════════
# General sets (pi1 != pi2 allowed, nonzero clean-up costs) -- tests 1, 2, 6.
SETS_GENERAL = [
    dict(T=2.0, lam1=8,  lam2=5,  h=0.10, Cf=20, cu=1.0, pi1=10, pi2=10, c1=5, c2=5, v2=1),
    dict(T=1.0, lam1=3,  lam2=10, h=0.50, Cf=5,  cu=2.0, pi1=12, pi2=4,  c1=8, c2=6, v2=2),
    dict(T=3.0, lam1=12, lam2=2,  h=0.05, Cf=50, cu=0.5, pi1=20, pi2=7,  c1=3, c2=9, v2=0.5),
    dict(T=0.5, lam1=6,  lam2=6,  h=1.00, Cf=10, cu=3.0, pi1=4,  pi2=15, c1=6, c2=6, v2=6),
    dict(T=2.5, lam1=10, lam2=8,  h=0.20, Cf=2,  cu=1.5, pi1=15, pi2=15, c1=0, c2=0, v2=0),
]

# Special-case sets for Theorem 3 / Sec 5 (pi1 == pi2, c1=c2=v2=0) -- tests 3,4,5,7.
SETS_T3 = [
    dict(T=2.0, lam1=8,  lam2=5,  h=0.10, Cf=20, cu=1.0, pi1=10, pi2=10),
    dict(T=1.5, lam1=6,  lam2=10, h=0.50, Cf=10, cu=3.0, pi1=4,  pi2=4),   # ratio~6.7: probe set
    dict(T=2.0, lam1=10, lam2=6,  h=0.30, Cf=5,  cu=2.0, pi1=8,  pi2=8),
    dict(T=1.0, lam1=4,  lam2=12, h=0.20, Cf=15, cu=1.0, pi1=6,  pi2=6),
    dict(T=3.0, lam1=9,  lam2=4,  h=0.80, Cf=8,  cu=2.5, pi1=12, pi2=12),
    dict(T=2.0, lam1=7,  lam2=9,  h=0.10, Cf=40, cu=0.8, pi1=20, pi2=20),  # large Cf: probe 7b
]

# Probe states appended in tests 7a/7b for the two suspected Region-1 flaws.
PROBE_SET_IDX = 1        # SETS_T3[1]: lam2*cu/(h+pi) = 30/4.5 ~ 6.67, m = 7


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1 & 2 -- exact closed forms vs DP (with dt-convergence check)
# ═══════════════════════════════════════════════════════════════════════
def test_closed_forms():
    header("TEST 1 & 2: closed-form V(I2<=0) and Eq.(15) vs DP  [EXACT results]")
    all_pass = True
    for si, d in enumerate(SETS_GENERAL):
        errs_by_N = {}
        for N in ([N_DEFAULT] if FAST else [N_DEFAULT, 2 * N_DEFAULT]):
            p = make_params(d, N=N, b1_probe_max=12, I2_probe_max=20)
            dp = TransshipmentDP(p)
            dp.solve(store_V=True, verbose=False)
            rows, max_err = [], 0.0
            n_samples = sorted({p.N, p.N // 2, p.N // 4, max(p.N // 10, 1)})
            for n in n_samples:
                tau = n * p.dt
                # TEST 1 grid: I2 in {-8..0}, b1 in {0, 3, 8, 12}
                for I2 in range(-8, 1):
                    for b1 in (0, 3, 8, 12):
                        v_dp = dp.get_value(n, I2, b1)
                        v_cf = V_closed_nonpos(I2, b1, tau, p)
                        e = v_dp - v_cf
                        max_err = max(max_err, abs(e))
                        if abs(e) > 0:
                            rows.append((f"set{si}", f"n={n}", f"tau={tau:.3f}",
                                         f"I2={I2}", f"b1={b1}", round(e, 6)))
                # TEST 2: Eq.(15) at I2 = 0
                for b1 in (0, 3, 8, 12):
                    e15 = dp.get_value(n, 0, b1) - V_eq15(b1, tau, p)
                    max_err = max(max_err, abs(e15))
            errs_by_N[N] = max_err
            print(f" set {si}: N={N:4d}  max|DP - closed form| = {max_err:.6g}")
            report_rows(rows, topk=3)
        # O(dt) convergence: doubling N should roughly halve the error
        if len(errs_by_N) == 2:
            e1, e2 = errs_by_N[N_DEFAULT], errs_by_N[2 * N_DEFAULT]
            ratio = e1 / max(e2, 1e-15)
            print(f" set {si}: convergence ratio err(N)/err(2N) = {ratio:.2f} "
                  f"(expect ~2 if the gap is pure O(dt) discretisation)")
        # tolerance scales with dt and cost magnitudes
        p0 = make_params(d, N=N_DEFAULT)
        tol = 5.0 * (d["pi1"] + d["pi2"]) * (d["lam1"] + d["lam2"]) * d["T"] * p0.dt
        all_pass &= verdict(f"set {si} closed forms", errs_by_N[N_DEFAULT], tol)
    return all_pass


# ═══════════════════════════════════════════════════════════════════════
#  TEST 3 -- Vw satisfies PDE (16) and value-matching at I2 = lam2*tau
# ═══════════════════════════════════════════════════════════════════════
def test_pde_verification():
    header("TEST 3: Eq.(19) Vw solves PDE (16); continuity at I2 = lam2*tau")
    eps = 1e-5
    all_pass = True
    for si, d in enumerate(SETS_T3[:5]):
        p = make_params(d)
        rows, max_res = [], 0.0
        rng = np.random.default_rng(42 + si)
        for _ in range(200):
            tau = float(rng.uniform(0.05, d["T"]))
            b1 = float(rng.uniform(0, 20))
            # sample both branches, away from the kink
            for I2 in (0.3 * p.lam2 * tau, 1.7 * p.lam2 * tau + 1.0):
                dV_tau = (Vw_fluid(I2, b1, tau + eps, p)
                          - Vw_fluid(I2, b1, tau - eps, p)) / (2 * eps)
                dV_b1 = (Vw_fluid(I2, b1 + eps, tau, p)
                         - Vw_fluid(I2, b1 - eps, tau, p)) / (2 * eps)
                dV_I2 = (Vw_fluid(I2 + eps, b1, tau, p)
                         - Vw_fluid(I2 - eps, b1, tau, p)) / (2 * eps)
                rhs = (p.h * I2 + p.pi1 * b1
                       + p.lam1 * dV_b1 - p.lam2 * dV_I2)
                res = dV_tau - rhs
                max_res = max(max_res, abs(res))
                if abs(res) > 1e-4:
                    rows.append((f"set{si}", f"tau={tau:.3f}", f"I2={I2:.2f}",
                                 f"b1={b1:.1f}", round(res, 8)))
            # value matching at the kink
            I2k = p.lam2 * tau
            jump = (Vw_fluid(I2k - 1e-9, b1, tau, p)
                    - Vw_fluid(I2k + 1e-9, b1, tau, p))
            max_res = max(max_res, abs(jump))
        report_rows(rows)
        all_pass &= verdict(f"set {si} PDE residual", max_res, 1e-3)
    print(" NOTE: this test only certifies the note's OWN algebra: Vw solves the")
    print(" PDE approximation (16), NOT the original difference equation (14).")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4 -- fluid gap: Vw vs exact no-dispatch DP
# ═══════════════════════════════════════════════════════════════════════
def test_fluid_gap():
    header("TEST 4: fluid approximation error  Vw (Eq.19)  vs  exact no-dispatch DP")
    print(" Expected: nonzero gap (Vw is a first-order Taylor / fluid surrogate).")
    print(" The gap is the mechanism behind the Theorem 3 threshold bias in TEST 7.")
    for si, d in enumerate(SETS_T3[:5]):
        p = make_params(d, b1_probe_max=15, I2_probe_max=25)
        Vnd = no_dispatch_dp(p)
        rows, max_rel = [], 0.0
        for n in sorted({p.N, p.N // 2, p.N // 4}):
            tau = n * p.dt
            for I2 in range(0, 21, 4):
                for b1 in (0, 5, 15):
                    v_dp = float(Vnd[n, I2 - p.I2_min, b1])
                    v_fl = Vw_fluid(I2, b1, tau, p)
                    gap = v_fl - v_dp
                    rel = abs(gap) / max(abs(v_dp), 1.0)
                    max_rel = max(max_rel, rel)
                    rows.append((f"set{si}", f"tau={tau:.2f}", f"I2={I2}",
                                 f"b1={b1}", f"DP={v_dp:.3f}",
                                 f"Vw={v_fl:.3f}", round(gap, 4)))
        print(f" set {si}: max relative fluid gap = {100 * max_rel:.2f}%  "
              f"(worst states below; sign of gap = Vw - DP)")
        report_rows(rows)
    return True   # informational: no pass/fail, quantifies the approximation


# ═══════════════════════════════════════════════════════════════════════
#  TEST 5 -- Proposition 1: lower-bound conditions and conclusion
# ═══════════════════════════════════════════════════════════════════════
def test_proposition1():
    header("TEST 5: Proposition 1 -- does f (= Vw glued to exact I2<0 form) "
           "satisfy (8), and is f <= V_DP?")
    eps = 1e-5
    for si, d in enumerate(SETS_T3[:5]):
        p = make_params(d, b1_probe_max=15, I2_probe_max=20)
        dp = TransshipmentDP(p)
        dp.solve(store_V=True, verbose=False)
        viol_a, viol_b, gap_rows = [], [], []
        min_gap = np.inf
        for n in sorted({p.N, p.N // 2}):
            tau = n * p.dt
            for I2 in range(-5, 18, 2):
                for b1 in range(0, 16, 3):
                    f0 = f_candidate(I2, b1, tau, p)
                    # (8a): difference-form generator inequality
                    df_tau = (f_candidate(I2, b1, tau + eps, p)
                              - f_candidate(I2, b1, tau - eps, p)) / (2 * eps)
                    rhs = (p.h * max(I2, 0) + p.pi2 * max(-I2, 0) + p.pi1 * b1
                           + p.lam1 * (f_candidate(I2, b1 + 1, tau, p) - f0)
                           - p.lam2 * (f0 - f_candidate(I2 - 1, b1, tau, p)))
                    if df_tau - rhs > 1e-6:
                        viol_a.append((f"set{si}", f"tau={tau:.2f}", f"I2={I2}",
                                       f"b1={b1}", round(df_tau - rhs, 5)))
                    # (8b): dispatch inequality
                    if I2 > 0 and b1 > 0:
                        best = min(p.Cf + p.cu * q
                                   + f_candidate(I2 - q, b1 - q, tau, p)
                                   for q in range(1, min(I2, b1) + 1))
                        if f0 - best > 1e-9:
                            viol_b.append((f"set{si}", f"tau={tau:.2f}",
                                           f"I2={I2}", f"b1={b1}",
                                           round(f0 - best, 5)))
                    # conclusion: f <= V_DP  (up to O(dt))
                    g = dp.get_value(n, I2, b1) - f0
                    min_gap = min(min_gap, g)
                    if g < 0:
                        gap_rows.append((f"set{si}", f"tau={tau:.2f}",
                                         f"I2={I2}", f"b1={b1}", round(g, 5)))
        print(f" set {si}: (8a) violations: {len(viol_a)}   "
              f"(8b) violations: {len(viol_b)}   min(V_DP - f) = {min_gap:.5f}")
        if viol_a:
            print("   (8a) worst -- EXPECTED where fluid Vw fails the discrete "
                  "generator (documents the Taylor-approx gap):")
            report_rows(viol_a)
        if viol_b:
            print("   (8b) worst:")
            report_rows(viol_b)
        if gap_rows:
            print("   f > V_DP at (only meaningful where (8a),(8b) hold!):")
            report_rows(gap_rows)
        print("   Interpretation: Prop 1 is conditional. If (8) is violated "
              "somewhere, f is simply not a certified lower bound there; "
              "f > V_DP at a state where (8) HOLDS on the whole reachable set "
              "would contradict Prop 1 itself.")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  TEST 5b -- lower ENVELOPE candidate:  f2 = min(Vw, one-shot dispatch)
# ═══════════════════════════════════════════════════════════════════════
def f_envelope(I2, b1, tau, p):
    """f2(I2,b1,tau) = min( Vw(I2,b1,tau),
                            min_{1<=q<=min(I2,b1)} Cf + cu*q + Vw(I2-q,b1-q,tau) )
    for I2 >= 0, glued to the exact I2 < 0 closed form (c1=c2=v2=0).

    Key structural facts (why f2 is a better Prop-1 candidate than Vw):
      (8b) holds BY CONSTRUCTION: dispatching q then q' from the envelope
           costs 2*Cf + cu*(q+q') + Vw(...), which the one-shot dispatch of
           q+q' (feasible, since q+q' <= min(I2,b1)) beats by Cf > 0.
      (8c) holds: Vw(.,0) = 0 = terminal, and the dispatch branch adds
           Cf + cu*q > 0, so the min at tau = 0 is 0.
      (8a) is the only condition left to audit numerically."""
    if I2 < 0:
        return f_candidate(I2, b1, tau, p)
    best = Vw_fluid(I2, b1, tau, p)
    if I2 > 0 and b1 > 0:
        for q in range(1, min(I2, b1) + 1):
            best = min(best,
                       p.Cf + p.cu * q + Vw_fluid(I2 - q, b1 - q, tau, p))
    return best


def test_proposition1_envelope():
    header("TEST 5b: Proposition 1 with f2 = min(Vw, dispatch envelope) -- "
           "certifiable analytic lower bound?")
    print(" (8b) and (8c) hold by construction (see f_envelope docstring);")
    print(" (8a) is audited below. f2 is a min of smooth branches, so at")
    print(" branch-switch kinks the central tau-difference smears: a state is")
    print(" flagged only if BOTH one-sided derivatives violate (8a).")
    eps = 1e-5
    for si, d in enumerate(SETS_T3[:5]):
        p = make_params(d, b1_probe_max=15, I2_probe_max=20)
        dp = TransshipmentDP(p)
        dp.solve(store_V=True, verbose=False)
        viol_a, viol_b, neg_rows = [], [], []
        min_gap, max_rel_gap, n_states = np.inf, 0.0, 0
        for n in sorted({p.N, p.N // 2}):
            tau = n * p.dt
            for I2 in range(-5, 18, 2):
                for b1 in range(0, 16, 3):
                    n_states += 1
                    f0 = f_envelope(I2, b1, tau, p)
                    # ── (8a): both one-sided tau-derivatives must violate ──
                    d_plus = (f_envelope(I2, b1, tau + eps, p) - f0) / eps
                    d_minus = (f0 - f_envelope(I2, b1, tau - eps, p)) / eps
                    rhs = (p.h * max(I2, 0) + p.pi2 * max(-I2, 0) + p.pi1 * b1
                           + p.lam1 * (f_envelope(I2, b1 + 1, tau, p) - f0)
                           - p.lam2 * (f0 - f_envelope(I2 - 1, b1, tau, p)))
                    excess = min(d_plus, d_minus) - rhs
                    if excess > 1e-6:
                        viol_a.append((f"set{si}", f"tau={tau:.2f}", f"I2={I2}",
                                       f"b1={b1}", round(excess, 5)))
                    # ── (8b): should be empty; confirms the closure argument ──
                    if I2 > 0 and b1 > 0:
                        best = min(p.Cf + p.cu * q
                                   + f_envelope(I2 - q, b1 - q, tau, p)
                                   for q in range(1, min(I2, b1) + 1))
                        if f0 - best > 1e-9:
                            viol_b.append((f"set{si}", f"tau={tau:.2f}",
                                           f"I2={I2}", f"b1={b1}",
                                           round(f0 - best, 5)))
                    # ── conclusion: f2 <= V_DP up to O(dt) ──
                    v_dp = dp.get_value(n, I2, b1)
                    g = v_dp - f0
                    min_gap = min(min_gap, g)
                    max_rel_gap = max(max_rel_gap, g / max(abs(v_dp), 1.0))
                    if g < 0:
                        neg_rows.append((f"set{si}", f"tau={tau:.2f}",
                                         f"I2={I2}", f"b1={b1}", round(g, 5)))
        # DP itself carries an O(dt) downward bias (TEST 1), so allow a small
        # negative min_gap of that magnitude before declaring failure.
        tol_dt = 5.0 * (p.pi1 + p.pi2) * (p.lam1 + p.lam2) * p.T * p.dt
        print(f" set {si}: (8a) violations: {len(viol_a)}   "
              f"(8b) violations: {len(viol_b)}   states: {n_states}")
        print(f" set {si}: min(V_DP - f2) = {min_gap:.5f}  "
              f"(O(dt) allowance = {tol_dt:.3f}),  "
              f"max relative slack of the bound = {100 * max_rel_gap:.2f}%")
        if viol_a:
            print("   (8a) worst (these states DE-CERTIFY the bound; check "
                  "whether they cluster at the envelope kink or in Region 1):")
            report_rows(viol_a)
        if viol_b:
            print("   (8b) worst (unexpected -- contradicts the closure "
                  "argument, investigate):")
            report_rows(viol_b)
        if neg_rows:
            print("   f2 > V_DP states:")
            report_rows(neg_rows)
        certified = (len(viol_a) == 0 and len(viol_b) == 0
                     and min_gap > -tol_dt)
        print(f"  -> {'CERTIFIED' if certified else 'NOT certified'}: "
              f"f2 {'is' if certified else 'is NOT'} a Prop-1 analytic lower "
              f"bound on this grid for set {si}.")
    print(" NOTE: certification here is grid-level numerical evidence; for the")
    print(" paper, (8a) for f2 still needs a short analytic proof per branch.")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  TEST 6 -- Theorem 2: threshold (up-set) structure of the DP policy
# ═══════════════════════════════════════════════════════════════════════
def _q_values(dp, n, I2, b1):
    """Recompute the Q-value of every feasible action at state (n, I2, b1)
    from the stored V_all[n-1].  Requires solve(store_V=True)."""
    p = dp.p
    Vprev = dp.V_all[n - 1]
    q_max = max(0, min(I2, b1)) if (I2 > 0 and b1 > 0) else 0
    out = {}
    for q in range(q_max + 1):
        I2a, b1a = I2 - q, b1 - q
        ii0 = dp._ii(dp._clip_I2(I2a))
        jj0 = dp._clip_b1(b1a)
        ii2 = dp._ii(dp._clip_I2(I2a - 1))
        jj1 = dp._clip_b1(b1a + 1)
        out[q] = (dp.g(I2, b1, q)
                  + p.p0 * Vprev[ii0, jj0]
                  + p.p1 * Vprev[ii0, jj1]
                  + p.p2 * Vprev[ii2, jj0])
    return out


def test_threshold_structure(N=N_DEFAULT):
    header(f"TEST 6: Theorem 2 -- for each (n, I2), {{b1 : q* > 0}} is an "
           f"up-set  [N={N}]")
    all_pass = True
    for si, d in enumerate(SETS_GENERAL):
        p = make_params(d, N=N, b1_probe_max=15, I2_probe_max=25)
        dp = TransshipmentDP(p)
        dp.solve(store_V=True, verbose=False)
        viol, viol_states = [], []
        n_samples = sorted({p.N, 3 * p.N // 4, p.N // 2, p.N // 4})
        for n in n_samples:
            for ii in range(dp._nI2):
                I2 = dp._I2(ii)
                if I2 <= 0:
                    continue
                dispatch = dp.policy[n, ii, :] > 0
                # once dispatching starts (in b1), it must not stop again
                started = False
                for b1 in range(min(p.b1_max, I2 + 25) + 1):
                    if dispatch[b1]:
                        started = True
                    elif started:
                        viol.append((f"set{si}", f"n={n}", f"I2={I2}",
                                     f"b1={b1}", "dispatch OFF above threshold"))
                        viol_states.append((n, I2, b1))
        print(f" set {si}: up-set violations = {len(viol)}")
        report_rows(viol, key=lambda r: 0)
        # ── DIAGNOSTIC: tie-band artifact vs genuine non-threshold policy ──
        # At each violating state, gap = Q(wait) - min_{q>0} Q(q).  The state
        # was labelled 'wait', so gap <= 0.  |gap| ~ 1e-6-or-less means the
        # DP is numerically indifferent there (argmin tie-breaking artifact);
        # a clearly nonzero |gap| means the dispatch region is GENUINELY not
        # an up-set in b1, i.e. Theorem 2's assumed structure fails for these
        # parameters -- a real counterexample worth keeping.
        if viol_states:
            print("   diagnostic: Q(wait) - best dispatch Q at violating states")
            print("   (near-zero => tie band / discretisation; O(1) => genuine)")
            for (n, I2, b1) in viol_states[:TOPK]:
                qv = _q_values(dp, n, I2, b1)
                best_disp = min(v for q, v in qv.items() if q > 0)
                gap = qv[0] - best_disp
                print(f"    n={n} I2={I2} b1={b1}:  gap = {gap:+.3e}  "
                      f"(argmin over q>0 at q={min((v, q) for q, v in qv.items() if q > 0)[1]})")
        all_pass &= verdict(f"set {si} threshold structure", float(len(viol)), 0.0)
    print(" NOTE: if violations persist with |gap| >> dt at 2N, they are real:")
    print(" rerun via test_threshold_structure(N=2*N_DEFAULT) to confirm.")
    print(" NOTE: also check (not covered by Thm 2's statement) whether q* maps")
    print(" the state back inside C -- the note asserts this without proof.")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════
#  TEST 7 -- Theorem 3 policy vs DP policy (+ probes 7a, 7b)
# ═══════════════════════════════════════════════════════════════════════
def test_theorem3():
    header("TEST 7: Theorem 3 (pi1=pi2, c1=c2=v2=0) vs DP optimal policy")
    for si, d in enumerate(SETS_T3):
        p = make_params(d, b1_probe_max=30, I2_probe_max=25)
        dp = TransshipmentDP(p)
        dp.solve(verbose=False)
        rows = []
        n_dec_mismatch = n_q_mismatch = n_total = 0
        n_dec_boundary = 0
        # ── DYNAMIC time sampling ──────────────────────────────────────
        # Theorem 3's structure hinges on two tau-vs-I2 ratios that differ
        # per parameter set:
        #   (a) tau* = cu/(pi+h)      Region-1 wait/dispatch switch;
        #   (b) tau  = I2/lam2        Region-1 <-> Region-2 boundary,
        #                             swept across the I2 probe grid.
        # Fixed fractions of the horizon can miss (a) entirely or leave one
        # side of (b) unsampled, so tau targets are derived from the model:
        tau_star = p.cu / (p.pi1 + p.h)
        I2_hi = 23                                 # top of the I2 probe grid
        tau_targets = {0.5 * tau_star, 1.5 * tau_star, 3.0 * tau_star,
                       0.25 * I2_hi / p.lam2, 0.5 * I2_hi / p.lam2,
                       1.0 * I2_hi / p.lam2,
                       0.5 * p.T, p.T}
        n_samples = sorted({min(max(int(round(t / p.dt)), 1), p.N)
                            for t in tau_targets})
        for n in n_samples:
            tau = n * p.dt
            for I2 in range(1, I2_hi + 1, 2):
                for b1 in range(1, 28, 3):
                    q_dp = dp.get_policy(n, I2, b1)
                    q_t3 = theorem3_policy(I2, b1, tau, p)
                    n_total += 1
                    # region membership recomputed per (n, I2): tau = n*dt
                    region = 1 if I2 >= p.lam2 * tau else 2
                    # near the kink I2 ~ lam2*tau the discrete (integer I2,
                    # gridded tau) region label is itself dt-sensitive:
                    # tag these so genuine theorem errors are not conflated
                    # with discretisation artifacts.
                    band = "~B" if abs(I2 - p.lam2 * tau) <= 1.0 else "  "
                    if (q_dp > 0) != (q_t3 > 0):
                        n_dec_mismatch += 1
                        if band == "~B":
                            n_dec_boundary += 1
                        rows.append((f"set{si}", f"tau={tau:.2f}",
                                     f"R{region}{band}",
                                     f"I2={I2}", f"b1={b1}",
                                     f"DP q*={q_dp}", f"T3 q*={q_t3}",
                                     abs(q_dp - q_t3)))
                    elif abs(q_dp - q_t3) > 1:      # tolerate +-1 rounding
                        n_q_mismatch += 1
                        rows.append((f"set{si}", f"tau={tau:.2f}",
                                     f"R{region}{band}",
                                     f"I2={I2}", f"b1={b1}",
                                     f"DP q*={q_dp}", f"T3 q*={q_t3}",
                                     abs(q_dp - q_t3)))
        print(f" set {si}: tau sampled = "
              f"{[round(n * p.dt, 3) for n in n_samples]}  "
              f"(tau* = cu/(pi+h) = {tau_star:.3f})")
        print(f" set {si}: states checked = {n_total}, "
              f"decision mismatches = {n_dec_mismatch} "
              f"(of which {n_dec_boundary} in the |I2 - lam2*tau| <= 1 band), "
              f"quantity mismatches (|dq| > 1) = {n_q_mismatch}")
        print("   READ-OUT: interior mismatches indict the theorem; "
              "band ('~B') mismatches may be region-label artifacts -- "
              "re-check those at 2N before counting them.")
        report_rows(rows)

    # ── 7a: Region 1 branch-crossing probe ─────────────────────────────
    header("TEST 7a probe: Region 1 claim 'q* = min(I2, b1)' vs retained "
           "safety stock lam2*cu/(h+pi)")
    d = SETS_T3[PROBE_SET_IDX]                    # ratio ~ 6.67, m = 7
    p = make_params(d, b1_probe_max=30, I2_probe_max=25)
    dp = TransshipmentDP(p)
    dp.solve(verbose=False)
    ratio = p.lam2 * p.cu / (p.h + p.pi1)
    print(f" params: lam2*cu/(h+pi) = {ratio:.3f}  (Theorem-3 Region 1 predicts "
          f"retained stock 0; corrected analysis predicts ~{round(ratio)})")
    # sweep several tau above cu/(pi+h); the region-1 I2 grid is re-derived
    # from lam2*tau at EACH tau (the tau/I2 ratio moves with the sample)
    tau_star = p.cu / (p.pi1 + p.h)
    for factor in (1.2, 1.4, 2.0):
        n = min(max(int(math.ceil(factor * tau_star / p.dt)), 1), p.N)
        tau = n * p.dt
        I2_lo = int(math.ceil(p.lam2 * tau)) + 2   # strictly inside Region 1
        if I2_lo >= p.I2_max - 1:
            print(f" tau = {tau:.3f}: lam2*tau = {p.lam2 * tau:.2f} exceeds the "
                  f"I2 grid; skipping this factor.")
            continue
        print(f"\n tau = {tau:.3f}  (= {factor:.1f} x cu/(pi+h) = "
              f"{factor:.1f} x {tau_star:.3f}),  Region 1 requires "
              f"I2 >= lam2*tau = {p.lam2 * tau:.2f}")
        print(f" {'I2':>4} {'b1':>4} | {'DP q*':>6} {'retained':>9} | "
              f"{'T3 q*':>6} {'T3 retained':>12} | {'corrected q*':>13}")
        for I2 in range(I2_lo, 24, 2):
            for b1 in (I2, I2 + 10):
                q_dp = dp.get_policy(n, I2, b1)
                q_t3 = theorem3_policy(I2, b1, tau, p)
                q_cx = theorem3_region1_corrected(I2, b1, tau, p)
                print(f" {I2:>4} {b1:>4} | {q_dp:>6} {I2 - q_dp:>9} | "
                      f"{q_t3:>6} {I2 - q_t3:>12} | {q_cx:>13}")
    print(" READ-OUT: if DP's retained stock clusters near "
          f"{round(ratio)} rather than 0 while Theorem 3 dispatches everything, "
          "the Eq.(23) branch-crossing flaw is confirmed.")

    # ── 7b: Region 1 fixed-cost probe ───────────────────────────────────
    header("TEST 7b probe: Region 1 dispatch test omits Cf")
    d = SETS_T3[5]                                # large Cf = 40
    p = make_params(d, b1_probe_max=20, I2_probe_max=25)
    dp = TransshipmentDP(p)
    dp.solve(verbose=False)
    tau_star = p.cu / (p.pi1 + p.h)
    n = min(int(math.ceil(1.1 * tau_star / p.dt)), p.N)
    tau = n * p.dt
    gain_pu = (p.pi1 + p.h) * tau - p.cu
    print(f" tau = {tau:.4f} slightly above cu/(pi+h) = {tau_star:.4f}; "
          f"variable-cost gain per unit = {gain_pu:.4f}, Cf = {p.Cf}")
    print(f" {'I2':>4} {'b1':>4} | {'DP q*':>6} | {'T3 q*':>6} | "
          f"{'gain*q vs Cf':>14}")
    for I2 in range(int(math.ceil(p.lam2 * tau)) + 1, 22, 3):
        for b1 in (1, 2, 4):
            q_dp = dp.get_policy(n, I2, b1)
            q_t3 = theorem3_policy(I2, b1, tau, p)
            g = gain_pu * min(I2, b1)
            print(f" {I2:>4} {b1:>4} | {q_dp:>6} | {q_t3:>6} | "
                  f"{g:.3f} vs {p.Cf}")
    print(" READ-OUT: whenever gain*q << Cf the DP should WAIT (q*=0) while "
          "Theorem 3 Region 1 says dispatch min(I2,b1): each such row is a "
          "counterexample to the theorem as stated.")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  TEST 8 -- solver.py <-> note consistency unit checks
# ═══════════════════════════════════════════════════════════════════════
def test_solver_consistency():
    header("TEST 8: solver.py vs note -- structural consistency")
    ok = True
    for si, d in enumerate(SETS_GENERAL):
        p = make_params(d, N=50)
        dp = TransshipmentDP(p)
        # (a) transition probabilities sum to 1
        ok &= abs(p.p0 + p.p1 + p.p2 - 1.0) < 1e-12
        # (b) terminal matches C_terminal of the note
        for I2 in (-4, 0, 7):
            for b1 in (0, 5):
                t_note = p.c1 * b1 + p.c2 * max(0, -I2) - p.v2 * max(0, I2)
                ok &= abs(dp.terminal(I2, b1) - t_note) < 1e-12
        # (c) g decomposition: g = Cf*1{q>0} + cu*q + dt*flow(post-dispatch)
        for (I2, b1, q) in [(10, 6, 0), (10, 6, 3), (10, 6, 6), (2, 9, 2)]:
            I2a, b1a = I2 - q, b1 - q
            flow = (p.h * max(0, I2a) + p.pi1 * b1a + p.pi2 * max(0, -I2a))
            g_note = (p.Cf if q > 0 else 0.0) + p.cu * q + p.dt * flow
            ok &= abs(dp.g(I2, b1, q) - g_note) < 1e-12
        print(f" set {si}: probabilities / terminal / g decomposition consistent: {ok}")
    print(" Reminders (manual, not assertable):")
    print("  * solver bundles dispatch with one dt-step (single dispatch per")
    print("    period) vs the note's instantaneous dispatches -- vanishes as dt->0.")
    print("  * Theorem 3 comparisons REQUIRE c1=c2=v2=0; solver defaults 5,5,1.")
    print("  * Both b1_max AND I2_min must be generous; this script sets")
    print("    I2_min <= -(lam2*T + 8*sqrt(lam2*T)) automatically.")
    return ok


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"validate_pdf.py  (FAST={FAST}, N_DEFAULT={N_DEFAULT})")
    print("WARNING: TransshipmentDP is a pure-python triple loop; the full run")
    print("takes tens of minutes. Use FAST=1 for a smoke test first.\n")
    results = {}
    results["T1+T2 exact closed forms"] = test_closed_forms()
    results["T3 PDE verification"] = test_pde_verification()
    results["T4 fluid gap (info)"] = test_fluid_gap()
    results["T5 Proposition 1 (info)"] = test_proposition1()
    results["T5b envelope lower bound"] = test_proposition1_envelope()
    results["T6 Theorem 2 structure"] = test_threshold_structure()
    results["T7 Theorem 3 (+probes)"] = test_theorem3()
    results["T8 solver consistency"] = test_solver_consistency()

    header("SUMMARY")
    for k, v in results.items():
        print(f"  {'PASS/OK' if v else 'FAIL':>7}  {k}")