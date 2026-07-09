"""
threshold_algorithm.py
===============================================================================
General-case (pi1 != pi2, arbitrary Cf) dispatch algorithm from the corrected
Theorem 3, plus threshold-curve extraction and self-tests.

FAST mode: O(1) closed-form decision (vertex + rounded neighbours).
SAFE mode: brute-force argmax over all feasible q; concavity of Delta
guarantees FAST == SAFE, and the self-test asserts it.

Validity: fluid rule. Symmetric case validated against the exact DP
(decision exact, quantity within one unit at all probe states).
Asymmetric case is analytic and NOT yet DP-validated.

Run:  python threshold_algorithm.py
Optional: if solver.py (TransshipmentDP) is importable, a DP cross-check
runs on the symmetric instance S1.
"""

import math
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
#  Core algorithm
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class ModelParams:
    lam1: float
    lam2: float
    h: float
    Cf: float
    cu: float
    pi1: float
    pi2: float
    name: str = ""

    @property
    def a(self):
        return (self.h + self.pi2) / (2.0 * self.lam2)

    @property
    def tau1_star(self):
        return self.cu / (self.pi1 + self.h)

    def S_star(self, tau):
        """Retention target S*(tau); tau-dependent unless pi1 == pi2."""
        return self.lam2 * (self.cu - (self.pi1 - self.pi2) * tau) / (self.h + self.pi2)


def delta_saving(q, I2, tau, p: ModelParams):
    """Fluid saving Delta(q) of dispatching q vs waiting. Valid on both
    branches of Vw via the two positive-part correction terms."""
    up0 = max(I2 - p.lam2 * tau, 0.0)          # pre-dispatch upper-branch excess
    up1 = max(I2 - q - p.lam2 * tau, 0.0)      # post-dispatch upper-branch excess
    S = p.S_star(tau)
    return p.a * (2.0 * (I2 - S) * q - q * q - up0 * up0 + up1 * up1)


def optimal_dispatch(I2, b1, tau, p: ModelParams, mode="fast"):
    """Return (q_star, saving). q_star = 0 means wait."""
    if I2 <= 0 or b1 <= 0:
        return 0, 0.0
    q_cap = min(I2, b1)
    if mode == "safe":                          # brute force, for self-tests
        best_q, best_d = 0, 0.0
        for q in range(1, q_cap + 1):
            d = delta_saving(q, I2, tau, p)
            if d > best_d:
                best_q, best_d = q, d
        return (best_q, best_d) if (best_q >= 1 and best_d >= p.Cf) else (0, best_d)
    # FAST: concavity => integer optimum among rounded neighbours of vertex
    q_v = I2 - p.S_star(tau)
    cands = {int(math.floor(q_v)), int(math.ceil(q_v))}
    cands = {min(max(q, 0), q_cap) for q in cands}
    q0, d0 = 0, 0.0
    for q in cands:
        if q >= 1:
            d = delta_saving(q, I2, tau, p)
            if d > d0:
                q0, d0 = q, d
    return (q0, d0) if (q0 >= 1 and d0 >= p.Cf) else (0, d0)


# ═══════════════════════════════════════════════════════════════════════
#  Threshold curves
# ═══════════════════════════════════════════════════════════════════════
def I2_bar(tau, p: ModelParams):
    """Participation threshold with ample backlog; math.inf if dispatch is
    never profitable at this tau.
    LEMMA: for tau <= tau1* no dispatch is ever profitable, because
    S*(tau) - lam2*tau is strictly decreasing in tau and equals zero at
    tau1*, so S* > lam2*tau, the slope of Delta at q = 0 is nonpositive,
    and concavity makes Delta <= 0 for every q. Holds for pi1 != pi2 too.
    Closed form on the pure lower branch; numeric inversion otherwise."""
    if tau <= p.tau1_star:
        return math.inf
    S = p.S_star(tau)
    root = S + math.sqrt(2.0 * p.lam2 * p.Cf / (p.h + p.pi2))
    if root <= p.lam2 * tau:                    # closed form is self-consistent
        return root
    lo, hi = p.lam2 * tau, p.lam2 * tau + 10.0 * (root + 1.0)   # numeric fallback
    q_hi = max(hi - S, 0.0)
    if delta_saving(q_hi, hi, tau, p) < p.Cf:   # no finite threshold in bracket
        return math.inf
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        q_v = max(mid - S, 0.0)
        if delta_saving(q_v, mid, tau, p) >= p.Cf:
            hi = mid
        else:
            lo = mid
    return hi


def b1_bar(I2, tau, p: ModelParams):
    """Smallest integer backlog at which dispatch pays; None if never."""
    for b1 in range(1, int(I2) + 1):            # cap: q <= min(I2, b1)
        q, d = optimal_dispatch(I2, b1, tau, p)
        if q >= 1:
            return b1
    q, d = optimal_dispatch(I2, 10 * int(I2) + 50, tau, p)      # ample backlog
    return None if q == 0 else int(I2)


# ═══════════════════════════════════════════════════════════════════════
#  Test instances
# ═══════════════════════════════════════════════════════════════════════
S1 = ModelParams(lam1=6, lam2=10, h=0.5, Cf=10, cu=3.0, pi1=4, pi2=4, name="S1 symmetric (DP-validated probes)")
S5 = ModelParams(lam1=7, lam2=9, h=0.1, Cf=40, cu=0.8, pi1=20, pi2=20, name="S5 symmetric (fixed-cost probe)")
A1 = ModelParams(lam1=6, lam2=10, h=0.5, Cf=10, cu=3.0, pi1=6, pi2=3, name="A1 asymmetric pi1 > pi2 (unvalidated)")
A2 = ModelParams(lam1=6, lam2=10, h=0.5, Cf=10, cu=3.0, pi1=3, pi2=8, name="A2 asymmetric pi1 < pi2 (unvalidated)")


def run_instance(p: ModelParams, states, taus_for_curves):
    print("=" * 74)
    print(p.name)
    print(f"  tau1* = {p.tau1_star:.4f}   a = {p.a:.4f}   "
          f"S*(0.5) = {p.S_star(0.5):.3f}   S*(1.0) = {p.S_star(1.0):.3f}")
    print(f"  {'I2':>4} {'b1':>4} {'tau':>6} | {'q*':>4} {'Delta':>8} | vs Cf")
    for (I2, b1, tau) in states:
        q, d = optimal_dispatch(I2, b1, tau, p)
        d_str = f"{d:8.3f}" if d > 0 else "  <= 0  "
        print(f"  {I2:>4} {b1:>4} {tau:>6.3f} | {q:>4} {d_str} | "
              f"{'dispatch' if q else 'wait'} (Cf = {p.Cf})")
    print("  participation threshold I2_bar(tau)  "
          "[inf = dispatch never profitable at this tau]:")
    for tau in taus_for_curves:
        v = I2_bar(tau, p)
        v_str = "    inf" if math.isinf(v) else f"{v:7.3f}"
        print(f"    tau = {tau:.3f}:  I2_bar = {v_str}   "
              f"S* = {p.S_star(tau):6.3f}   "
              f"{'(tau <= tau1*)' if tau <= p.tau1_star else ''}")
    tau0 = taus_for_curves[-1]
    print(f"  backlog threshold b1_bar(I2, tau = {tau0:.3f}):")
    for I2 in (8, 12, 16, 20):
        bb = b1_bar(I2, tau0, p)
        print(f"    I2 = {I2}:  b1_bar = {bb}")


def self_test():
    """FAST vs SAFE agreement on a random grid; concavity makes them equal."""
    import random
    rng = random.Random(0)
    n_checked = 0
    for p in (S1, S5, A1, A2):
        for _ in range(400):
            I2 = rng.randint(1, 30)
            b1 = rng.randint(1, 35)
            tau = rng.uniform(0.02, 1.5)
            qf, df = optimal_dispatch(I2, b1, tau, p, mode="fast")
            qs, ds = optimal_dispatch(I2, b1, tau, p, mode="safe")
            assert (qf > 0) == (qs > 0), (p.name, I2, b1, tau, qf, qs)
            if qf > 0:
                assert abs(df - ds) < 1e-9, (p.name, I2, b1, tau, df, ds)
            n_checked += 1
    print(f"[self-test] FAST == SAFE on {n_checked} random states across 4 instances.")


def probe_regression():
    """Reproduce the DP-validated probe numbers of Table 12 (instance S1)."""
    q, d = optimal_dispatch(23, 23, 0.802, S1)
    assert q == 0 and abs(d - 9.51) < 0.05, (q, d)      # saving 9.50, wait
    q, d = optimal_dispatch(18, 18, 0.934, S1)
    assert q == 11 and d >= S1.Cf, (q, d)               # DP dispatches 12
    q, d = optimal_dispatch(5, 4, 0.045, S5)
    assert q == 0, (q, d)                               # 7b probe: wait
    print("[regression] Table-12 probe values reproduced "
          "(wait at 9.51 < 10; q = 11 vs DP 12; S5 waits).")


def _evaluate_policy_under_dp(p, policy_fn):
    """EXACT policy evaluation: backward induction with the algorithm's
    policy held fixed (no minimisation). Returns V_alg over the full grid.
    V_alg >= V_opt everywhere; the difference is the suboptimality cost."""
    import numpy as np
    nI2 = p.I2_max - p.I2_min + 1
    nb1 = p.b1_max + 1
    V = np.zeros((nI2, nb1))
    for i in range(nI2):
        I2 = i + p.I2_min
        for j in range(nb1):
            V[i, j] = p.c1 * j + p.c2 * max(0, -I2) - p.v2 * max(0, I2)
    for n in range(1, p.N + 1):
        tau = n * p.dt
        Vn = np.empty_like(V)
        for i in range(nI2):
            I2 = i + p.I2_min
            for j in range(nb1):
                b1 = j
                q = policy_fn(I2, b1, tau) if (I2 > 0 and b1 > 0) else 0
                q = max(0, min(q, min(I2, b1))) if (I2 > 0 and b1 > 0) else 0
                I2a, b1a = I2 - q, b1 - q
                g = ((p.Cf if q > 0 else 0.0) + p.cu * q
                     + p.dt * (p.h * max(0, I2a) + p.pi1 * b1a
                               + p.pi2 * max(0, -I2a)))
                i0 = min(max(I2a, p.I2_min), p.I2_max) - p.I2_min
                i2 = min(max(I2a - 1, p.I2_min), p.I2_max) - p.I2_min
                j0 = min(b1a, p.b1_max)
                j1 = min(b1a + 1, p.b1_max)
                Vn[i, j] = (g + p.p0 * V[i0, j0]
                            + p.p1 * V[i0, j1] + p.p2 * V[i2, j0])
        V = Vn
    return V


def _simulate_policy(p, policy_fn, I2_0, b1_0, R=4000, seed=1):
    """Discrete-time Monte Carlo under the algorithm's policy.
    NOTE: this estimates V_alg (the algorithm's cost), NOT V_opt, so its
    confidence interval must be checked against the exact policy-evaluation
    value; the distance to V_opt is the suboptimality, not simulation error."""
    import random
    rng = random.Random(seed)
    costs = []
    for _ in range(R):
        I2, b1, total = I2_0, b1_0, 0.0
        for n in range(p.N, 0, -1):
            tau = n * p.dt
            q = policy_fn(I2, b1, tau) if (I2 > 0 and b1 > 0) else 0
            q = max(0, min(q, min(I2, b1))) if (I2 > 0 and b1 > 0) else 0
            I2 -= q
            b1 -= q
            total += ((p.Cf if q > 0 else 0.0) + p.cu * q
                      + p.dt * (p.h * max(0, I2) + p.pi1 * b1
                                + p.pi2 * max(0, -I2)))
            u = rng.random()
            if u < p.p1:
                b1 += 1
            elif u < p.p1 + p.p2:
                I2 -= 1
        total += p.c1 * b1 + p.c2 * max(0, -I2) - p.v2 * max(0, I2)
        costs.append(total)
    m = sum(costs) / R
    var = sum((c - m) ** 2 for c in costs) / (R - 1)
    hw = 1.96 * math.sqrt(var / R)
    return m, hw


def _simulate_policy_ct(p, policy_fn, I2_0, b1_0, R=4000, seed=2):
    """CONTINUOUS-TIME Gillespie simulation under the algorithm's policy.
    Inter-event times ~ Exp(lam1 + lam2); event types assigned by rate
    ratio; flow cost integrated EXACTLY over each inter-event interval.
    Dispatch decisions are re-examined at every event epoch with the true
    remaining time tau, plus once at t = 0. CAVEAT: between events the state
    is frozen but tau keeps falling, so a time-triggered dispatch could in
    principle occur mid-interval. In the symmetric case this cannot happen,
    because Delta(q_v) is nondecreasing in tau, so the dispatch region only
    shrinks as time passes and a waiting state stays waiting until the next
    event; epoch checking is therefore exact for S1. For asymmetric
    instances with pi1 < pi2 the region can grow as tau falls and a
    mid-interval trigger check must be added before reusing this simulator.
    NOTE: this shares NO discretisation structure with the DP, so its mean
    is expected to sit ABOVE the discrete values by a positive O(dt)-free...
    more precisely: the CT mean estimates the algorithm's cost in the TRUE
    continuous-time model; its deviation from the discrete V_alg is the
    discretisation error of the MDP approximation, expected positive and
    shrinking as N grows."""
    import random
    rng = random.Random(seed)
    lam = p.lam1 + p.lam2
    T = p.N * p.dt
    costs = []
    for _ in range(R):
        I2, b1, t, total = I2_0, b1_0, 0.0, 0.0

        def try_dispatch(I2, b1, t, total):
            tau = T - t
            if I2 > 0 and b1 > 0 and tau > 0:
                q = policy_fn(I2, b1, tau)
                q = max(0, min(q, min(I2, b1)))
                if q > 0:
                    total += p.Cf + p.cu * q
                    I2 -= q
                    b1 -= q
            return I2, b1, total

        I2, b1, total = try_dispatch(I2, b1, t, total)     # decision at t = 0
        while True:
            s = rng.expovariate(lam)                       # next inter-event time
            seg = min(s, T - t)
            total += seg * (p.h * max(0, I2) + p.pi1 * b1
                            + p.pi2 * max(0, -I2))         # exact integration
            t += seg
            if t >= T - 1e-12:
                break
            if rng.random() < p.lam1 / lam:                # event type by rates
                b1 += 1
            else:
                I2 -= 1
            I2, b1, total = try_dispatch(I2, b1, t, total)  # decide at each epoch
        total += p.c1 * b1 + p.c2 * max(0, -I2) - p.v2 * max(0, I2)
        costs.append(total)
    m = sum(costs) / R
    var = sum((c - m) ** 2 for c in costs) / (R - 1)
    hw = 1.96 * math.sqrt(var / R)
    return m, hw


def compare_with_dp(R_sim=4000):
    """Four-level comparison of the algorithm against the exact DP on the
    symmetric instance S1.
      Level 1: state-by-state policy agreement (diagnostic).
      Level 2: EXACT suboptimality via policy evaluation (primary result).
      Level 3: discrete-time MC, implementation check (pass/fail vs V_alg).
      Level 4: continuous-time Gillespie MC, discretisation check
               (expect positive O(dt) bias, shrinking with N)."""
    try:
        from solver import Params, TransshipmentDP
    except ImportError:
        print("[dp-comparison] solver.py not found; skipped.")
        return
    print("=" * 74)
    print("DP COMPARISON on instance S1 (symmetric, c1 = c2 = v2 = 0)")
    p = Params(T=1.5, N=300, lam1=6, lam2=10, h=0.5, Cf=10, cu=3.0,
               pi1=4, pi2=4, c1=0, c2=0, v2=0,
               I2_max=25, I2_min=-40, b1_max=70)
    dp = TransshipmentDP(p)
    dp.solve(verbose=False)
    policy_fn = lambda I2, b1, tau: optimal_dispatch(I2, b1, tau, S1)[0]

    # ── Level 1: policy agreement ───────────────────────────────────────
    mism_dec = mism_dec_band = mism_q = n_states = 0
    for n in (100, 200, 300):
        tau = n * p.dt
        for I2 in range(1, 24, 2):
            for b1 in range(1, 30, 4):
                n_states += 1
                q_dp = dp.get_policy(n, I2, b1)
                q_al = policy_fn(I2, b1, tau)
                band = abs(I2 - p.lam2 * tau) <= 1.0
                if (q_dp > 0) != (q_al > 0):
                    mism_dec += 1
                    mism_dec_band += band
                elif abs(q_dp - q_al) > 1:
                    mism_q += 1
    print(f" Level 1  policy agreement on {n_states} states:")
    print("          MEANING: where the algorithm and the DP disagree; a")
    print("          mismatch locates a difference but does not price it,")
    print("          so this level is diagnostic only.")
    print(f"          decision mismatches = {mism_dec} "
          f"(of which {mism_dec_band} in the |I2 - lam2*tau| <= 1 band), "
          f"|dq| > 1 = {mism_q}")

    # ── Level 2: exact suboptimality (primary result) ───────────────────
    V_alg = _evaluate_policy_under_dp(p, policy_fn)
    init_states = [(5, 0), (10, 0), (15, 5), (18, 18), (20, 10)]
    print(" Level 2  EXACT policy evaluation (no Monte Carlo noise):")
    print("          MEANING: V_alg is the expected cost of RUNNING the")
    print("          algorithm, computed exactly by backward induction with")
    print("          the policy frozen; gap = V_alg - V_opt >= 0 is the")
    print("          suboptimality. A negative gap would signal a bug.")
    print(f"          {'I2':>4} {'b1':>4} | {'V_opt':>9} {'V_alg':>9} "
          f"{'gap':>8} {'gap %':>7}")
    max_rel = 0.0
    rows_for_l3 = []
    for (I2, b1) in init_states:
        v_opt = dp.get_value(p.N, I2, b1)
        v_alg = float(V_alg[I2 - p.I2_min, b1])
        gap = v_alg - v_opt
        rel = 100.0 * gap / max(abs(v_opt), 1e-9)
        max_rel = max(max_rel, rel)
        rows_for_l3.append((I2, b1, v_opt, v_alg))
        print(f"          {I2:>4} {b1:>4} | {v_opt:>9.3f} {v_alg:>9.3f} "
              f"{gap:>8.3f} {rel:>6.2f}%")
    print(f"          max relative suboptimality of the algorithm = "
          f"{max_rel:.2f}%  (this is the headline number)")

    # ── Level 3: discrete-time Monte Carlo (implementation check) ───────
    print(f" Level 3  discrete-time Monte Carlo under the algorithm policy "
          f"(R = {R_sim}):")
    print("          MEANING: same discretisation as the DP, so the mean is")
    print("          an unbiased estimate of V_alg. PASS criterion: the 95%")
    print("          CI contains the exact V_alg of Level 2. A failure here")
    print("          means an implementation bug, not a model issue.")
    print(f"          {'I2':>4} {'b1':>4} | {'V_alg exact':>11} "
          f"{'sim mean':>9} {'95% CI':>19} | inside CI?")
    for (I2, b1, v_opt, v_alg) in rows_for_l3:
        m, hw = _simulate_policy(p, policy_fn, I2, b1, R=R_sim)
        ok = (m - hw) <= v_alg <= (m + hw)
        print(f"          {I2:>4} {b1:>4} | {v_alg:>11.3f} {m:>9.3f} "
              f"[{m - hw:>8.3f},{m + hw:>8.3f}] | "
              f"{'yes' if ok else 'NO - investigate'}")

    # ── Level 4: continuous-time Gillespie (discretisation check) ───────
    print(f" Level 4  continuous-time Gillespie under the algorithm policy "
          f"(R = {R_sim}):")
    print("          MEANING: shares NO discretisation with the DP; the mean")
    print("          estimates the algorithm's cost in the TRUE continuous-")
    print("          time model. PASS criterion is NOT 'CI contains V_alg':")
    print("          expect a small POSITIVE bias (CT mean above discrete")
    print("          V_alg) of order dt that shrinks as N grows; compare the")
    print("          bias column against the Mode-B pattern of Section 2.")
    print(f"          {'I2':>4} {'b1':>4} | {'V_alg exact':>11} "
          f"{'CT mean':>9} {'95% CI':>19} | {'bias':>7} {'rel.':>6}")
    for (I2, b1, v_opt, v_alg) in rows_for_l3:
        m, hw = _simulate_policy_ct(p, policy_fn, I2, b1, R=R_sim)
        bias = m - v_alg
        rel = 100.0 * bias / max(abs(v_alg), 1e-9)
        print(f"          {I2:>4} {b1:>4} | {v_alg:>11.3f} {m:>9.3f} "
              f"[{m - hw:>8.3f},{m + hw:>8.3f}] | {bias:>+7.3f} {rel:>5.2f}%")

    # ── Summary of what each level established ──────────────────────────
    print(" SUMMARY OF LEVELS")
    print("  L1 policy agreement   -> WHERE the algorithm differs from the DP")
    print("                           (diagnostic only; a decision mismatch")
    print("                           may cost almost nothing).")
    print("  L2 exact evaluation   -> HOW MUCH the differences cost: the")
    print("                           suboptimality gap, exact, the headline.")
    print("  L3 discrete-time MC   -> the evaluator and the policy code agree")
    print("                           (pass/fail; CI must contain V_alg).")
    print("  L4 continuous-time MC -> the discrete MDP faithfully represents")
    print("                           the continuous model under this policy")
    print("                           (expect positive O(dt) bias; rerun at")
    print("                           larger N to see it shrink).")


if __name__ == "__main__":
    self_test()
    probe_regression()
    run_instance(S1, [(23, 23, 0.802), (18, 18, 0.934), (20, 20, 1.335),
                      (16, 16, 0.934), (12, 5, 1.0)],
                 taus_for_curves=[0.5, 0.8, 1.0, 1.2])
    run_instance(S5, [(5, 4, 0.045), (20, 4, 0.045), (20, 15, 1.0)],
                 taus_for_curves=[0.1, 0.5, 1.0, 1.5])
    run_instance(A1, [(15, 15, 0.5), (15, 15, 0.9), (10, 3, 0.7)],
                 taus_for_curves=[0.3, 0.5, 0.7, 0.9])
    run_instance(A2, [(15, 15, 0.5), (15, 15, 1.0), (20, 20, 1.2)],
                 taus_for_curves=[0.3, 0.6, 0.9, 1.2])
    compare_with_dp()