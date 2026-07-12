"""
cf0_tasks.py
===============================================================================
Three tasks for the Cf = 0 switching-control question (Bo's red remark).

TASK 1  DP threshold convergence (THE ARBITER).
        Extract the participation threshold Ibar_DP(tau) from the exact DP at
        N = 200 / 400 / 800 and compare against two predictions:
          (a) Bo's fluid constant  c~ = lam2*cu/(h+pi);
          (b) the exact-pasting fixed point
              Ibar(tau) = c~ + 1 + E[(Ibar-1-N)+],  N ~ Poisson(lam2*tau),
              derived from the exact Poisson solution of the waiting
              equation (difference form, no PDE step).
        READ-OUT: if the DP curve converges (as dt -> 0) to (a), the dynamic
        threshold was a discretisation artifact and Bo's remark stands; if it
        converges to (b), the difference-to-derivative step is the cause and
        the true threshold is mildly tau-dependent.

TASK 2  Continuous-time discrete-event cost comparison (Bo's experiment).
        Simulate the true continuous-time system under the two unit-dispatch
        threshold policies (constant vs dynamic), with COMMON RANDOM NUMBERS,
        and report the PAIRED cost difference.
        READ-OUT: expect a tiny difference even if the thresholds differ by
        1-2 units (the cost is flat near the optimum); report thresholds AND
        costs, not costs alone.

TASK 3  Switching-class gap probe (pi2 > pi1).
        The switching formulation never clears backlog accumulated while
        waiting. For pi2 > pi1 the dispatch region GROWS as time passes, so a
        state can enter it carrying old backlog which the full 3-D model can
        clear but the switching class cannot. Compute
            gap(I2, b1, tau) = [V_switch(I2,tau) + pi1*b1*tau] - V_opt3D(I2,b1,tau)
        READ-OUT: gap ~ O(dt) everywhere  => class lossless, Theorem 4 policy
        globally optimal; gap >> O(dt) at "time-passage entry" states => the
        class restriction binds and the note needs a policy-class caveat.

Requires solver.py. Not executed here; run with
    python cf0_tasks.py          # full
    FAST=1 python cf0_tasks.py   # coarse smoke test
===============================================================================
"""

import math
import os

FAST = bool(int(os.environ.get("FAST", "0")))


# ═══════════════════════════════════════════════════════════════════════
#  Poisson helpers (no scipy needed)
# ═══════════════════════════════════════════════════════════════════════
def poisson_pmf_upto(mu, kmax):
    """Return [pmf(0), ..., pmf(kmax)] iteratively (stable for mu <~ 700)."""
    p = math.exp(-mu)
    out = [p]
    for k in range(1, kmax + 1):
        p *= mu / k
        out.append(p)
    return out


def e_pos_part_below(x, mu):
    """E[(x - N)+] for N ~ Poisson(mu), x real >= 0."""
    if x <= 0:
        return 0.0
    kmax = int(math.floor(x - 1e-12))
    pmf = poisson_pmf_upto(mu, kmax)
    return sum((x - k) * pk for k, pk in enumerate(pmf))


def exact_pasting_threshold(tau, lam2, cu, h, pi):
    """Fixed point Ibar = c~ + 1 + E[(Ibar - 1 - N_{lam2*tau})+].
    The map has derivative P(N <= Ibar-1) < 1, so iteration converges.
    Heuristic smooth pasting on the EXACT difference equation; the DP of
    Task 1 is the arbiter, this is the quantitative prediction."""
    c_tilde = lam2 * cu / (h + pi)
    mu = lam2 * tau
    I = c_tilde + 1.0
    for _ in range(200):
        I_new = c_tilde + 1.0 + e_pos_part_below(I - 1.0, mu)
        if abs(I_new - I) < 1e-10:
            break
        I = I_new
    return I


# ═══════════════════════════════════════════════════════════════════════
#  TASK 1 -- DP threshold convergence at Cf = 0
# ═══════════════════════════════════════════════════════════════════════
# Bo's example: lam1 = lam2 = 5, cu = 5, h = 1, pi = 4  =>  tau* = 1, c~ = 5.
T1_PAR = dict(T=3.0, lam1=5, lam2=5, h=1.0, cu=5.0, pi=4.0)
B1_AMPLE = 30


def solve_dp_cf0(N, par=T1_PAR):
    from solver import Params, TransshipmentDP
    lamT2 = par["lam2"] * par["T"]
    lamT1 = par["lam1"] * par["T"]
    p = Params(T=par["T"], N=N,
               lam1=par["lam1"], lam2=par["lam2"], h=par["h"],
               Cf=0.0, cu=par["cu"], pi1=par["pi"], pi2=par["pi"],
               c1=0.0, c2=0.0, v2=0.0,
               I2_max=22,
               I2_min=-int(math.ceil(lamT2 + 8 * math.sqrt(lamT2))),
               b1_max=int(math.ceil(B1_AMPLE + lamT1 + 8 * math.sqrt(lamT1))))
    dp = TransshipmentDP(p)
    dp.solve(verbose=False)
    return p, dp


def dp_threshold(p, dp, n):
    """Smallest I2 >= 1 at which the DP dispatches, ample backlog; None if
    it never dispatches at this n. Also return the retained stock at the
    first dispatching I2 two units above the threshold (integer probe)."""
    thr = None
    for I2 in range(1, p.I2_max + 1):
        if dp.get_policy(n, I2, B1_AMPLE) >= 1:
            thr = I2
            break
    ret = None
    if thr is not None and thr + 2 <= p.I2_max:
        q = dp.get_policy(n, thr + 2, B1_AMPLE)
        ret = thr + 2 - q
    return thr, ret


def task1_threshold_convergence():
    print("=" * 78)
    print("TASK 1: Cf = 0 participation threshold, DP vs constant vs "
          "exact-pasting curve")
    par = T1_PAR
    c_tilde = par["lam2"] * par["cu"] / (par["h"] + par["pi"])
    tau_star = par["cu"] / (par["pi"] + par["h"])
    print(f" params: {par}   c~ = {c_tilde:.3f}   tau* = {tau_star:.3f}")
    print(" MEANING: column 'fluid' is Bo's constant threshold; column")
    print(" 'exact fp' is the fixed-point prediction from the exact Poisson")
    print(" waiting solution. If the DP columns converge (left to right) to")
    print(" 'fluid', the dynamics was a discretisation artifact; if they")
    print(" converge to 'exact fp', the difference-to-derivative step is the")
    print(" cause and the true threshold is mildly tau-dependent.")
    Ns = (100, 200) if FAST else (200, 400, 800)
    solves = {}
    for N in Ns:
        solves[N] = solve_dp_cf0(N)
        print(f"   ... solved N = {N}")
    taus = [1.1, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0]
    hdr = f" {'tau':>5} |" + "".join(f" DP N={N:>4}" for N in Ns) \
        + f" | {'fluid':>6} {'exact fp':>9} | retained (finest N)"
    print(hdr)
    for tau in taus:
        row = f" {tau:>5.2f} |"
        ret_str = ""
        for N in Ns:
            p, dp = solves[N]
            n = min(max(int(round(tau / p.dt)), 1), p.N)
            thr, ret = dp_threshold(p, dp, n)
            row += f" {('None' if thr is None else thr):>9}"
            if N == Ns[-1]:
                ret_str = f"{ret}"
        fp = exact_pasting_threshold(tau, par["lam2"], par["cu"],
                                     par["h"], par["pi"])
        row += f" | {c_tilde:>6.2f} {fp:>9.3f} | {ret_str}"
        print(row)
    print(" NOTE: integer thresholds; compare the DP column with round(exact")
    print(" fp) and with round(fluid). The retained column should track")
    print(" (threshold - 1) if the DP dispatches down to just below the")
    print(" participation level.")
    return solves


# ═══════════════════════════════════════════════════════════════════════
#  TASK 2 -- continuous-time discrete-event comparison (CRN, paired)
# ═══════════════════════════════════════════════════════════════════════
def make_event_stream(par, rng, T):
    """One replication's event stream: sorted (time, type) with
    type 1 = retailer-1 arrival, type 2 = retailer-2 demand."""
    lam = par["lam1"] + par["lam2"]
    t, events = 0.0, []
    while True:
        t += rng.expovariate(lam)
        if t >= T:
            break
        typ = 1 if rng.random() < par["lam1"] / lam else 2
        events.append((t, typ))
    return events


def ct_cost_under_threshold(events, thr_fn, par, T, I2_0):
    """Cost of one path under a unit-dispatch threshold policy.
    Serve a retailer-1 arrival at time t (tau = T - t) iff I2 >= thr_fn(tau)
    and I2 >= 1; otherwise the demand is backordered and PREPAYS pi1*tau
    (pathwise identical to flow accounting because the switching class never
    clears old backlog). Flow h*I2+ + pi2*I2- integrated exactly between
    events. Terminal costs zero (c1 = c2 = v2 = 0)."""
    I2, t_prev, cost = I2_0, 0.0, 0.0
    for (t, typ) in events:
        cost += (t - t_prev) * (par["h"] * max(I2, 0)
                                + par["pi"] * max(-I2, 0))
        t_prev = t
        tau = T - t
        if typ == 2:
            I2 -= 1
        else:
            if I2 >= 1 and I2 >= thr_fn(tau):
                cost += par["cu"]
                I2 -= 1
            else:
                cost += par["pi"] * tau
    cost += (T - t_prev) * (par["h"] * max(I2, 0) + par["pi"] * max(-I2, 0))
    return cost


def task2_ct_comparison(R=20000, I2_0=15, dp_thr_table=None):
    print("=" * 78)
    print("TASK 2: continuous-time cost comparison, constant vs dynamic "
          "threshold (CRN)")
    import random
    par = T1_PAR
    T = par["T"]
    c_tilde = par["lam2"] * par["cu"] / (par["h"] + par["pi"])
    tau_star = par["cu"] / (par["pi"] + par["h"])

    def thr_const(tau):                      # Bo's Theorem 4
        return c_tilde if tau > tau_star else math.inf

    if dp_thr_table:                          # dynamic curve from Task 1
        knots = sorted(dp_thr_table.items())

        def thr_dyn(tau):
            if tau <= tau_star:
                return math.inf
            best = None
            for tk, vk in knots:
                if vk is not None and abs(tk - tau) <= (
                        abs(best[0] - tau) if best else math.inf):
                    best = (tk, vk)
            return best[1] if best else math.inf
        dyn_name = "DP threshold table (Task 1, finest N)"
    else:
        def thr_dyn(tau):
            if tau <= tau_star:
                return math.inf
            return exact_pasting_threshold(tau, par["lam2"], par["cu"],
                                           par["h"], par["pi"])
        dyn_name = "exact-pasting fixed-point curve"

    print(f" policies: A = constant {c_tilde:.2f} (tau > tau*);  "
          f"B = {dyn_name}")
    print(" MEANING: common random numbers, so the PAIRED column is the")
    print(" statistically meaningful one. Expect |paired diff| to be small")
    print(" even if the thresholds differ by 1-2 units: the cost surface is")
    print(" flat near the optimum. A tie here does NOT contradict Task 1;")
    print(" it means the two policies are practically equivalent while the")
    print(" exact threshold is still the dynamic one.")
    rng = random.Random(7)
    ca, cb, diff = [], [], []
    for _ in range(R):
        ev = make_event_stream(par, rng, T)
        a = ct_cost_under_threshold(ev, thr_const, par, T, I2_0)
        b = ct_cost_under_threshold(ev, thr_dyn, par, T, I2_0)
        ca.append(a)
        cb.append(b)
        diff.append(a - b)
    def stats(v):
        m = sum(v) / len(v)
        var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
        return m, 1.96 * math.sqrt(var / len(v))
    ma, ha = stats(ca)
    mb, hb = stats(cb)
    md, hd = stats(diff)
    print(f" I2_0 = {I2_0}, R = {R}")
    print(f"  A constant : mean = {ma:9.4f}  95% CI +- {ha:.4f}")
    print(f"  B dynamic  : mean = {mb:9.4f}  95% CI +- {hb:.4f}")
    print(f"  PAIRED A-B : mean = {md:+9.4f}  95% CI +- {hd:.4f}   "
          f"({'B better' if md > hd else 'A better' if md < -hd else 'tie'})")


# ═══════════════════════════════════════════════════════════════════════
#  TASK 3 -- switching-class gap probe (pi2 > pi1)
# ═══════════════════════════════════════════════════════════════════════
# Their Figure-1 example: lam1=3, lam2=4, h=1.5, cu=6, pi1=2, pi2=5.
T3_PAR = dict(T=3.0, lam1=3, lam2=4, h=1.5, cu=6.0, pi1=2.0, pi2=5.0)


def switching_value_2d(par, N):
    """Exact 2-D DP of the SWITCHING class (prepaid accounting):
      mode 0: flow h*I2+ + pi2*I2- + lam1*pi1*tau ; I2 falls at rate lam2
      mode 1 (I2>=1): flow h*I2 + lam1*cu        ; I2 falls at rate lam1+lam2
    Terminal 0. Returns V[n, ii] and the mode-1 region for threshold read-off."""
    import numpy as np
    T, lam1, lam2 = par["T"], par["lam1"], par["lam2"]
    h, cu, pi1, pi2 = par["h"], par["cu"], par["pi1"], par["pi2"]
    dt = T / N
    I2_min = -int(math.ceil(lam2 * T + 8 * math.sqrt(lam2 * T)))
    I2_max = 22
    nI2 = I2_max - I2_min + 1
    I2v = np.arange(I2_min, I2_max + 1, dtype=float)
    p2, pL = lam2 * dt, (lam1 + lam2) * dt
    V = np.zeros(nI2)
    Vall = np.zeros((N + 1, nI2))
    mode1 = np.zeros((N + 1, nI2), dtype=bool)
    for n in range(1, N + 1):
        tau = n * dt
        Vm1 = np.concatenate([V[:1], V[:-1]])          # V at I2 - 1 (clip)
        c0 = dt * (h * np.maximum(I2v, 0) + pi2 * np.maximum(-I2v, 0)
                   + lam1 * pi1 * tau) + p2 * Vm1 + (1 - p2) * V
        c1 = dt * (h * I2v + lam1 * cu) + pL * Vm1 + (1 - pL) * V
        c1[I2v < 1] = np.inf                            # mode 1 needs stock
        V = np.minimum(c0, c1)
        Vall[n] = V
        mode1[n] = c1 < c0 - 1e-12
    return Vall, mode1, I2_min, dt


def task3_class_gap(N=None):
    print("=" * 78)
    print("TASK 3: switching-class gap, pi2 > pi1 (region grows as time "
          "passes)")
    from solver import Params, TransshipmentDP
    par = T3_PAR
    N = N or (150 if FAST else 400)
    kappa = par["h"] + par["pi2"]
    beta = par["lam2"] * par["cu"] / kappa
    m = par["lam2"] * (par["pi2"] - par["pi1"]) / kappa
    tau_star = par["cu"] / (par["pi1"] + par["h"])
    print(f" params: {par}")
    print(f" Ibar(tau) = {beta:.3f} + {m:.3f}*tau  for tau > tau* = "
          f"{tau_star:.3f}  (increasing => region grows as tau falls... ")
    print(" i.e. as TIME PASSES the threshold falls and states enter the")
    print(" region carrying backlog accumulated while waiting).")
    # switching-class value (2-D, prepaid)
    Vsw, mode1, I2_min_sw, dt = switching_value_2d(par, N)
    # full 3-D optimum
    lamT1, lamT2 = par["lam1"] * par["T"], par["lam2"] * par["T"]
    p = Params(T=par["T"], N=N,
               lam1=par["lam1"], lam2=par["lam2"], h=par["h"],
               Cf=0.0, cu=par["cu"], pi1=par["pi1"], pi2=par["pi2"],
               c1=0.0, c2=0.0, v2=0.0,
               I2_max=22,
               I2_min=-int(math.ceil(lamT2 + 8 * math.sqrt(lamT2))),
               b1_max=int(math.ceil(12 + lamT1 + 8 * math.sqrt(lamT1))))
    dp = TransshipmentDP(p)
    dp.solve(store_V=True, verbose=False)      # get_value(n, ...) needs V_all
    print(" MEANING of the gap: cost of the best SWITCHING-CLASS policy")
    print(" minus the unrestricted 3-D optimum, at the same state. gap ~")
    print(" O(dt) everywhere => class lossless (Theorem 4 globally optimal);")
    print(" gap >> O(dt) at b1 > 0 entry states => the class restriction")
    print(" binds and the note needs an explicit policy-class caveat.")
    tol_dt = 3.0 * (par["pi1"] + par["pi2"]) * (par["lam1"]
                                                + par["lam2"]) * par["T"] / N
    print(f" O(dt) allowance ~ {tol_dt:.3f}")
    print(f" {'tau':>5} {'I2':>4} {'b1':>4} | {'V_switch+sunk':>13} "
          f"{'V_opt3D':>9} | {'gap':>8} | binding?")
    # probe: states below the time-0 threshold that the falling boundary
    # will sweep, with accumulated backlog b1
    for tau in (3.0, 2.5, 2.0):
        n = min(int(round(tau / dt)), N)
        for I2 in (6, 8, 10):
            for b1 in (0, 2, 4, 6):
                v_sw = Vsw[n, I2 - I2_min_sw] + par["pi1"] * b1 * (n * dt)
                v_op = dp.get_value(n, I2, b1)
                gap = v_sw - v_op
                flag = "YES" if gap > tol_dt else "no"
                print(f" {tau:>5.2f} {I2:>4} {b1:>4} | {v_sw:>13.4f} "
                      f"{v_op:>9.4f} | {gap:>8.4f} | {flag}")
    # threshold cross-check: 3-D DP boundary vs the fluid line
    print(" 3-D DP participation threshold (ample b1) vs fluid line "
          "beta + m*tau:")
    for tau in (2.0, 2.5, 3.0):
        n = min(int(round(tau / dt)), N)
        thr = None
        for I2 in range(1, p.I2_max + 1):
            if dp.get_policy(n, I2, 12) >= 1:
                thr = I2
                break
        print(f"   tau = {tau:.2f}:  DP thr = {thr}   fluid = "
              f"{beta + m * tau:.3f}")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"cf0_tasks.py  (FAST={FAST})")
    solves = task1_threshold_convergence()
    # feed Task 1's finest DP threshold into Task 2 as the dynamic policy
    p, dp = solves[max(solves)]
    table = {}
    for tau in (1.1, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0):
        n = min(max(int(round(tau / p.dt)), 1), p.N)
        thr, _ = dp_threshold(p, dp, n)
        table[tau] = thr
    task2_ct_comparison(R=4000 if FAST else 20000, I2_0=15,
                        dp_thr_table=table)
    task3_class_gap()