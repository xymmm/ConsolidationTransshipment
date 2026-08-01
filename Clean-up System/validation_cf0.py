"""
validation_Cf0.py — systematic validation of the Cf = 0 switching-control note
==============================================================================
Primary instance = the note's own example: lam1 = lam2 = 5, h = 1, cu = 5,
pi1 = pi2 = 4, T = 5, zero terminal cost.

  A  value function, waiting region:
       exact discrete benchmark beta(I2,tau)  vs  2-D DP  (O(dt) match)
       note's continuum closed form (15)      vs  exact   (O(1) gap = the
       dI2 -> partial approximation, quantified)
  B  threshold: DP at N = 2000 / 8000 / 32000  vs  exact staircase
       min{m : E[min(K,m)] >= g(tau)}          vs  continuum line (22)
       -> the dynamic threshold survives dt -> 0; it is integer-I2, not
          discrete-time
  C  continuous-time discrete-event policy duel (Bo's suggested experiment):
       NEVER / LINE (22) / STAIRCASE / DP on common random paths
  D  Bo's cu-question, equal-pi: V at a waiting state vs cu  (prediction: flat)
  E  cu-question, pi2 > pi1: same test  (prediction: cu-dependent)

Requires solver_cf0_2d.py in the same directory. numpy only.
"""
import numpy as np
from math import exp, sqrt
from solver_cf0_2d import ParamsCf0, SwitchingDPCf0, analytic_threshold

LAM1, LAM2, H, CU, PI1, PI2, T = 5.0, 5.0, 1.0, 5.0, 4.0, 4.0, 5.0


# ─────────────────────────────────────────────────────────────────────
#  exact discrete benchmark: never-dispatch value  beta(I2, tau)
#  (= the general-Cf note's beta; equals V on the waiting region because
#   the region is absorbing for pi1 >= pi2)
# ─────────────────────────────────────────────────────────────────────
def beta(I2, tau, lam1=LAM1, lam2=LAM2, h=H, pi1=PI1, pi2=PI2):
    base = (pi1 * lam1 + pi2 * lam2) / 2 * tau ** 2 - pi2 * I2 * tau
    if I2 <= 0:
        return base
    mu = lam2 * tau
    s, pmf = 0.0, exp(-mu)
    for k in range(0, I2 + 1):
        s += pmf * (I2 - k) * (I2 - k + 1)
        pmf *= mu / (k + 1)
    return (base + (h + pi2) * I2 * (I2 + 1) / (2 * lam2)
            - (h + pi2) / (2 * lam2) * s)


def formula15(I2, tau, lam1=LAM1, lam2=LAM2, h=H, pi1=PI1, pi2=PI2):
    """The note's continuum closed form, Eq. (15)."""
    if I2 <= lam2 * tau:
        return ((h + pi2) / (2 * lam2) * I2 ** 2 - pi2 * tau * I2
                + (pi1 * lam1 + pi2 * lam2) / 2 * tau ** 2)
    return h * tau * I2 + (pi1 * lam1 - h * lam2) / 2 * tau ** 2


def line22(tau, cu=CU, lam2=LAM2, h=H, pi1=PI1, pi2=PI2):
    """The note's continuum boundary, Eq. (22), with the tau* cutoff."""
    if tau < cu / (h + pi1):
        return np.inf
    return lam2 * (cu + (pi2 - pi1) * tau) / (h + pi2)


def mk(cu=CU, N=8000, T_=T, lam1=LAM1, lam2=LAM2, h=H, pi1=PI1, pi2=PI2):
    p = ParamsCf0(T=T_, N=N, lam1=lam1, lam2=lam2, h=h, cu=cu,
                  pi1=pi1, pi2=pi2, c2=0.0, v2=0.0).with_auto_bounds()
    dp = SwitchingDPCf0(p)
    dp.solve(store_V=True, verbose=False)
    return dp


print("=" * 78)
print("A  VALUE FUNCTION ON THE WAITING REGION")
print("=" * 78)
for N in (4000, 8000):
    dp = mk(N=N)
    err = 0.0
    for tau in (0.5, 1.0, 2.0, 3.0, 5.0):
        n = dp.n_for_tau(tau)
        thr = analytic_threshold(tau, dp.p)
        hi = int(min(thr, 12)) if np.isfinite(thr) else 12
        for I2 in range(-10, hi):
            err = max(err, abs(dp.get_value(n, I2) - beta(I2, tau)))
    print(f"   N={N:>6}  dt={T/N:.6f}   max |V_DP - beta| on waiting region "
          f"= {err:.5f}")
print("   -> halves with dt: pure time discretisation; beta is exact.")
print()
print("   gap of the note's continuum form (15) to the exact value, tau=3:")
print("   I2      exact=beta     Eq.(15)      exact-(15)    (h+pi2)I2/(2 lam2)")
for I2 in range(1, 6):
    b, f = beta(I2, 3.0), formula15(I2, 3.0)
    print(f"   {I2:>2}   {b:>12.4f} {f:>12.4f} {b-f:>12.4f} {((H+PI2)*I2/(2*LAM2)):>16.4f}")
print("   -> O(1) gap ~ (h+pi2)I2/(2 lam2) minus a Poisson tail:")
print("      Eq.(15) is the space-continuum approximation, not the exact value.")

print()
print("=" * 78)
print("B  THRESHOLD: DP vs exact staircase vs continuum line (22)")
print("=" * 78)
dps = {N: mk(N=N) for N in (2000, 8000, 32000)}
taus = [1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
hdr = "   tau   | " + " | ".join(f"DP N={N}" for N in dps) + \
      " | staircase | line (22)"
print(hdr)
for tau in taus:
    cells = []
    for N, dp in dps.items():
        v = dp.threshold_at_tau(tau)
        cells.append("inf" if np.isinf(v) else f"{v:.0f}")
    st = analytic_threshold(tau, dps[8000].p)
    ln = line22(tau)
    print(f"   {tau:>5} | " + " | ".join(f"{c:>7}" for c in cells)
          + f" | {('inf' if np.isinf(st) else int(st)):>9}"
          + f" | {('inf' if np.isinf(ln) else f'{ln:.0f}'):>9}")
print("   -> DP == staircase at every N; the line stays at 5.")
print("      The dynamic threshold survives dt -> 0, so it is not a")
print("      discrete-TIME artefact; it is the integer-I2 (space) effect")
print("      introduced by the dI2 -> partial approximation.")

print()
print("=" * 78)
print("C  CONTINUOUS-TIME POLICY DUEL  (common random numbers)")
print("=" * 78)
R, KMAX, I0 = 200_000, 110, 8
rng = np.random.default_rng(20260801)
lam = LAM1 + LAM2
times = np.cumsum(rng.exponential(1 / lam, size=(R, KMAX)), axis=1)
is1 = rng.random((R, KMAX)) < LAM1 / lam
assert (times[:, -1] > T).all()
dp8 = dps[8000]


def stair_disp(I2, tau):
    return I2 >= analytic_threshold(tau, dp8.p)


_stair_cache = {}


def stair_thr(tau):
    key = round(tau, 6)
    if key not in _stair_cache:
        _stair_cache[key] = analytic_threshold(tau, dp8.p)
    return _stair_cache[key]


def simulate(mode):
    I2 = np.full(R, I0, np.int64)
    cost = np.zeros(R)
    t = np.zeros(R)
    for k in range(KMAX):
        tk = times[:, k]
        seg = np.maximum(np.minimum(tk, T) - t, 0.0)
        cost += seg * (H * np.maximum(I2, 0) + PI2 * np.maximum(-I2, 0))
        t = np.minimum(tk, T)
        alive = tk < T
        if not alive.any():
            break
        tau = T - t
        r1 = alive & is1[:, k]
        r2 = alive & ~is1[:, k]
        if mode == "never":
            disp = np.zeros(R, bool)
        elif mode == "line":
            disp = tau >= CU / (H + PI1)
            disp &= I2 >= np.where(disp, line22_vec(tau), np.inf)
        elif mode == "stair":
            thr = np.array([stair_thr(tv) for tv in np.round(tau, 3)])
            disp = I2 >= thr
        else:  # dp policy lookup
            n = np.clip(np.round(tau / dp8.p.dt).astype(np.int64), 1, dp8.p.N)
            disp = dp8.policy[n, np.clip(I2, dp8.p.I2_min, dp8.p.I2_max)
                              - dp8.p.I2_min].astype(bool)
        disp &= (I2 >= 1) & r1
        cost += np.where(disp, CU, 0.0)
        cost += np.where(r1 & ~disp, PI1 * tau, 0.0)
        I2 -= (disp | r2).astype(np.int64)
    return cost


def line22_vec(tau):
    return LAM2 * (CU + (PI2 - PI1) * tau) / (H + PI2)


res = {m: simulate(m) for m in ("never", "line", "stair", "dp")}
n8 = dp8.n_for_tau(T)
print(f"   start (I2, tau) = ({I0}, {T}),  {R:,} paths")
print(f"   NEVER : {res['never'].mean():9.3f} ± "
      f"{1.96*res['never'].std()/sqrt(R):.3f}    beta({I0},{T}) = "
      f"{beta(I0, T):.3f}")
print(f"   LINE  : {res['line'].mean():9.3f} ± "
      f"{1.96*res['line'].std()/sqrt(R):.3f}    (note's boundary, Eq. 22)")
print(f"   STAIR : {res['stair'].mean():9.3f} ± "
      f"{1.96*res['stair'].std()/sqrt(R):.3f}    (exact staircase)")
print(f"   DP    : {res['dp'].mean():9.3f} ± "
      f"{1.96*res['dp'].std()/sqrt(R):.3f}    V_DP({I0},{T}) = "
      f"{dp8.get_value(n8, I0):.3f}")
d1 = res["line"] - res["stair"]
d2 = res["dp"] - res["stair"]
print(f"   paired:  LINE - STAIR = {d1.mean():+.4f} ± "
      f"{1.96*d1.std()/sqrt(R):.4f}     DP - STAIR = {d2.mean():+.4f} ± "
      f"{1.96*d2.std()/sqrt(R):.4f}")

print()
print("=" * 78)
print("D  BO'S QUESTION, EQUAL PI: V AT A WAITING STATE vs cu")
print("=" * 78)
I2q, tauq = 3, 3.0
print(f"   state (I2, tau) = ({I2q}, {tauq});  beta = {beta(I2q, tauq):.4f}"
      f"  (cu-free)")
print(f"   {'cu':>5} | {'staircase(tau)':>14} | {'V_DP(3,3)':>12} | "
      f"{'V_DP - beta':>12}")
for cu in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
    dpq = mk(cu=cu, N=6000, T_=tauq)
    v = dpq.get_value(dpq.p.N, I2q)
    st = analytic_threshold(tauq, dpq.p)
    print(f"   {cu:>5} | {('inf' if np.isinf(st) else int(st)):>14} | "
          f"{v:>12.4f} | {v - beta(I2q, tauq):>12.4f}")
print("   -> flat at beta once the state is in the waiting region:")
print("      the region is ABSORBING for pi1 >= pi2 (I2 never rises and the")
print("      boundary never falls as time passes), so no path ever")
print("      transships and cu never enters. cu moves the REGION, not the")
print("      value inside it.")

print()
print("=" * 78)
print("E  BO'S QUESTION, PI2 > PI1: THE SAME TEST")
print("=" * 78)
l1e, l2e, he, p1e, p2e, Te = 5.0, 1.0, 1.0, 0.5, 6.0, 5.0
I2e = 5
print(f"   instance lam1={l1e} lam2={l2e} h={he} pi1={p1e} pi2={p2e}; "
      f"state ({I2e}, {Te})")
print(f"   {'cu':>5} | {'thr(now)':>9} | {'min thr later':>13} | "
      f"{'V_DP':>12}")
for cu in (1.5, 2.0, 3.0):
    dpe = mk(cu=cu, N=8000, T_=Te, lam1=l1e, lam2=l2e, h=he,
             pi1=p1e, pi2=p2e)
    v = dpe.get_value(dpe.p.N, I2e)
    thr_now = analytic_threshold(Te, dpe.p)
    thr_min = min(analytic_threshold(tv, dpe.p)
                  for tv in np.arange(0.2, Te, 0.05))
    f = lambda x: "inf" if np.isinf(x) else f"{x:.0f}"
    print(f"   {cu:>5} | {f(thr_now):>9} | {f(thr_min):>13} | {v:>12.4f}")
print("   -> with pi2 > pi1 the boundary FALLS as time passes and can sweep")
print("      past a waiting state, so V there depends on cu even though the")
print("      state is in the waiting region today.")