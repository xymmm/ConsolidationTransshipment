"""
validation_Cf0.py — does the analytic policy equal the DP policy when Cf = 0?
=============================================================================
Instance = the note's own example: lam1 = lam2 = 5, h = 1, cu = 5,
pi1 = pi2 = 4, T = 5, zero terminal cost.

No b1 state in this model and, with Cf = 0, no batch: every Retailer-1
demand is either served on arrival with one unit at cost cu or rejected at
cost pi1*tau. The comparison is therefore:

  1  DECISION   dispatch/wait at every state (I2, tau): analytic vs DP
  2  THRESHOLD  Ibar(tau): analytic vs DP  (+ figure)
  3  COST       exact policy evaluation on one common recursion, plus a
                continuous-time discrete-event cross-check
  4  the cu question, with figures

Two analytic forms compared throughout:
  STAIRCASE  min{ m : E[min(K,m)] >= g(tau) }   exact difference threshold
  LINE       g(tau) = lam2 (cu + (pi2-pi1) tau)/(h+pi2)   note's Eq. (22)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import exp, sqrt
from solver_cf0_2d import ParamsCf0, SwitchingDPCf0

LAM1, LAM2, H, CU, PI1, PI2, T = 5.0, 5.0, 1.0, 5.0, 4.0, 4.0, 5.0
N = 8000
IMIN, IMAX = -60, 18
TIE = 1e-9


def Emin(m, mu):
    E, pmf, cdf = 0.0, exp(-mu), exp(-mu)
    for k in range(1, m + 1):
        E += 1.0 - cdf
        pmf *= mu / k
        cdf += pmf
    return E


def stair(tau, cu=CU, lam2=LAM2, h=H, pi1=PI1, pi2=PI2, mmax=200):
    if tau <= 0:
        return np.inf
    g = lam2 * (cu + (pi2 - pi1) * tau) / (h + pi2)
    mu = lam2 * tau
    E, pmf, cdf = 0.0, exp(-mu), exp(-mu)
    for m in range(1, mmax + 1):
        E += 1.0 - cdf
        if E >= g:
            return m
        pmf *= mu / m
        cdf += pmf
    return np.inf


def line(tau, cu=CU, lam2=LAM2, h=H, pi1=PI1, pi2=PI2):
    if tau < cu / (h + pi1):
        return np.inf
    return lam2 * (cu + (pi2 - pi1) * tau) / (h + pi2)


def solve(rule=None, cu=CU, lam1=LAM1, lam2=LAM2, h=H, pi1=PI1, pi2=PI2,
          T_=T, N_=N, imin=IMIN, imax=IMAX, want_decision=False):
    """One common recursion. rule None -> optimal DP; else
    rule(tau, I2 array) -> bool dispatch. Returns (I2 grid, V at n=N_, dec)."""
    dt_ = T_ / N_
    p1, p2 = lam1 * dt_, lam2 * dt_
    p0 = 1 - p1 - p2
    I2 = np.arange(imin, imax + 1)
    down = np.clip(I2 - 1, imin, imax) - imin
    flow = dt_ * (h * np.maximum(I2, 0) + pi2 * np.maximum(-I2, 0))
    V = np.zeros(len(I2))
    dec = np.zeros((N_ + 1, len(I2)), np.int8) if want_decision else None
    for n in range(1, N_ + 1):
        tau = n * dt_
        rej = pi1 * tau + V
        dis = np.where(I2 >= 1, cu + V[down], np.inf)
        if rule is None:
            r1 = np.minimum(rej, dis)
            if want_decision:
                dec[n] = (dis < rej - TIE).astype(np.int8)
                dec[n][np.abs(dis - rej) <= TIE] = 2      # exact tie
        else:
            d = rule(tau, I2) & (I2 >= 1)
            r1 = np.where(d, dis, rej)
        V = flow + p1 * r1 + p2 * V[down] + p0 * V
    return I2, V, dec


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


dt = T / N
ii = lambda x: int(x - IMIN)
I2g, V_dp, DEC = solve(want_decision=True)

p_chk = ParamsCf0(T=T, N=2000, lam1=LAM1, lam2=LAM2, h=H, cu=CU,
                  pi1=PI1, pi2=PI2, c2=0.0, v2=0.0).with_auto_bounds()
dp_chk = SwitchingDPCf0(p_chk)
dp_chk.solve(store_V=True, verbose=False)
_, V_chk, _ = solve(N_=2000)
d0 = max(abs(V_chk[ii(x)] - dp_chk.get_value(2000, x)) for x in range(-5, 12))
print(f"[sanity] local recursion vs solver_cf0_2d, N=2000: max diff = {d0:.2e}")

print()
print("=" * 78)
print("1  DECISION AGREEMENT, state by state")
print("=" * 78)
taus = [round(0.05 * k, 4) for k in range(1, 101)]
I2rng = range(1, 16)
tot = a_st = a_ln = ties = 0
mism_ln = []
for tau in taus:
    n = round(tau / dt)
    s_thr = stair(tau)
    l_thr = line(tau)
    for I2 in I2rng:
        d_dp = DEC[n, ii(I2)]
        d_st = int(I2 >= s_thr)
        d_ln = int(I2 >= l_thr)
        tot += 1
        if d_dp == 2:
            ties += 1
            a_st += 1
            a_ln += (d_ln == 1)
            continue
        a_st += (d_dp == d_st)
        a_ln += (d_dp == d_ln)
        if d_dp != d_ln:
            mism_ln.append((tau, I2, int(d_dp), d_ln))
print(f"   grid: I2 = 1..15 x tau = 0.05..5.00 step 0.05 -> {tot} states")
print(f"   DP vs STAIRCASE : {a_st}/{tot} = {100*a_st/tot:.2f}%   "
      f"(exact ties, counted as agreement: {ties})")
print(f"   DP vs LINE      : {a_ln}/{tot} = {100*a_ln/tot:.2f}%")
n_fire = sum(1 for *_, ddp, dln in mism_ln if dln == 1 and ddp == 0)
n_wait = sum(1 for *_, ddp, dln in mism_ln if dln == 0 and ddp == 1)
print(f"   LINE mismatch direction: line fires / DP waits = {n_fire},   "
      f"line waits / DP fires = {n_wait}")
band = {}
for tau, I2, *_ in mism_ln:
    band.setdefault(tau, []).append(I2)
if band:
    ex = sorted(band.items())[:3] + sorted(band.items())[-2:]
    print("   mismatch band (tau: I2 cells): "
          + "; ".join(f"{t}: {sorted(v)}" for t, v in ex))

def margin(I2, tau):
    """analytic per-decision margin: (h+pi2)/lam2 (E[min(K,I2)] - g);
    positive -> dispatch strictly better."""
    g = LAM2 * (CU + (PI2 - PI1) * tau) / (H + PI2)
    return (H + PI2) / LAM2 * (Emin(I2, LAM2 * tau) - g)

# classify every mismatch cell by |margin|
def classify(cells):
    near = [abs(margin(I2, tau)) for tau, I2, *_ in cells
            if abs(margin(I2, tau)) < 1e-2]
    far = [(tau, I2, margin(I2, tau)) for tau, I2, *_ in cells
           if abs(margin(I2, tau)) >= 1e-2]
    return near, far

mism_st = []
for tau in taus:
    n = round(tau / dt)
    s_thr = stair(tau)
    for I2 in I2rng:
        d_dp = DEC[n, ii(I2)]
        if d_dp != 2 and d_dp != int(I2 >= s_thr):
            mism_st.append((tau, I2))
near_st, far_st = classify(mism_st)
near_ln, far_ln = classify(mism_ln)
print(f"   STAIRCASE mismatches by analytic margin: "
      f"{len(near_st)} cells with |margin| < 1e-2 "
      f"(max {max(near_st) if near_st else 0:.1e}), "
      f"{len(far_st)} structural")
print(f"   LINE mismatches by analytic margin:      "
      f"{len(near_ln)} near-tie, {len(far_ln)} structural "
      f"(all in the tau ~ tau* sliver; largest |margin| "
      f"{max((abs(m) for *_, m in far_ln), default=0):.3f})")

print()
print("=" * 78)
print("2  THRESHOLD Ibar(tau)")
print("=" * 78)
print(f"   {'tau':>5} | {'DP':>4} | {'staircase':>9} | {'line (22)':>9}")
def dp_thr(tau):
    n = round(tau / dt)
    for I2 in range(1, IMAX + 1):
        if DEC[n, ii(I2)] >= 1:
            return I2
    return np.inf
for tau in (1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0):
    st_, ln_ = stair(tau), line(tau)
    f = lambda x: "inf" if np.isinf(x) else (f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}")
    print(f"   {tau:>5} | {f(dp_thr(tau)):>4} | {f(st_):>9} | {f(ln_):>9}")
print("   (DP scan counts exact ties as dispatch; the large-tau staircase")
print("    cell m=5 is an epsilon-tie with margin ~1e-4..1e-7)")

tg = np.linspace(0.9, 5.0, 400)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.step(tg, [stair(t) for t in tg], where="post", lw=1.8,
        color="#1F618D", label="staircase (exact)")
ax.plot(tg, [line(t) for t in tg], lw=1.6, ls="--", color="#B03A2E",
        label="line, Eq. (22)")
pts_t, pts_v = [], []
for tau in np.arange(0.95, 5.001, 0.05):
    v = dp_thr(round(tau, 4))
    if np.isfinite(v):
        pts_t.append(tau); pts_v.append(v)
ax.plot(pts_t, pts_v, "o", ms=3, color="#F39C12", alpha=0.8,
        label="DP (ties count as dispatch)")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\bar I_2(\tau)$")
ax.set_ylim(4, 12); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig("fig_threshold.pdf")
print("   saved fig_threshold.pdf")

print()
print("=" * 78)
print("3  COST: analytic policy vs DP policy")
print("=" * 78)
_stc, _lnc = {}, {}
def rule_st(tau, I2):
    k = round(tau, 6)
    if k not in _stc: _stc[k] = stair(k)
    return I2 >= _stc[k]
def rule_ln(tau, I2):
    k = round(tau, 6)
    if k not in _lnc: _lnc[k] = line(k)
    return I2 >= _lnc[k]
_, V_st, _ = solve(rule=rule_st)
_, V_ln, _ = solve(rule=rule_ln)
print("   exact policy evaluation, one common recursion, dt = T/8000:")
print(f"   {'(I2,tau)':>9} | {'V_DP':>10} | {'V_staircase':>11} | "
      f"{'V_line':>10} | {'stair-DP':>9} | {'line-DP':>9}")
for I2 in (8, 6, 5, 3):
    print(f"   ({I2:>2},{T:>3}) | {V_dp[ii(I2)]:>10.4f} | "
          f"{V_st[ii(I2)]:>11.4f} | {V_ln[ii(I2)]:>10.4f} | "
          f"{V_st[ii(I2)]-V_dp[ii(I2)]:>9.5f} | "
          f"{V_ln[ii(I2)]-V_dp[ii(I2)]:>9.5f}")

# structural sliver near tau* = 1: start inside it
for Tq in (1.05,):
    _, Vd_q, _ = solve(T_=Tq, N_=2000)
    _, Vs_q, _ = solve(rule=rule_st, T_=Tq, N_=2000)
    _, Vl_q, _ = solve(rule=rule_ln, T_=Tq, N_=2000)
    print(f"   sliver start ({8},{Tq}): V_DP={Vd_q[ii(8)]:.4f}  "
          f"stair-DP={Vs_q[ii(8)]-Vd_q[ii(8)]:+.5f}  "
          f"line-DP={Vl_q[ii(8)]-Vd_q[ii(8)]:+.5f}")

R, KMAX, I0 = 150_000, 110, 8
rng = np.random.default_rng(20260801)
lam = LAM1 + LAM2
times = np.cumsum(rng.exponential(1 / lam, size=(R, KMAX)), axis=1)
is1 = rng.random((R, KMAX)) < LAM1 / lam
assert (times[:, -1] > T).all()

def sim(kind):
    I2 = np.full(R, I0, np.int64)
    cost = np.zeros(R); t = np.zeros(R)
    for k in range(KMAX):
        tk = times[:, k]
        seg = np.maximum(np.minimum(tk, T) - t, 0.0)
        cost += seg * (H * np.maximum(I2, 0) + PI2 * np.maximum(-I2, 0))
        t = np.minimum(tk, T)
        alive = tk < T
        if not alive.any(): break
        tau = T - t
        r1 = alive & is1[:, k]; r2 = alive & ~is1[:, k]
        if kind == "stair":
            thr = np.array([_stc.setdefault(round(tv, 3), stair(round(tv, 3)))
                            for tv in tau])
        else:
            thr = np.where(tau >= CU / (H + PI1),
                           LAM2 * (CU + (PI2 - PI1) * tau) / (H + PI2), np.inf)
        disp = r1 & (I2 >= 1) & (I2 >= thr)
        cost += np.where(disp, CU, 0.0)
        cost += np.where(r1 & ~disp, PI1 * tau, 0.0)
        I2 -= (disp | r2).astype(np.int64)
    return cost

cs, cl = sim("stair"), sim("line")
d = cl - cs
print(f"\n   DES cross-check, continuous time, {R:,} common paths, "
      f"start ({I0},{T}):")
print(f"   STAIRCASE {cs.mean():.3f} ± {1.96*cs.std()/sqrt(R):.3f}   "
      f"LINE {cl.mean():.3f} ± {1.96*cl.std()/sqrt(R):.3f}   "
      f"paired LINE-STAIR = {d.mean():+.4f} ± {1.96*d.std()/sqrt(R):.4f}")

print()
print("=" * 78)
print("4  V AT A WAITING STATE vs cu")
print("=" * 78)
cus = np.arange(1.0, 8.01, 0.25)
vals = []
for cu_ in cus:
    _, Vq, _ = solve(cu=cu_, T_=3.0, N_=4000)
    vals.append(Vq[ii(3)])
b33 = beta(3, 3.0)
cu_crit = Emin(3, LAM2 * 3.0) * (H + PI2) / LAM2
flat = [v for c, v in zip(cus, vals) if c >= cu_crit + 0.25]
print(f"   equal-pi state (3, tau=3): beta = {b33:.4f}, "
      f"critical cu = (h+pi2)E[min(K,3)]/lam2 = {cu_crit:.4f}")
print(f"   max |V - beta| for cu >= {cu_crit+0.25:.2f}: "
      f"{max(abs(v-b33) for v in flat):.5f}")

l1e, l2e, he, p1e, p2e, Te, I2e = 5.0, 1.0, 1.0, 0.5, 6.0, 5.0, 5
cus2 = np.arange(0.75, 4.01, 0.25)
vals2 = []
for cu_ in cus2:
    _, Vq, _ = solve(cu=cu_, lam1=l1e, lam2=l2e, h=he, pi1=p1e, pi2=p2e,
                     T_=Te, N_=6000)
    vals2.append(Vq[ii(I2e)])
print(f"   pi2>pi1 state (5, tau=5): V spans [{min(vals2):.3f}, "
      f"{max(vals2):.3f}] over cu in [{cus2[0]}, {cus2[-1]}] -> cu-dependent")

fig2, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.3))
axL.axhline(b33, color="0.55", lw=1.0, ls=":",
            label=rf"$\beta(3,3)={b33:.1f}$ (no $c_u$)")
axL.plot(cus, vals, "o-", ms=3.5, lw=1.5, color="#1F618D",
         label=r"$V^{\rm DP}(3,\tau{=}3)$")
axL.axvline(cu_crit, color="#B03A2E", lw=1.0, ls="--",
            label=rf"critical $c_u={cu_crit:.2f}$")
axL.set_xlabel(r"$c_u$"); axL.set_ylabel(r"$V(3,\tau{=}3)$")
axL.set_title(r"$\pi_1=\pi_2$: flat inside the waiting region", fontsize=10)
axL.legend(fontsize=7, loc="lower right"); axL.grid(True, alpha=0.3)
axR.plot(cus2, vals2, "o-", ms=3.5, lw=1.5, color="#1F618D")
axR.set_xlabel(r"$c_u$"); axR.set_ylabel(r"$V(5,\tau{=}5)$")
axR.set_title(r"$\pi_2>\pi_1$: $c_u$-dependent inside the waiting region",
              fontsize=10)
axR.grid(True, alpha=0.3)
fig2.tight_layout(); fig2.savefig("fig_cu.pdf")
print("   saved fig_cu.pdf")