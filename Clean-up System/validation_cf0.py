"""
validation_Cf0.py — validation of "Exact optimal threshold in Cf = 0" (Aug 2)
=============================================================================
Instance = the note's own example: lam1=3, lam2=5, h=1, pi1=4, pi2=5.5,
cu=5, T=5, zero terminal cost.  pi1 < pi2: the valley case of Theorem 4.

  1  threshold: DP vs staircase Eq.(20)-(22), breakpoints, Figure-1 redo
  2  decision agreement state by state
  3  value: Eq.(14) vs DP, absorbing part vs catchable band
  4  cost: staircase policy vs DP policy (policy evaluation + DES)
  5  structure: tau*, valley, lam1-independence
  6  cu question on this example (+ figure)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import exp, sqrt
from solver_cf0_2d import ParamsCf0, SwitchingDPCf0

L1, L2, H, CU, P1, P2, T = 3.0, 5.0, 1.0, 5.0, 4.0, 5.5, 5.0
N = 8000
IMIN, IMAX = -60, 16
TIE = 1e-9


def Mfun(n, tau, lam2=L2):
    mu = lam2 * tau
    E, pmf, cdf = 0.0, exp(-mu), exp(-mu)
    for k in range(1, n + 1):
        E += 1.0 - cdf
        pmf *= mu / k
        cdf += pmf
    return E


def gfun(tau, cu=CU, lam2=L2, h=H, pi1=P1, pi2=P2):
    return lam2 * (cu + (pi2 - pi1) * tau) / (h + pi2)


def stair(tau, cu=CU, lam2=L2, h=H, pi1=P1, pi2=P2, mmax=200):
    if tau <= 0:
        return np.inf
    g = gfun(tau, cu, lam2, h, pi1, pi2)
    mu = lam2 * tau
    E, pmf, cdf = 0.0, exp(-mu), exp(-mu)
    for m in range(1, mmax + 1):
        E += 1.0 - cdf
        if E >= g:
            return m
        pmf *= mu / m
        cdf += pmf
    return np.inf


def beta(I2, tau, lam1=L1, lam2=L2, h=H, pi1=P1, pi2=P2):
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


def solve(rule=None, cu=CU, lam1=L1, lam2=L2, h=H, pi1=P1, pi2=P2,
          T_=T, N_=N, imin=IMIN, imax=IMAX, want_decision=False, keep=()):
    """One common recursion. keep = iterable of n at which to snapshot V."""
    dt_ = T_ / N_
    p1, p2 = lam1 * dt_, lam2 * dt_
    p0 = 1 - p1 - p2
    I2 = np.arange(imin, imax + 1)
    down = np.clip(I2 - 1, imin, imax) - imin
    flow = dt_ * (h * np.maximum(I2, 0) + pi2 * np.maximum(-I2, 0))
    V = np.zeros(len(I2))
    keep = set(keep)
    snaps = {}
    dec = np.zeros((N_ + 1, len(I2)), np.int8) if want_decision else None
    for n in range(1, N_ + 1):
        tau = n * dt_
        rej = pi1 * tau + V
        dis = np.where(I2 >= 1, cu + V[down], np.inf)
        if rule is None:
            r1 = np.minimum(rej, dis)
            if want_decision:
                dec[n] = (dis < rej - TIE).astype(np.int8)
                dec[n][np.abs(dis - rej) <= TIE] = 2
        else:
            d = rule(tau, I2) & (I2 >= 1)
            r1 = np.where(d, dis, rej)
        V = flow + p1 * r1 + p2 * V[down] + p0 * V
        if n in keep:
            snaps[n] = V.copy()
    return I2, V, dec, snaps


dt = T / N
ii = lambda x: int(x - IMIN)
keepn = {round(t / dt) for t in (0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0)}
_, V_dp, DEC, SN = solve(want_decision=True, keep=keepn)

p_chk = ParamsCf0(T=T, N=2000, lam1=L1, lam2=L2, h=H, cu=CU,
                  pi1=P1, pi2=P2, c2=0.0, v2=0.0).with_auto_bounds()
dp_chk = SwitchingDPCf0(p_chk)
dp_chk.solve(store_V=True, verbose=False)
_, V_chk, _, _ = solve(N_=2000)
d0 = max(abs(V_chk[ii(x)] - dp_chk.get_value(2000, x)) for x in range(-5, 12))
print(f"[sanity] local recursion vs solver_cf0_2d, N=2000: max diff = {d0:.2e}")


def dp_thr(tau, dec, N_):
    n = round(tau * N_ / T)
    for I2 in range(1, IMAX + 1):
        if dec[n, ii(I2)] >= 1:
            return I2
    return np.inf


print()
print("=" * 78)
print("1  THRESHOLD: DP vs staircase Eq. (20)-(22)")
print("=" * 78)
_, _, DEC32, _ = solve(N_=32000, want_decision=True)
print(f"   {'tau':>5} | {'DP N=8000':>9} | {'DP N=32000':>10} | {'staircase':>9}")
for tau in (1.02, 1.05, 1.08, 1.2, 1.5, 2.0, 2.6, 2.75, 3.0, 3.5, 3.7,
            4.0, 4.4, 4.5, 5.0):
    v8 = dp_thr(tau, DEC, N)
    v32 = dp_thr(tau, DEC32, 32000)
    st = stair(tau)
    f = lambda x: "inf" if np.isinf(x) else f"{int(x)}"
    mark = "" if v8 == st else "   <-- differs"
    print(f"   {tau:>5} | {f(v8):>9} | {f(v32):>10} | {f(st):>9}{mark}")

tg = np.linspace(0.95, 5.0, 800)
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.step(tg, [stair(t) for t in tg], where="post", lw=1.8, color="#1F618D",
        label="staircase, Eq. (20)-(22)")
pts_t, pts_v = [], []
for tau in np.arange(1.0, 5.001, 0.02):
    v = dp_thr(round(tau, 4), DEC, N)
    if np.isfinite(v):
        pts_t.append(tau); pts_v.append(v)
ax.plot(pts_t, pts_v, "o", ms=2.4, color="#F39C12", alpha=0.8,
        label="DP threshold")
ax.axvline(1.0, color="#B03A2E", lw=1.0, ls="--",
           label=r"$\tau^\ast=c_u/(h+\pi_1)=1$")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\bar I_2(\tau)$")
ax.set_ylim(6, 12); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig("fig_threshold.pdf")
print("   saved fig_threshold.pdf")

print()
print("   breakpoints of {tau : stair(tau) <= n} vs the note's table:")
def crossings(nlev, lo=1.0, hi=5.0):
    Phi = lambda t: Mfun(nlev, t) - gfun(t)
    ts = np.linspace(lo, hi, 4001)
    vals = np.array([Phi(t) for t in ts])
    pos = vals >= 0
    if not pos.any():
        return None
    a = ts[np.argmax(pos)]
    b = ts[len(pos) - 1 - np.argmax(pos[::-1])]
    for _ in range(60):
        pass
    def bis(f, x0, x1):
        for _ in range(60):
            xm = 0.5 * (x0 + x1)
            if f(x0) * f(xm) <= 0: x1 = xm
            else: x0 = xm
        return 0.5 * (x0 + x1)
    a_ = bis(Phi, max(lo, a - 0.01), a) if Phi(max(lo, a - 0.01)) < 0 else a
    b_ = bis(Phi, b, min(hi, b + 0.01)) if Phi(min(hi, b + 0.01)) < 0 else b
    return a_, b_
note_tab = {7: (1.104, 2.708), 8: (1.039, 3.596), 9: (1.016, 4.466)}
for nlev, (na, nb) in note_tab.items():
    ab = crossings(nlev)
    print(f"   n={nlev}: computed [{ab[0]:.3f}, {ab[1]:.3f})   "
          f"note [{na}, {nb})")

print()
print("=" * 78)
print("2  DECISION AGREEMENT")
print("=" * 78)
taus = [round(0.05 * k, 4) for k in range(1, 101)]
tot = agr = ties = 0
mism = []
for tau in taus:
    n = round(tau / dt)
    s_thr = stair(tau)
    for I2 in range(1, 13):
        d_dp = DEC[n, ii(I2)]
        d_st = int(I2 >= s_thr)
        tot += 1
        if d_dp == 2:
            ties += 1; agr += 1; continue
        if d_dp == d_st:
            agr += 1
        else:
            mism.append((tau, I2, int(d_dp), d_st))
print(f"   grid I2=1..12 x tau=0.05..5.00 step 0.05 -> {tot} states")
print(f"   DP vs staircase: {agr}/{tot} = {100*agr/tot:.2f}%  "
      f"(exact numeric ties: {ties})")
if mism:
    n_se = sum(1 for *_, ddp, dst in mism if dst == 1 and ddp == 0)
    n_sl = sum(1 for *_, ddp, dst in mism if dst == 0 and ddp == 1)
    print(f"   direction: staircase dispatches / DP waits = {n_se},  "
          f"staircase waits / DP dispatches = {n_sl}")
    marg = [(H + P2) / L2 * (Mfun(i, t) - gfun(t)) for t, i, *_ in mism]
    print(f"   note-criterion margin at mismatch cells: "
          f"min {min(marg):+.4f}  max {max(marg):+.4f}")
    cells = sorted(set((t, i) for t, i, *_ in mism))
    print(f"   cells: {cells[:10]}{' ...' if len(cells) > 10 else ''}")

print()
print("=" * 78)
print("3  VALUE: Eq. (14) vs DP")
print("=" * 78)
floor = min(stair(round(t, 3)) for t in np.arange(1.0, 5.001, 0.005))
print(f"   valley floor = {int(floor)}; absorbing part: I2 <= {int(floor)-1}")
errA = 0.0
for tau in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
    n = round(tau / dt)
    Vn = SN[n] if n in SN else None
    for I2 in range(-6, int(floor)):
        errA = max(errA, abs(Vn[ii(I2)] - beta(I2, tau)))
print(f"   (a) absorbing part, max |V_DP - Eq.(14)| = {errA:.5f}  (O(dt))")
print(f"   (b) catchable band, gap Eq.(14) - V_DP  (option value):")
print(f"       {'tau':>5} | " + " | ".join(f"I2={i}" for i in range(7, 10)))
for tau in (3.0, 3.5, 4.0, 4.5, 5.0):
    n = round(tau / dt)
    Vn = SN[n]
    st_ = stair(tau)
    row = []
    for I2 in range(7, 10):
        if I2 < st_:
            row.append(f"{beta(I2, tau) - Vn[ii(I2)]:8.4f}")
        else:
            row.append(f"{'--':>8}")
    print(f"       {tau:>5} | " + " | ".join(row))

print()
print("=" * 78)
print("4  COST: staircase policy vs DP policy")
print("=" * 78)
_sc = {}
def rule_st(tau, I2):
    k = round(tau, 6)
    if k not in _sc: _sc[k] = stair(k)
    return I2 >= _sc[k]
_, V_st, _, SN_st = solve(rule=rule_st, keep=keepn)
print("   exact policy evaluation, one common recursion, dt=T/8000:")
print(f"   {'(I2,tau)':>10} | {'V_DP':>10} | {'V_stair':>10} | {'stair-DP':>9}")
for (I2, tau) in ((10, 5.0), (8, 5.0), (8, 4.0), (7, 3.0), (5, 3.0)):
    n = round(tau / dt)
    vd, vs = SN[n][ii(I2)], SN_st[n][ii(I2)]
    print(f"   ({I2:>2},{tau:>4}) | {vd:>10.4f} | {vs:>10.4f} | "
          f"{vs - vd:>9.5f}")

R, KMAX, I0 = 150_000, 110, 10
rng = np.random.default_rng(20260802)
lam = L1 + L2
times = np.cumsum(rng.exponential(1 / lam, size=(R, KMAX)), axis=1)
is1 = rng.random((R, KMAX)) < L1 / lam
assert (times[:, -1] > T).all()

def sim(kind):
    I2 = np.full(R, I0, np.int64)
    cost = np.zeros(R); t = np.zeros(R)
    for k in range(KMAX):
        tk = times[:, k]
        seg = np.maximum(np.minimum(tk, T) - t, 0.0)
        cost += seg * (H * np.maximum(I2, 0) + P2 * np.maximum(-I2, 0))
        t = np.minimum(tk, T)
        alive = tk < T
        if not alive.any(): break
        tau = T - t
        r1 = alive & is1[:, k]; r2 = alive & ~is1[:, k]
        if kind == "stair":
            thr = np.array([_sc.setdefault(round(tv, 3), stair(round(tv, 3)))
                            for tv in tau])
            disp = r1 & (I2 >= 1) & (I2 >= thr)
        else:
            n = np.clip(np.round(tau / dt).astype(np.int64), 1, N)
            idx = np.clip(I2, IMIN, IMAX) - IMIN
            disp = r1 & (I2 >= 1) & (DEC[n, idx] == 1)
        cost += np.where(disp, CU, 0.0)
        cost += np.where(r1 & ~disp, P1 * tau, 0.0)
        I2 -= (disp | r2).astype(np.int64)
    return cost

cs, cd = sim("stair"), sim("dp")
d = cs - cd
print(f"\n   DES, continuous time, {R:,} common paths, start ({I0},{T}):")
print(f"   STAIRCASE {cs.mean():.3f} ± {1.96*cs.std()/sqrt(R):.3f}   "
      f"DP {cd.mean():.3f} ± {1.96*cd.std()/sqrt(R):.3f}   "
      f"paired STAIR-DP = {d.mean():+.4f} ± {1.96*d.std()/sqrt(R):.4f}")

print()
print("=" * 78)
print("5  STRUCTURE: tau*, valley, lam1-independence")
print("=" * 78)
print(f"   DP threshold at tau=0.98: "
      f"{dp_thr(0.98, DEC, N)}   (note: inf for tau <= tau* = 1)")
vals = [dp_thr(round(t, 3), DEC, N) for t in np.arange(1.02, 5.001, 0.02)]
vmin = min(v for v in vals if np.isfinite(v))
mono_ok = True
seen_min = False
prev = np.inf
for v in vals:
    if not np.isfinite(v): continue
    if not seen_min:
        if v > prev: seen_min = True
    else:
        if v < prev: mono_ok = False
    prev = v
print(f"   DP valley: minimum threshold = {int(vmin)}; "
      f"valley shape (noninc then nondec): {mono_ok}")
_, _, DEC_l6, _ = solve(lam1=6.0, want_decision=True)
same = all(dp_thr(t, DEC, N) == dp_thr(t, DEC_l6, N)
           for t in (1.05, 1.2, 2.0, 3.0, 4.0, 5.0))
print(f"   lam1-independence (lam1=3 vs 6, six taus): {same}")

print()
print("=" * 78)
print("6  THE cu QUESTION ON THIS EXAMPLE")
print("=" * 78)
cus = np.arange(2.0, 8.01, 0.25)
vL, vR = [], []
for cu_ in cus:
    _, Vq, _, _ = solve(cu=cu_, T_=4.0, N_=5000)
    vL.append(Vq[ii(5)])
    vR.append(Vq[ii(8)])
# critical cu at which state (5, tau<=4) becomes catchable
ts = np.linspace(0.05, 4.0, 2000)
maxPhi = max((H + P2) / L2 * Mfun(5, t) - (P2 - P1) * t for t in ts)
b54 = beta(5, 4.0)
flat = [v for c, v in zip(cus, vL) if c >= maxPhi + 0.25]
print(f"   state (5, tau=4): beta = {b54:.4f}; critical cu = {maxPhi:.3f}; "
      f"max |V - beta| for cu > critical: "
      f"{max(abs(v - b54) for v in flat):.5f}")
print(f"   state (8, tau=4): V spans [{min(vR):.3f}, {max(vR):.3f}] "
      f"over cu in [2, 8]  -> cu-dependent")

fig2, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.3))
axL.axhline(b54, color="0.55", lw=1.0, ls=":",
            label=rf"Eq.(14)$={b54:.1f}$ (no $c_u$)")
axL.plot(cus, vL, "o-", ms=3.5, lw=1.5, color="#1F618D",
         label=r"$V^{\rm DP}(5,\tau{=}4)$")
axL.axvline(maxPhi, color="#B03A2E", lw=1.0, ls="--",
            label=rf"critical $c_u={maxPhi:.2f}$")
axL.set_xlabel(r"$c_u$"); axL.set_ylabel(r"$V(5,\tau{=}4)$")
axL.set_title(r"$I_2=5$ below the valley floor: flat once absorbing",
              fontsize=9.5)
axL.legend(fontsize=7, loc="lower right"); axL.grid(True, alpha=0.3)
axR.plot(cus, vR, "o-", ms=3.5, lw=1.5, color="#1F618D")
axR.set_xlabel(r"$c_u$"); axR.set_ylabel(r"$V(8,\tau{=}4)$")
axR.set_title(r"$I_2=8$ inside the sweep zone: $c_u$-dependent",
              fontsize=9.5)
axR.grid(True, alpha=0.3)
fig2.tight_layout(); fig2.savefig("fig_cu.pdf")
print("   saved fig_cu.pdf")