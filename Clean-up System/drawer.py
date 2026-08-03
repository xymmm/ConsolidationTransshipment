"""
drawer.py  —  Local, standalone plots of the DP threshold vs tau.

    Figure 1:  b1* threshold (Case 1)  vs  tau     (one curve per I2)
    Figure 2:  Ibar2 threshold (Case 2) vs  tau

Figure 2 can additionally overlay, when Cf = 0:
    - the 2-D Cf=0 DP        solver_cf0_2d.py   [the note's own model]
    - the note's analytic staircase              eq. (20)-(22)

Value extraction and styling match the app.

Run:
    python drawer.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from solver import Params, TransshipmentDP


# ══════════════════════════════════════════════════════════════════════
#  PARAMETERS  —  edit everything here
# ══════════════════════════════════════════════════════════════════════
T    = 5.0      # horizon length
N    = 800      # periods for the 3-D (I2, b1) DP

lam1 = 3.0      # Retailer 1 demand rate
lam2 = 5.0      # Retailer 2 demand rate

h    = 1.0      # holding cost
Cf   = 0.0      # fixed ship cost
cu   = 2.0      # unit ship cost
pi1  = 4.0      # Retailer 1 penalty
pi2  = 5.5      # Retailer 2 penalty

c1   = 0.0      # terminal: clear R1 backlog
c2   = 0.0      # terminal: clear R2 backlog
v2   = 0.0      # terminal: salvage R2 inventory

# --- plot controls ---
b1_fixed = 20   # fixed b1 used for the Case-2 (Ibar2) threshold
n_curves = 3    # number of I2 curves in the Case-1 (b1*) figure

# RETAINED: applies to the Case-2 figure only.
#   False -> participation threshold Ibar2 (smallest I2 that dispatches)
#   True  -> retained level Ibar2 - 1 (inventory kept after dispatch)
RETAINED = False

# --- Cf = 0 comparison overlays (Figure 2 only, ignored when Cf > 0) ---
SHOW_CF0_2D       = True    # overlay the 2-D Cf=0 DP (the note's own model)
SHOW_CF0_ANALYTIC = True    # overlay the note's analytic staircase
N_CF0             = 4000    # periods for the 2-D DP; a 1-D state makes this cheap
PRINT_COMPARISON  = True    # print a numeric table of the three curves

# --- plot grid density (tau axis) ---
N_UNIFORM  = 300   # uniform tau points over (0, T]
N_DENSE    = 400   # extra points clustered just above tau*
DENSE_BAND = 0.6   # width of the dense cluster above tau*

# --- state space for the 3-D DP ---
AUTO_BOUNDS = True
I2_MIN, I2_MAX, B1_MAX = -10, 40, 40   # used only when AUTO_BOUNDS = False

# --- Figure 3: b1bar vs I2 at EXACT tau values (full-scan DP + analytic) ---
TAU_LIST_B1BAR = [1.0, 2.0, 5.0]

# --- Figure 4: dispatch-vs-wait margin vs I2, and a printed Q-table --------
B1_FOR_MARGIN  = 3            # fixed b1 for the margin curve
STATE_FOR_Q    = (6, 3, 5.0)  # (I2, b1, tau) for the printed action values

SAVE_CASE1 = "threshold_case1_b1star.png"
SAVE_CASE2 = "threshold_case2_I2bar.png"
SAVE_B1BAR = "threshold_b1bar_vs_I2.png"
SAVE_MARGIN = "margin_vs_I2.png"
# ══════════════════════════════════════════════════════════════════════


# ── state-space bounds for the 3-D DP ─────────────────────────────────
if AUTO_BOUNDS:
    s1, s2 = lam1 * T, lam2 * T
    I2_min = -int(math.ceil(s2 + 4.0 * math.sqrt(s2)))
    I2_max =  int(math.ceil(max(40.0, s2 + 4.0 * math.sqrt(s2))))
    b1_max =  int(math.ceil(s1 + 4.0 * math.sqrt(s1)))
else:
    I2_min, I2_max, b1_max = I2_MIN, I2_MAX, B1_MAX

print(f"3-D DP state space: I2 in [{I2_min}, {I2_max}], b1_max = {b1_max}")


# ── solve the 3-D (I2, b1) DP once ────────────────────────────────────
p = Params(
    T=T, N=N, lam1=lam1, lam2=lam2,
    h=h, Cf=Cf, cu=cu, pi1=pi1, pi2=pi2,
    c1=c1, c2=c2, v2=v2,
    I2_max=I2_max, I2_min=I2_min, b1_max=b1_max,
)
dp = TransshipmentDP(p)
dp.solve(store_V=True, verbose=False)   # store_V: needed for Figures 3-4

title_params = f"lam2={lam2}, cu={cu}, h={h}, Cf={Cf}, pi1={pi1}, pi2={pi2}, T={T}"


def n_for_tau(tau):
    dt = T / N
    return min(N, max(1, round(tau / dt)))


def tau_grid():
    """
    Dense tau grid. A uniform grid over (0, T] plus a fine cluster just above
    tau* = cu/(h+pi1), where the threshold plunges through one-unit microsteps.
    A coarse uniform grid aliases those microsteps away.
    """
    tau_star = cu / max(h + pi1, 1e-9)
    base  = np.linspace(0.05, T, N_UNIFORM)
    lo    = max(0.05, tau_star - 0.05)
    hi    = min(T, tau_star + DENSE_BAND)
    dense = np.linspace(lo, hi, N_DENSE) if hi > lo else np.array([])
    return np.unique(np.concatenate([base, dense]))


# ── DP threshold extractors (same scan order as the app) ──────────────
def dp_b1star(I2, tau):
    """Smallest b1 >= 1 at which the DP dispatches, at the given I2."""
    n = n_for_tau(tau)
    I2q = max(I2_min, min(I2_max, I2))
    for b1t in range(1, b1_max + 1):    # FULL scan: capping at I2 silently
        if dp.get_policy(n, I2q, b1t) > 0:  # hides thresholds > I2
            return b1t
    return np.nan


def dp_I2bar(tau, b1):
    """Smallest I2 >= 1 at which the DP dispatches, at the given b1."""
    n   = n_for_tau(tau)
    b1q = max(0, min(b1_max, b1))
    for I2t in range(1, I2_max + 1):
        if dp.get_policy(n, I2t, b1q) > 0:
            return I2t
    return np.nan


xs = tau_grid()


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 1:  b1* threshold (Case 1) vs tau
# ══════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 5.5))
colours   = cm.tab10(np.linspace(0, 0.9, n_curves))
vary_vals = np.linspace(1, I2_max, n_curves).astype(int)

for vv, col in zip(vary_vals, colours):
    ys_dp = [dp_b1star(int(vv), float(x)) for x in xs]
    ax1.plot(xs, ys_dp, color=col, lw=2, label=f"DP  I2={vv}")

ax1.invert_xaxis()
ax1.set_xlabel("tau  <-  end of horizon", fontsize=11)
ax1.set_ylabel("b1* threshold (Case 1)", fontsize=11)
ax1.set_title(f"b1* threshold (Case 1)  vs  tau (remaining time)\n{title_params}",
              fontsize=10)
ax1.legend(fontsize=8, loc='best', framealpha=0.85)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(SAVE_CASE1, dpi=150)
print(f"Saved {SAVE_CASE1}")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 2:  Ibar2 threshold (Case 2) vs tau
# ══════════════════════════════════════════════════════════════════════
offset = 1.0 if RETAINED else 0.0      # retained level = threshold - 1

ys_3d = np.array([dp_I2bar(float(x), b1_fixed) for x in xs], dtype=float) - offset
y_label_txt = "Retained I2  (= Ibar2 - 1)" if RETAINED else "Ibar2 threshold (Case 2)"

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
ax2.plot(xs, ys_3d, color='steelblue', lw=2,
         label="3-D DP (I2, b1)  [solver.py]")

# ── Cf = 0 overlays ───────────────────────────────────────────────────
ys_2d = ys_an = None
if Cf == 0 and (SHOW_CF0_2D or SHOW_CF0_ANALYTIC):
    from solver_cf0_2d import ParamsCf0, SwitchingDPCf0, analytic_curve

    p0 = ParamsCf0(T=T, N=N_CF0, lam1=lam1, lam2=lam2,
                   h=h, cu=cu, pi1=pi1, pi2=pi2,
                   c2=c2, v2=v2).with_auto_bounds()

    if SHOW_CF0_2D:
        dp2 = SwitchingDPCf0(p0)
        dp2.solve(verbose=False)
        ys_2d = dp2.threshold_curve(xs) - offset
        ax2.plot(xs, ys_2d, color='seagreen', lw=1.8, ls='--',
                 label="2-D Cf=0 DP (I2, tau)  [note's model]")

    if SHOW_CF0_ANALYTIC:
        ys_an = analytic_curve(xs, p0) - offset
        ax2.plot(xs, ys_an, color='crimson', lw=1.8,
                 label="Note analytic staircase  (eq. 20-22)")

ax2.invert_xaxis()
ax2.set_xlabel("tau  <-  end of horizon", fontsize=11)
ax2.set_ylabel(y_label_txt, fontsize=11)
ax2.set_title(f"{y_label_txt}  vs  tau (remaining time)\n{title_params}",
              fontsize=10)
ax2.legend(fontsize=8, loc='best', framealpha=0.85)
ax2.grid(True, alpha=0.3)
finite = ys_3d[np.isfinite(ys_3d)]
if len(finite):
    ax2.set_ylim(0, min(np.nanmax(finite) * 1.15, 40))
fig2.tight_layout()
fig2.savefig(SAVE_CASE2, dpi=150)
print(f"Saved {SAVE_CASE2}")


# ── numeric comparison ────────────────────────────────────────────────
if PRINT_COMPARISON and Cf == 0 and ys_2d is not None and ys_an is not None:
    print("\n  tau | 3-D DP | 2-D DP | analytic")
    print("  ----+--------+--------+---------")
    for tau in [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:
        i = int(np.argmin(np.abs(xs - tau)))
        print(f" {tau:4.1f} | {ys_3d[i]:6.0f} | {ys_2d[i]:6.0f} | {ys_an[i]:7.0f}")


# ══════════════════════════════════════════════════════════════════════
#  Analytic threshold of the general-Cf note, Eq. (36)  (for Figure 3)
# ══════════════════════════════════════════════════════════════════════
def _Em(mu, mmax):
    E = np.zeros(mmax + 1)
    pmf, cdfb, M = math.exp(-mu), 0.0, 0.0
    for m in range(1, mmax + 1):
        M += max(1.0 - (cdfb + pmf), 0.0)
        E[m] = M
        cdfb += pmf
        pmf *= mu / m
    return E


def an_b1bar_general(I2, tau):
    """b1bar of Eq. (36); np.nan means +infinity. Valid for any Cf >= 0."""
    if I2 < 1 or tau <= 0:
        return np.nan
    E = _Em(lam2 * tau, I2)
    d = lambda m: (h + pi2) / lam2 * E[m] - cu + (pi1 - pi2) * tau
    S = 0.0
    for b in range(1, I2 + 1):
        dm = d(I2 - b + 1)
        if dm <= 0:
            return np.nan
        S += dm
        if S > Cf:
            return float(b)
    return np.nan


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 3:  b1bar vs I2 at exact tau values (DP full scan + analytic)
# ══════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(10, 5))
cols3 = cm.tab10(np.linspace(0, 0.9, max(len(TAU_LIST_B1BAR), 2)))
I2_axis = np.arange(1, I2_max + 1)
for tv, col in zip(TAU_LIST_B1BAR, cols3):
    n = n_for_tau(tv)
    te = n * T / N                       # effective tau actually used by the DP
    ys_dp3 = [dp_b1star(int(I2), tv) for I2 in I2_axis]
    ax3.step(I2_axis, ys_dp3, where="mid", color=col, lw=2,
             label=f"DP  tau={tv:g} (eff {te:.4g})")
    ys_an3 = [an_b1bar_general(int(I2), te) for I2 in I2_axis]
    ax3.step(I2_axis, ys_an3, where="mid", color=col, lw=1.4, ls="--",
             alpha=0.75, label=f"analytic eq.(36)  tau={te:.4g}")
ax3.set_xlabel("I2", fontsize=11)
ax3.set_ylabel("b1bar(I2, tau)", fontsize=11)
ax3.set_title("Dispatch threshold vs I2 at exact tau\n"
              "missing points = +infinity (dispatch never pays)   "
              + title_params, fontsize=10)
ax3.legend(fontsize=8, loc="best", framealpha=0.85)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(SAVE_B1BAR, dpi=150)
print(f"Saved {SAVE_B1BAR}")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 4:  dispatch-vs-wait margin, two sweeps
#  margin = Q(wait) - min_q Q(q), both under OPTIMAL continuation.
#  >0: dispatching now is strictly cheaper. A dip below 0 between positive
#  neighbours is a waiting pocket: the crease, measured in cost units.
#  Orange bands mark every stretch where waiting is strictly cheaper.
# ══════════════════════════════════════════════════════════════════════
def _shade_negatives(ax, xs, ms, label_fmt="wait wins by {:.3f}"):
    ms = np.asarray(ms, dtype=float)
    neg = np.isfinite(ms) & (ms < 0)
    runs, i0 = [], None
    for k, v in enumerate(neg):
        if v and i0 is None:
            i0 = k
        elif not v and i0 is not None:
            runs.append((i0, k - 1)); i0 = None
    if i0 is not None:
        runs.append((i0, len(neg) - 1))
    step = (xs[1] - xs[0]) if len(xs) > 1 else 1.0
    for (s0, s1) in runs:
        ax.axvspan(xs[s0] - 0.5 * step, xs[s1] + 0.5 * step,
                   color="#F5B041", alpha=0.30, zorder=1)
        kmin = s0 + int(np.nanargmin(ms[s0:s1 + 1]))
        ax.annotate(label_fmt.format(abs(ms[kmin])),
                    xy=(xs[kmin], ms[kmin]), xytext=(0, -24),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#7E5109",
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#7E5109"))
    return runs


fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 7))

# (a) sweep along I2 at fixed b1, one line per tau
I2_axis = np.arange(1, I2_max + 1)
ax4a.axhline(0, color="0.35", lw=1.0)
for tv, col in zip(TAU_LIST_B1BAR, cols3):
    n = n_for_tau(tv)
    te = n * T / N
    ms = [dp.wait_margin(n, int(I2), B1_FOR_MARGIN) for I2 in I2_axis]
    ax4a.plot(I2_axis, ms, marker="o", ms=3, lw=1.6, color=col,
              label=f"tau={tv:g} (eff {te:.4g})", zorder=3)
    if abs(tv - TAU_LIST_B1BAR[-1]) < 1e-12:
        runs = _shade_negatives(ax4a, I2_axis, ms)
        if runs:
            print("\nWaiting pockets along I2 at "
                  f"b1={B1_FOR_MARGIN}, tau={tv:g}:")
            for s0, s1 in runs:
                print(f"   I2 in [{I2_axis[s0]}, {I2_axis[s1]}]: "
                      f"waiting cheaper by up to "
                      f"{abs(min(ms[s0:s1+1])):.4f}")
ax4a.set_xlabel("I2", fontsize=11)
ax4a.set_ylabel("Q(wait) - best dispatch", fontsize=11)
ax4a.set_title(f"(a) margin vs I2 at b1={B1_FOR_MARGIN}   "
               "(>0: dispatch now is cheaper; orange: waiting is cheaper)",
               fontsize=10)
ax4a.legend(fontsize=8, loc="best", framealpha=0.85)
ax4a.grid(True, alpha=0.3)

# (b) sweep along tau at the crease state
qI2c, qb1c = STATE_FOR_Q[0], STATE_FOR_Q[1]
ns = np.unique(np.linspace(1, N, 200).astype(int))
tau_axis = ns * T / N
msb = [dp.wait_margin(int(nn), qI2c, qb1c) for nn in ns]
ax4b.axhline(0, color="0.35", lw=1.0)
ax4b.plot(tau_axis, msb, lw=1.8, color="#1F618D", zorder=3)
runs_b = _shade_negatives(ax4b, tau_axis, msb)
ax4b.set_xlabel("tau", fontsize=11)
ax4b.set_ylabel("Q(wait) - best dispatch", fontsize=11)
ax4b.set_title(f"(b) margin vs tau at (I2={qI2c}, b1={qb1c})", fontsize=10)
ax4b.grid(True, alpha=0.3)
if runs_b:
    print(f"\nWaiting pockets along tau at (I2={qI2c}, b1={qb1c}):")
    for s0, s1 in runs_b:
        print(f"   tau in [{tau_axis[s0]:.4g}, {tau_axis[s1]:.4g}]: "
              f"waiting cheaper by up to {abs(min(msb[s0:s1+1])):.4f}")

fig4.suptitle(title_params, fontsize=9)
fig4.tight_layout()
fig4.savefig(SAVE_MARGIN, dpi=150)
print(f"Saved {SAVE_MARGIN}")


# ── printed action values at one state ────────────────────────────────
qI2, qb1, qtau = STATE_FOR_Q
nq = n_for_tau(qtau)
print(f"\nAction values at (I2={qI2}, b1={qb1}, tau_eff={nq*T/N:.4g}):")
for q, v in dp.action_values(nq, qI2, qb1):
    tag = "wait    " if q == 0 else f"q = {q}   "
    print(f"   {tag}Q = {v:12.4f}")
mg = dp.wait_margin(nq, qI2, qb1)
print(f"   margin Q(wait) - best dispatch = {mg:+.4f}  "
      f"({'dispatch' if mg > 0 else 'wait'} is strictly cheaper)")


plt.show()