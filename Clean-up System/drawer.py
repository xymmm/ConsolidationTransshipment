"""
drawer.py  —  Local, standalone plots of the DP threshold vs τ.

    Figure 1:  b₁* threshold (Case 1)  vs  τ     (one curve per I₂)
    Figure 2:  Ī₂ threshold (Case 2)   vs  τ     (single curve)

Only the DP result is drawn. Value extraction and styling match the app.

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
N    = 400      # number of periods (discretisation; use >= 400 to keep the
                # microsteps near τ* inside the DP solution)

lam1 = 3.0      # Retailer 1 demand rate
lam2 = 5.0      # Retailer 2 demand rate

h    = 1.0      # holding cost
Cf   = 0.0      # fixed ship cost
cu   = 5.0      # unit ship cost
pi1  = 4.0      # Retailer 1 penalty
pi2  = 5.5      # Retailer 2 penalty

c1   = 0.0      # terminal: clear R1 backlog
c2   = 0.0      # terminal: clear R2 backlog
v2   = 0.0      # terminal: salvage R2 inventory

# --- plot controls ---
b1_fixed = 5    # fixed b₁ used for the Case-2 (Ī₂) threshold
n_curves = 3    # number of I₂ curves in the Case-1 (b₁*) figure

# RETAINED: applies to the Case-2 (Ī₂) figure only.
#   False -> participation threshold Ī₂ (smallest I₂ that dispatches)
#   True  -> retained level Ī₂ − 1 (inventory kept after dispatch, i.e. the
#            largest I₂ that does NOT dispatch). Use to match a figure that
#            plots retained inventory instead of the participation threshold.
RETAINED = False

# --- plot grid density (τ axis) ---
N_UNIFORM = 300   # uniform τ points over (0, T]
N_DENSE   = 400   # extra points clustered just above τ* to show the microsteps
DENSE_BAND = 0.6  # width of the dense cluster above τ*

# --- state space ---
AUTO_BOUNDS = True       # scale bounds with demand (recommended)
I2_MIN, I2_MAX, B1_MAX = -10, 40, 40   # used only when AUTO_BOUNDS = False

SAVE_CASE1 = "threshold_case1_b1star.png"
SAVE_CASE2 = "threshold_case2_I2bar.png"
# ══════════════════════════════════════════════════════════════════════


# ── state-space bounds ────────────────────────────────────────────────
if AUTO_BOUNDS:
    s1, s2 = lam1 * T, lam2 * T
    I2_min = -int(math.ceil(s2 + 4.0 * math.sqrt(s2)))
    I2_max =  int(math.ceil(max(40.0, s2 + 4.0 * math.sqrt(s2))))
    b1_max =  int(math.ceil(s1 + 4.0 * math.sqrt(s1)))
else:
    I2_min, I2_max, b1_max = I2_MIN, I2_MAX, B1_MAX

print(f"State space: I2 in [{I2_min}, {I2_max}], b1_max = {b1_max}")


# ── solve the DP once ─────────────────────────────────────────────────
p = Params(
    T=T, N=N, lam1=lam1, lam2=lam2,
    h=h, Cf=Cf, cu=cu, pi1=pi1, pi2=pi2,
    c1=c1, c2=c2, v2=v2,
    I2_max=I2_max, I2_min=I2_min, b1_max=b1_max,
)
dp = TransshipmentDP(p)
dp.solve(store_V=False, verbose=False)   # only the policy tensor is needed

title_params = f"λ₂={lam2}, cu={cu}, h={h}, Cf={Cf}, π₁={pi1}, π₂={pi2}, T={T}"


def n_for_tau(tau):
    dt = T / N
    return min(N, max(1, round(tau / dt)))


def tau_grid():
    """
    Dense τ grid. A uniform grid over (0, T] plus a fine cluster just above
    τ* = cu/(h+π₁), where the threshold plunges from +∞ through a few one-unit
    microsteps. A coarse uniform grid samples too few points there and aliases
    the microsteps away.
    """
    tau_star = cu / max(h + pi1, 1e-9)
    base  = np.linspace(0.05, T, N_UNIFORM)
    lo    = max(0.05, tau_star - 0.05)
    hi    = min(T, tau_star + DENSE_BAND)
    dense = np.linspace(lo, hi, N_DENSE) if hi > lo else np.array([])
    return np.unique(np.concatenate([base, dense]))


# ── DP threshold extractors (same scan order as the app) ──────────────
def dp_b1star(I2, tau):
    """Smallest b₁ >= 1 at which the DP dispatches, at the given I₂."""
    n = n_for_tau(tau)
    I2q = max(I2_min, min(I2_max, I2))
    for b1t in range(1, min(I2q, b1_max) + 1):
        if dp.get_policy(n, I2q, b1t) > 0:
            return b1t
    return np.nan


def dp_I2bar(tau, b1):
    """Smallest I₂ >= 1 at which the DP dispatches, at the given b₁."""
    n   = n_for_tau(tau)
    b1q = max(0, min(b1_max, b1))
    for I2t in range(1, I2_max + 1):
        if dp.get_policy(n, I2t, b1q) > 0:
            return I2t
    return np.nan


xs = tau_grid()


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 1:  b₁* threshold (Case 1) vs τ   (one DP curve per I₂)
# ══════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 5.5))
colours   = cm.tab10(np.linspace(0, 0.9, n_curves))
vary_vals = np.linspace(1, I2_max, n_curves).astype(int)     # I₂ values

for vv, col in zip(vary_vals, colours):
    ys_dp = [dp_b1star(int(vv), float(x)) for x in xs]
    ax1.plot(xs, ys_dp, color=col, lw=2, label=f"DP  I₂={vv}")

ax1.invert_xaxis()
ax1.set_xlabel("τ  ←  end of horizon", fontsize=11)
ax1.set_ylabel("b₁* threshold (Case 1)", fontsize=11)
ax1.set_title(f"b₁* threshold (Case 1)  vs  τ (remaining time)\n{title_params}",
              fontsize=10)
ax1.legend(fontsize=8, loc='best', framealpha=0.85)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(SAVE_CASE1, dpi=150)
print(f"Saved {SAVE_CASE1}")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 2:  Ī₂ threshold (Case 2) vs τ   (single DP curve)
# ══════════════════════════════════════════════════════════════════════
offset = 1.0 if RETAINED else 0.0      # retained level = threshold − 1

ys_dp = np.array([dp_I2bar(float(x), b1_fixed) for x in xs], dtype=float) - offset
y_label_txt = "Retained I₂  (= Ī₂ − 1)" if RETAINED else "Ī₂ threshold (Case 2)"

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
ax2.plot(xs, ys_dp, color='steelblue', lw=2, label="DP")

ax2.invert_xaxis()
ax2.set_xlabel("τ  ←  end of horizon", fontsize=11)
ax2.set_ylabel(y_label_txt, fontsize=11)
ax2.set_title(f"{y_label_txt}  vs  τ (remaining time)\n{title_params}",
              fontsize=10)
ax2.legend(fontsize=8, loc='best', framealpha=0.85)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(SAVE_CASE2, dpi=150)
print(f"Saved {SAVE_CASE2}")

plt.show()