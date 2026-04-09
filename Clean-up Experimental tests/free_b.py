"""
figure_cf0_boundary.py
======================
Produces a single figure to verify that at Cf=0 the optimal dispatch
decision depends only on (I2, tau) and NOT on b1.

For several values of b1, we extract the minimum I2 at which dispatch
is triggered (q* > 0) at each time snapshot tau.  If all curves
coincide, the policy is independent of b1.

Usage:
    python figure_cf0_boundary.py

Requires solver.py in the same directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from solver import Params, TransshipmentDP

# ── Parameters ──────────────────────────────────────────────────────
p = Params(
    T=2.0, N=200,
    lam1=8.0, lam2=5.0,
    h=0.1, Cf=0.0, cu=11.0,   # Cf = 0 is the key setting
    pi1=10.0, pi2=10.0,
    c1=5.0, c2=5.0, v2=1.0,
    I2_max=25, I2_min=-15, b1_max=80,
)

# ── Solve ────────────────────────────────────────────────────────────
print("Solving DP with Cf=0 ...")
dp = TransshipmentDP(p)
dp.solve(store_V=False, verbose=True)
print("Done.\n")

# ── Extract dispatch boundary ────────────────────────────────────────
# For each (tau, b1), find the minimum I2 >= 1 such that q*(n, I2, b1) > 0.
# If no such I2 exists in [1, I2_max], record NaN.

b1_values  = [1, 3, 5, 8, 12, 18]          # b1 levels to compare
I2_search  = list(range(1, p.I2_max + 1))   # I2 from 1 to 25
n_values   = list(range(1, p.N + 1))        # n = 1 .. N
tau_values = [n * p.dt for n in n_values]   # corresponding tau

# boundary[b1][n_index] = min I2 that triggers dispatch
boundary = {}
for b1 in b1_values:
    trig = []
    for n in n_values:
        found = np.nan
        for I2 in I2_search:
            if dp.get_policy(n, I2, b1) > 0:
                found = I2
                break
        trig.append(found)
    boundary[b1] = np.array(trig)

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(b1_values)))

for b1, col in zip(b1_values, colors):
    ax.plot(tau_values, boundary[b1],
            label=f'$b_1 = {b1}$', color=col, linewidth=1.8)

ax.set_xlabel(r'Remaining time $\tau$', fontsize=13)
ax.set_ylabel(r'Dispatch trigger $I_2^{\min}(\tau)$', fontsize=13)
ax.set_title(r'Dispatch boundary in $(I_2, \tau)$ plane at $C_f = 0$'
             '\n(curves for different $b_1$ should coincide)',
             fontsize=12)
ax.legend(title='$b_1$', fontsize=10, title_fontsize=10,
          loc='upper left', framealpha=0.8)
ax.set_xlim(0, p.T)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure_cf0_boundary.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {out_path}")