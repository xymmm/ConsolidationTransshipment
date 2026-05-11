"""
plot_Cf0_threshold_vs_tau.py
============================
Plots the Case-2 inventory threshold I2*(tau) vs remaining time tau,
for Cf=0, pi1=pi2, with varying cu values.

CSV output columns
------------------
  cu                        unit transshipment cost
  alpha                     lambda2*cu/(h+pi)
  alpha_ceil                ceil(alpha) = analytical integer prediction
  tau                       remaining time
  dp_threshold              DP-extracted I2* (blank = no dispatch observed)
  dispatch_occurs           yes / no
  diff_dp_minus_analytical  dp_threshold - alpha_ceil (blank if no dispatch)
  note                      plain-language explanation
"""

import math, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from solver import Params, TransshipmentDP

# ── parameters ─────────────────────────────────────────────────────────
LAM1, LAM2 = 5.0, 3.0
H, PI      = 1.0, 6.0      # h < pi always
C1=C2=V2=CF = 0.0
T, N       = 2.0, 400

CU_VALUES    = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
I2_MAX, I2_MIN, B1_MAX = 80, -10, 40
#40, -5, 20
TAU_VALUES   = np.linspace(0.05, T, 80)

# ── helpers ─────────────────────────────────────────────────────────────
def build_and_solve(cu):
    p = Params(T=T, N=N, lam1=LAM1, lam2=LAM2, h=H, Cf=CF, cu=cu,
               pi1=PI, pi2=PI, c1=C1, c2=C2, v2=V2,
               I2_max=I2_MAX, I2_min=I2_MIN, b1_max=B1_MAX)
    dp = TransshipmentDP(p)
    dp.solve(store_V=False, verbose=False)
    return dp

def extract_threshold(dp, tau):
    dt = dp.p.T / dp.p.N
    n  = min(dp.p.N, max(1, round(tau / dt)))
    for I2 in range(1, dp.p.I2_max + 1):
        if dp.get_policy(n, I2, 1) > 0:
            return I2
    return None

def alpha(cu):
    return LAM2 * cu / (H + PI)

# ── main ────────────────────────────────────────────────────────────────
records = []
fig, ax = plt.subplots(figsize=(9, 6))
colours = cm.RdYlBu_r(np.linspace(0.1, 0.9, len(CU_VALUES)))

for cu, col in zip(CU_VALUES, colours):
    a_val  = alpha(cu)
    a_ceil = math.ceil(a_val)
    label  = f'$c_u={cu}$  ($\\alpha={a_val:.2f}$)'

    print(f"Solving DP for cu={cu}...", end=" ", flush=True)
    dp = build_and_solve(cu)
    print("done.")

    thresholds = []
    for tau in TAU_VALUES:
        t = extract_threshold(dp, tau)
        thresholds.append(t if t is not None else np.nan)

        # plain-language note
        if t is None:
            diff = ''
            note = ('threshold exceeds I2_max: finite-horizon transient effect, '
                    'DP too conservative to dispatch at this tau')
        else:
            diff = t - a_ceil
            if diff == 0:
                note = 'matches analytical prediction exactly'
            elif abs(diff) == 1:
                note = 'within integer rounding (|diff|=1, OK)'
            elif diff > 1:
                note = (f'DP threshold {diff} unit(s) above analytical: '
                        f'finite-horizon transient term raises effective threshold')
            else:
                note = f'DP threshold {abs(diff)} unit(s) below analytical'

        records.append({
            'cu'                      : cu,
            'alpha'                   : round(a_val, 4),
            'alpha_ceil'              : a_ceil,
            'tau'                     : round(float(tau), 4),
            'dp_threshold'            : t if t is not None else '',
            'dispatch_occurs'         : 'yes' if t is not None else 'no',
            'diff_dp_minus_analytical': diff,
            'note'                    : note,
        })

    ax.plot(TAU_VALUES, thresholds,
            color=col, lw=2, marker='o', markersize=2.5, label=label)
    ax.axhline(a_ceil, color=col, lw=0.8, ls='--', alpha=0.5)

# ── plot formatting ──────────────────────────────────────────────────────
ax.set_ylabel(r'$I_2^*(\tau)$  (DP threshold)', fontsize=12)
ax.set_title(r'DP threshold vs $\tau$' + '\n' +
             r'$C_f=0$, $\pi_1=\pi_2=6$, $h=1$, varying $c_u$', fontsize=11)
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.set_xlim(T, 0)
ax.set_ylim(0, None)
ax.set_xlabel(r'$\tau$ (remaining time)  $\longleftarrow$  end of horizon',
              fontsize=11)
ax.grid(True, alpha=0.3)

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([T, T*0.75, T*0.5, T*0.25, 0.05])
ax2.set_xticklabels(['start', '', 'mid', '', 'end'], fontsize=9)
ax2.set_xlabel('planning horizon', fontsize=9)

plt.tight_layout()
# plt.savefig('Cf0_threshold_vs_tau.pdf', bbox_inches='tight', dpi=150)
plt.savefig('Cf0_threshold_vs_tau.png', bbox_inches='tight', dpi=150)
print("Saved: Cf0_threshold_vs_tau.pdf / .png")

# ── CSV export ────────────────────────────────────────────────────────────
FIELDS = ['cu','alpha','alpha_ceil','tau',
          'dp_threshold','dispatch_occurs',
          'diff_dp_minus_analytical','note']
with open('Cf0_threshold_vs_tau.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(records)

print(f"Saved: Cf0_threshold_vs_tau.csv  ({len(records)} rows, {len(CU_VALUES)} cu values x {len(TAU_VALUES)} tau points)")
plt.show()