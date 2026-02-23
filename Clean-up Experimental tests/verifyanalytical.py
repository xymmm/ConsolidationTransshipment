"""
verify_analytical.py — Verify Analytical Results Against DP
============================================================

Experiments:
  E17  Symmetric deep dive  (verify Theorems 3 & 4 closed-form thresholds)
  E18  Direct Vw comparison (compare DP value function with analytical Vw)
  E19  Trigger convergence  (single-dispatch approximation quality vs Cf)

Reuses: solver.py (Params, TransshipmentDP)

Usage:
    python verify_analytical.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import math

from solver import Params, TransshipmentDP

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'analytical_verification')
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

COMMON = dict(T=2.0, N=200, lam1=8.0, lam2=5.0, h=0.1,
              I2_max=25, I2_min=-15, b1_max=30)


# ═══════════════════════════════════════════════════════════════════════
#  Analytical Formulas
# ═══════════════════════════════════════════════════════════════════════

def Vw_analytical(I2, b1, tau, h, pi1, pi2, c1, c2, lam1, lam2):
    """Closed-form waiting-region value function (Eq. 17 of analytical PDF)."""
    return ((h + pi2) / (2 * lam2) * I2**2
            + ((h + pi2) / (2 * lam2) - pi2 * tau - c2) * I2
            + (pi1 * tau + c1) * b1
            + (pi1 * lam1 + pi2 * lam2) / 2 * tau**2
            + (c1 * lam1 + c2 * lam2) * tau)


def q_axis(I2, tau, h, pi1, pi2, cu, c1, c2, lam2):
    """Axis of symmetry of the dispatch-quantity parabola (Eq. 20)."""
    return lam2 * ((pi1 - pi2) * tau - cu + c1 - c2) / (h + pi2) + I2 + 0.5


def predicted_floor(tau, h, pi1, pi2, cu, c1, c2, lam2):
    """Analytical safety floor (I2 - q*) from Eq. (floor-formula)."""
    offset = lam2 * ((pi1 - pi2) * tau - cu + c1 - c2) / (h + pi2)
    return max(0, -offset - 0.5)


def I2_bar_symmetric(Cf, cu, h, pi, lam2):
    """Analytical dispatch trigger for symmetric case, Case 2 (b1 >> I2).
    From Theorem 3 (Eq. in Section 5.1) for lambda2*cu/(h+pi) <= 0.5,
    or Theorem 4 (Section 5.2) for > 0.5.
    """
    ratio = lam2 * cu / (h + pi)
    if ratio <= 0.5:
        # Theorem 3, Case 2
        A = 1 - 2 * ratio
        disc = A**2 + 8 * lam2 * Cf / (h + pi)
        return 0.5 * (math.sqrt(disc) - A)
    else:
        # Theorem 4, Case 2
        A = 2 * ratio - 1
        inner_floor = math.floor(ratio - 0.5)  # [lambda2*cu/(h+pi) - 1/2]
        correction = inner_floor * (2 * (ratio - 0.5) - inner_floor)
        disc = A**2 + 4 * (2 * lam2 * Cf / (h + pi) - correction)
        if disc < 0:
            return float('inf')  # never dispatch
        return 0.5 * (A + math.sqrt(disc))


def b1_bar_symmetric(I2, Cf, cu, h, pi, lam2):
    """Analytical dispatch threshold b_bar(I2) for symmetric case, Case 1 (b1 <= I2).
    From Theorem 3/4, Case 1.
    """
    ratio = lam2 * cu / (h + pi)
    A = 2 * I2 + 1 - 2 * ratio
    disc = A**2 - 8 * lam2 * Cf / (h + pi)
    if disc < 0:
        return float('inf')  # never dispatch at this I2
    return 0.5 * (A - math.sqrt(disc))


# ═══════════════════════════════════════════════════════════════════════
#  E17: Symmetric Deep Dive
# ═══════════════════════════════════════════════════════════════════════

def run_E17():
    """Verify Theorems 3 & 4 with multiple symmetric scenarios."""
    print("\n" + "="*70)
    print("  E17: SYMMETRIC DEEP DIVE")
    print("="*70)

    scenarios = OrderedDict()
    for pi in [3, 5, 10, 20]:
        for Cf in [5, 20, 50]:
            for cu in [1, 3]:
                ratio = 5.0 * cu / (0.1 + pi)
                regime = "Thm4" if ratio > 0.5 else "Thm3"
                name = f"pi={pi},Cf={Cf},cu={cu} ({regime})"
                scenarios[name] = Params(
                    **COMMON, Cf=Cf, cu=cu, pi1=pi, pi2=pi,
                    c1=5, c2=5, v2=1)

    results = []
    for name, p in scenarios.items():
        print(f"\n  Solving {name} ...")
        dp = TransshipmentDP(p)
        dp.solve(store_V=True, verbose=False)

        pi = p.pi1
        ratio = p.lam2 * p.cu / (p.h + pi)
        regime = "Thm4" if ratio > 0.5 else "Thm3"

        # Analytical predictions
        I2_bar_theory = I2_bar_symmetric(p.Cf, p.cu, p.h, pi, p.lam2)
        floor_theory = predicted_floor(1.0, p.h, pi, pi, p.cu, p.c1, p.c2, p.lam2)

        # Computational: extract trigger and floor at mid-horizon
        n_mid = p.N // 2
        # Trigger: min I2 such that q* > 0 for some b1
        trigger_comp = None
        for I2 in range(1, p.I2_max + 1):
            dispatched = False
            for b1 in range(1, p.b1_max + 1):
                if dp.get_policy(n_mid, I2, b1) > 0:
                    dispatched = True
                    break
            if dispatched:
                trigger_comp = I2
                break

        # Floor: average I2 - q* at trigger, sweep b1
        floors = []
        for b1 in range(1, 21):
            for I2 in range(1, p.I2_max + 1):
                q = dp.get_policy(n_mid, I2, b1)
                if q > 0:
                    floors.append(I2 - q)
                    break
        floor_comp = np.mean(floors) if floors else np.nan

        # b_bar(I2=10): minimum b1 to trigger dispatch at I2=10
        b_bar_comp = None
        for b1 in range(1, p.b1_max + 1):
            if dp.get_policy(n_mid, 10, b1) > 0:
                b_bar_comp = b1
                break
        b_bar_theory = b1_bar_symmetric(10, p.Cf, p.cu, p.h, pi, p.lam2)

        # Full-clear rate
        n_disp = 0; n_fc = 0
        for I2 in range(1, 21):
            for b1 in range(1, 21):
                q = dp.get_policy(n_mid, I2, b1)
                if q > 0:
                    n_disp += 1
                    if q == min(I2, b1):
                        n_fc += 1
        fc_rate = n_fc / n_disp if n_disp > 0 else np.nan

        row = {
            'scenario': name, 'pi': pi, 'Cf': p.Cf, 'cu': p.cu,
            'regime': regime, 'ratio': round(ratio, 3),
            'I2_bar_theory': round(I2_bar_theory, 2),
            'trigger_comp': trigger_comp,
            'floor_theory': round(floor_theory, 2),
            'floor_comp': round(floor_comp, 2),
            'b_bar_theory_I2eq10': round(b_bar_theory, 2)
                if b_bar_theory != float('inf') else 'inf',
            'b_bar_comp_I2eq10': b_bar_comp,
            'fc_rate': round(fc_rate, 4) if not np.isnan(fc_rate) else np.nan,
        }
        results.append(row)
        print(f"    regime={regime} ratio={ratio:.3f} | "
              f"trigger: theory={I2_bar_theory:.1f} comp={trigger_comp} | "
              f"floor: theory={floor_theory:.1f} comp={floor_comp:.2f} | "
              f"b_bar(I2=10): theory={b_bar_theory:.1f} comp={b_bar_comp}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT, 'E17_symmetric_deep_dive.csv'), index=False)
    print(f"\n  Saved E17 results ({len(df)} scenarios)")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  E18: Direct Vw Comparison
# ═══════════════════════════════════════════════════════════════════════

def run_E18():
    """Compare DP V^n with analytical Vw in the wait region."""
    print("\n" + "="*70)
    print("  E18: DIRECT Vw COMPARISON")
    print("="*70)

    # Use a high-Cf scenario where wait region is large
    test_cases = [
        ("Baseline Cf=20", Params(**COMMON, Cf=20, cu=1, pi1=10, pi2=10, c1=5, c2=5, v2=1)),
        ("High Cf=50",     Params(**COMMON, Cf=50, cu=1, pi1=10, pi2=10, c1=5, c2=5, v2=1)),
        ("Asymmetric",     Params(**COMMON, Cf=20, cu=1, pi1=5, pi2=20, c1=5, c2=5, v2=1)),
    ]

    for case_name, p in test_cases:
        print(f"\n  Solving {case_name} (store_V=True) ...")
        dp = TransshipmentDP(p)
        dp.solve(store_V=True, verbose=False)

        snapshots = {'start': p.N, 'mid': p.N // 2, 'late': p.N // 4, 'near_T': p.N // 10}
        rows = []

        for snap_label, n in snapshots.items():
            tau = n * p.dt
            for I2 in range(1, 21):
                for b1 in range(0, 21):
                    q = dp.get_policy(n, I2, b1)
                    if q == 0:  # Wait region only
                        V_dp = dp.get_value(n, I2, b1)
                        V_an = Vw_analytical(I2, b1, tau, p.h, p.pi1, p.pi2,
                                             p.c1, p.c2, p.lam1, p.lam2)
                        rows.append({
                            'snap': snap_label, 'n': n, 'tau': tau,
                            'I2': I2, 'b1': b1,
                            'V_dp': round(V_dp, 4),
                            'V_analytical': round(V_an, 4),
                            'gap': round(V_dp - V_an, 4),
                            'gap_pct': round((V_dp - V_an) / max(abs(V_an), 1e-6) * 100, 4),
                        })

        df = pd.DataFrame(rows)
        fname = f"E18_Vw_comparison_{case_name.replace(' ', '_')}.csv"
        df.to_csv(os.path.join(OUTPUT, fname), index=False)

        # Summary statistics
        print(f"    {case_name}: {len(df)} wait-region states")
        for snap in snapshots:
            sub = df[df['snap'] == snap]
            if len(sub) > 0:
                print(f"      {snap:8s}: n={len(sub):4d}, "
                      f"mean_gap={sub['gap'].mean():8.2f}, "
                      f"max_|gap|={sub['gap'].abs().max():8.2f}, "
                      f"mean_gap%={sub['gap_pct'].mean():6.2f}%")

        # Plot: V_dp vs V_analytical scatter for mid-horizon
        mid_df = df[df['snap'] == 'mid']
        if len(mid_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax = axes[0]
            ax.scatter(mid_df['V_analytical'], mid_df['V_dp'], s=5, alpha=0.5)
            lims = [min(mid_df['V_analytical'].min(), mid_df['V_dp'].min()),
                    max(mid_df['V_analytical'].max(), mid_df['V_dp'].max())]
            ax.plot(lims, lims, 'r--', lw=1, label='Perfect match')
            ax.set_xlabel('$V_w$ (analytical)')
            ax.set_ylabel('$V^n$ (DP)')
            ax.set_title(f'{case_name}: mid-horizon wait region')
            ax.legend()

            ax = axes[1]
            ax.hist(mid_df['gap'], bins=50, edgecolor='k', alpha=0.7)
            ax.axvline(0, color='r', ls='--')
            ax.set_xlabel('Gap ($V^n_{DP} - V_w$)')
            ax.set_ylabel('Count')
            ax.set_title(f'Gap distribution (mean={mid_df["gap"].mean():.2f})')

            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT, f'E18_scatter_{case_name.replace(" ", "_")}.png'))
            plt.close()

    print(f"\n  E18 complete.")


# ═══════════════════════════════════════════════════════════════════════
#  E19: Trigger Convergence (Single-Dispatch Approximation vs Cf)
# ═══════════════════════════════════════════════════════════════════════

def run_E19():
    """Test whether analytical trigger converges to DP trigger as Cf increases."""
    print("\n" + "="*70)
    print("  E19: TRIGGER CONVERGENCE (analytical vs DP across Cf)")
    print("="*70)

    Cf_values = [2, 5, 10, 20, 50, 100]
    pi = 10  # symmetric
    cu = 1
    h = 0.1

    results = []
    for Cf in Cf_values:
        p = Params(**COMMON, Cf=Cf, cu=cu, pi1=pi, pi2=pi, c1=5, c2=5, v2=1)
        print(f"\n  Solving Cf={Cf} ...")
        dp = TransshipmentDP(p)
        dp.solve(store_V=False, verbose=False)

        I2_bar_theory = I2_bar_symmetric(Cf, cu, h, pi, p.lam2)

        # Extract DP trigger at multiple time snapshots
        snapshots = {'start': p.N, 'mid': p.N // 2, 'late': p.N // 4, 'near_T': p.N // 10}
        for snap_label, n in snapshots.items():
            # Trigger: min I2 at which dispatch occurs for any b1
            trigger = None
            for I2 in range(1, p.I2_max + 1):
                for b1 in range(1, p.b1_max + 1):
                    if dp.get_policy(n, I2, b1) > 0:
                        trigger = I2
                        break
                if trigger is not None:
                    break

            # b_bar at I2=10
            b_bar = None
            for b1 in range(1, p.b1_max + 1):
                if dp.get_policy(n, 10, b1) > 0:
                    b_bar = b1
                    break

            b_bar_theory = b1_bar_symmetric(10, Cf, cu, h, pi, p.lam2)

            results.append({
                'Cf': Cf, 'snap': snap_label,
                'I2_bar_theory': round(I2_bar_theory, 2),
                'I2_trigger_DP': trigger,
                'trigger_gap': trigger - I2_bar_theory if trigger else None,
                'b_bar_theory': round(b_bar_theory, 2) if b_bar_theory != float('inf') else 'inf',
                'b_bar_DP': b_bar,
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT, 'E19_trigger_convergence.csv'), index=False)

    # Plot: trigger gap vs Cf at mid-horizon
    mid = df[df['snap'] == 'mid'].copy()
    mid['trigger_gap'] = pd.to_numeric(mid['trigger_gap'], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(mid['Cf'], mid['I2_bar_theory'], 'o--', label='Analytical $\\bar{I}_2$', ms=8)
    ax.plot(mid['Cf'], mid['I2_trigger_DP'], 's-', label='DP trigger', ms=8)
    ax.set_xlabel('$C_f$')
    ax.set_ylabel('Dispatch trigger ($I_2$)')
    ax.set_title('Dispatch Trigger: Analytical vs DP (mid-horizon)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(mid['Cf'], mid['trigger_gap'], 'o-', color='red', ms=8)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('$C_f$')
    ax.set_ylabel('Gap (DP trigger $-$ Analytical)')
    ax.set_title('Single-Dispatch Approximation Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'E19_trigger_convergence.png'))
    plt.close()

    print(f"\n  E19 complete. Results:")
    print(mid[['Cf', 'I2_bar_theory', 'I2_trigger_DP', 'trigger_gap']].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
#  E20: Time-independence of trigger (analytical says tau-free for symmetric)
# ═══════════════════════════════════════════════════════════════════════

def run_E20():
    """Check if DP trigger varies with tau (analytical predicts time-independent for symmetric)."""
    print("\n" + "="*70)
    print("  E20: TIME-DEPENDENCE OF TRIGGER (symmetric case)")
    print("="*70)

    test_cases = [
        ("pi=10,Cf=20", Params(**COMMON, Cf=20, cu=1, pi1=10, pi2=10, c1=5, c2=5, v2=1)),
        ("pi=10,Cf=50", Params(**COMMON, Cf=50, cu=1, pi1=10, pi2=10, c1=5, c2=5, v2=1)),
        ("pi=5,Cf=20",  Params(**COMMON, Cf=20, cu=1, pi1=5, pi2=5, c1=5, c2=5, v2=1)),
    ]

    fig, axes = plt.subplots(1, len(test_cases), figsize=(6*len(test_cases), 5))
    if len(test_cases) == 1:
        axes = [axes]

    for idx, (case_name, p) in enumerate(test_cases):
        print(f"\n  Solving {case_name} ...")
        dp = TransshipmentDP(p)
        dp.solve(store_V=False, verbose=False)

        pi = p.pi1
        I2_bar_theory = I2_bar_symmetric(p.Cf, p.cu, p.h, pi, p.lam2)

        # Extract trigger at many time points
        time_steps = list(range(10, p.N + 1, 5))
        taus = [n * p.dt for n in time_steps]
        triggers = []
        for n in time_steps:
            trigger = None
            for I2 in range(1, p.I2_max + 1):
                for b1 in range(1, p.b1_max + 1):
                    if dp.get_policy(n, I2, b1) > 0:
                        trigger = I2
                        break
                if trigger is not None:
                    break
            triggers.append(trigger)

        ax = axes[idx]
        ax.plot(taus, triggers, 'b-', lw=2, label='DP trigger')
        ax.axhline(I2_bar_theory, color='r', ls='--', lw=2,
                   label=f'Analytical $\\bar{{I}}_2$ = {I2_bar_theory:.1f}')
        ax.set_xlabel('Remaining time $\\tau$')
        ax.set_ylabel('Dispatch trigger (min $I_2$ with $q^*>0$)')
        ax.set_title(case_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT, 'E20_trigger_vs_time.png'))
    plt.close()
    print("\n  E20 complete.")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*70)
    print("  ANALYTICAL VERIFICATION EXPERIMENTS")
    print("="*70)

    df17 = run_E17()
    run_E18()
    run_E19()
    run_E20()

    print("\n" + "="*70)
    print(f"  ALL DONE. Outputs in {OUTPUT}/")
    print("="*70)