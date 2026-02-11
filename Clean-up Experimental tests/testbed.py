"""
testbed.py — Comprehensive Experimental Test Bed
=================================================

Extended Model with End-of-Horizon Clean-up (Section 3).

Design principle:
  - Each experiment group (E1–E16) sweeps ONE parameter while holding others
    at a baseline OR at a deliberately zeroed-out value to ISOLATE effects.
  - Zero-parameter cases reveal which cost component drives the structure.
  - Interaction groups test joint effects.

Pipeline:
  Part 0  Define scenario grid + experiment groups
  Part 1  Solve all (backward induction)
  Part 2  Simulation validation (DP vs Monte Carlo)
  Part 3  Q1 — dispatch threshold / quantity / safety vs TIME (per group)
  Part 4  Q2 — two thresholds in I₂ dimension (per group)
  Part 5  Policy heatmaps (per group)
  Part 6  Cost summary bar charts (per group)
  Part 7  Full-clear rate analysis

Usage:
    python testbed.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import time as _time

from solver     import Params, TransshipmentDP
from simulation import validate

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


# ═══════════════════════════════════════════════════════════════════════
#  Part 0 — Scenario Grid
# ═══════════════════════════════════════════════════════════════════════
def build_scenarios():
    """
    Returns
    -------
    S : OrderedDict[str, Params]    all scenarios (name -> params)
    G : OrderedDict[str, list[str]] experiment groups (group -> scenario names)

    BASELINE = T=2, N=200, lam1=8, lam2=5, h=0.1, Cf=20, cu=1,
               pi1=10, pi2=10, c1=5, c2=5, v2=1

    E1-E8:   Single-parameter sweeps (each includes 0)
    E9-E11:  Isolation (turn off entire cost layers)
    E12-E16: Interactions and degenerate cases
    """
    common = dict(T=2.0, N=200, lam1=8.0, lam2=5.0, h=0.1, cu=1.0,
                  I2_max=25, I2_min=-15, b1_max=30)

    S = OrderedDict()
    G = OrderedDict()

    # ══════════════════════════════════════════════════════════════
    #  SINGLE-PARAMETER SWEEPS
    # ══════════════════════════════════════════════════════════════

    # ── E1. Fixed cost Cf ──
    # Cf=0 removes consolidation incentive -> dispatch every period
    # Cf large -> deep consolidation (wait longer)
    g = 'E1_Cf_sweep'
    G[g] = []
    for Cf in [0, 2, 5, 10, 20, 50]:
        name = f'E1: Cf={Cf}'
        S[name] = Params(**common, Cf=Cf, pi1=10, pi2=10, c1=5, c2=5, v2=1)
        G[g].append(name)

    # ── E2. Unit transshipment cost cu ──
    # cu=0 -> free variable cost; only Cf governs shipment economics
    # cu large -> per-unit penalty for dispatching
    g = 'E2_cu_sweep'
    G[g] = []
    for cu in [0, 0.5, 1, 3, 5]:
        name = f'E2: cu={cu}'
        S[name] = Params(**{**common, 'cu': cu},
                         Cf=20, pi1=10, pi2=10, c1=5, c2=5, v2=1)
        G[g].append(name)

    # ── E3. pi1 sweep (R1 backlog penalty) with pi2=0 ──
    # Isolate R1 urgency: pi2=0 means R2 stockout free -> no safety motive
    g = 'E3_pi1_sweep_pi2zero'
    G[g] = []
    for pi1 in [0, 5, 10, 20, 30]:
        name = f'E3: pi1={pi1} (pi2=0)'
        S[name] = Params(**common, Cf=20, pi1=pi1, pi2=0, c1=5, c2=5, v2=1)
        G[g].append(name)

    # ── E4. pi2 sweep (R2 stockout penalty) ──
    # pi2=0 -> no safety-stock motive
    # pi2 large -> strong safety floor (retain inventory for R2)
    g = 'E4_pi2_sweep'
    G[g] = []
    for pi2 in [0, 5, 10, 20, 30]:
        name = f'E4: pi2={pi2}'
        S[name] = Params(**common, Cf=20, pi1=10, pi2=pi2, c1=5, c2=5, v2=1)
        G[g].append(name)

    # ── E5. c1 sweep (terminal cost for R1 backlog) ──
    # c1=0 -> no end-of-horizon pressure on R1 backlog
    g = 'E5_c1_sweep'
    G[g] = []
    for c1 in [0, 2, 5, 10, 20, 30]:
        name = f'E5: c1={c1}'
        S[name] = Params(**common, Cf=20, pi1=10, pi2=10, c1=c1, c2=5, v2=1)
        G[g].append(name)

    # ── E6. c2 sweep (terminal cost for R2 backlog) ──
    # c2=0 -> R2 terminal stockout free -> no terminal retention motive
    # Note: v2 <= c2 required
    g = 'E6_c2_sweep'
    G[g] = []
    for c2 in [0, 2, 5, 10, 20]:
        v2_val = min(1, c2)
        name = f'E6: c2={c2}'
        S[name] = Params(**common, Cf=20, pi1=10, pi2=10, c1=5, c2=c2, v2=v2_val)
        G[g].append(name)

    # ── E7. v2 sweep (salvage value) ──
    # v2=0 -> no salvage incentive to retain I2
    # v2 -> c2 -> maximum retention incentive
    # Use c2=10 to allow v2 range
    g = 'E7_v2_sweep'
    G[g] = []
    for v2 in [0, 1, 3, 5, 9]:
        name = f'E7: v2={v2}'
        S[name] = Params(**common, Cf=20, pi1=10, pi2=10, c1=5, c2=10, v2=v2)
        G[g].append(name)

    # ── E8. Holding cost h ──
    # h=0 -> no cost to keep inventory -> encourages retention
    # h large -> penalises hoarding
    g = 'E8_h_sweep'
    G[g] = []
    for h in [0, 0.1, 0.5, 1.0, 2.0]:
        name = f'E8: h={h}'
        S[name] = Params(**{**common, 'h': h},
                         Cf=20, pi1=10, pi2=10, c1=5, c2=5, v2=1)
        G[g].append(name)

    # ══════════════════════════════════════════════════════════════
    #  ISOLATION EXPERIMENTS: turn off entire cost layers
    # ══════════════════════════════════════════════════════════════

    # ── E9. Per-period penalty ONLY (h=0, c1=c2=v2=0) ──
    # Pure flow-cost-driven: only pi1, pi2, Cf matter
    g = 'E9_penalty_only'
    G[g] = []
    for pi1, pi2 in [(10, 0), (10, 10), (10, 20), (20, 10), (0, 10)]:
        name = f'E9: pi1={pi1},pi2={pi2}'
        S[name] = Params(**{**common, 'h': 0},
                         Cf=20, c1=0, c2=0, v2=0, pi1=pi1, pi2=pi2)
        G[g].append(name)

    # ── E10. Terminal ONLY (pi1=pi2=h=Cf=cu=0) ──
    # All dynamics from V^0; no consolidation friction, no per-period costs
    g = 'E10_terminal_only'
    G[g] = []
    for c1, c2, v2 in [(5, 0, 0), (5, 5, 0), (5, 5, 4),
                        (10, 5, 1), (20, 10, 5), (0, 10, 5)]:
        name = f'E10: c1={c1},c2={c2},v2={v2}'
        S[name] = Params(**{**common, 'cu': 0, 'h': 0},
                         Cf=0, pi1=0, pi2=0, c1=c1, c2=c2, v2=v2)
        G[g].append(name)

    # ── E11. Terminal + consolidation friction (Cf > 0) ──
    # No per-period costs, but Cf forces batching
    g = 'E11_terminal_with_Cf'
    G[g] = []
    for c1 in [5, 10, 20]:
        for Cf in [5, 20]:
            name = f'E11: c1={c1},Cf={Cf}'
            S[name] = Params(**{**common, 'h': 0},
                             Cf=Cf, pi1=0, pi2=0, c1=c1, c2=5, v2=1)
            G[g].append(name)

    # ══════════════════════════════════════════════════════════════
    #  INTERACTION EXPERIMENTS
    # ══════════════════════════════════════════════════════════════

    # ── E12. Penalty asymmetry grid ──
    # How does pi1/pi2 ratio shape the two thresholds?
    g = 'E12_penalty_asymmetry'
    G[g] = []
    for pi1, pi2 in [(5, 5), (10, 5), (5, 10), (10, 10),
                      (20, 5), (5, 20), (20, 10), (10, 20),
                      (30, 5), (5, 30), (20, 20), (30, 10)]:
        name = f'E12: pi1={pi1},pi2={pi2}'
        S[name] = Params(**common, Cf=20, pi1=pi1, pi2=pi2, c1=5, c2=5, v2=1)
        G[g].append(name)

    # ── E13. c1 vs c2 interaction ──
    # Terminal cost balance: when one is 0, only the other matters
    g = 'E13_c1_vs_c2'
    G[g] = []
    for c1, c2 in [(0, 0), (5, 0), (0, 5), (5, 5), (10, 5),
                    (5, 10), (10, 10), (20, 5), (5, 20), (20, 20)]:
        v2_val = min(1, c2)
        name = f'E13: c1={c1},c2={c2}'
        S[name] = Params(**common, Cf=20, pi1=10, pi2=10,
                         c1=c1, c2=c2, v2=v2_val)
        G[g].append(name)

    # ── E14. Cf x pi1 interaction ──
    # Consolidation depth vs backlog urgency
    g = 'E14_Cf_x_pi1'
    G[g] = []
    for Cf in [0, 5, 20, 50]:
        for pi1 in [5, 10, 20]:
            name = f'E14: Cf={Cf},pi1={pi1}'
            S[name] = Params(**common, Cf=Cf, pi1=pi1, pi2=10,
                             c1=5, c2=5, v2=1)
            G[g].append(name)

    # ── E15. v2 x pi2 interaction ──
    # Both create retention incentive — additive or not?
    g = 'E15_v2_x_pi2'
    G[g] = []
    for v2 in [0, 3, 8]:
        for pi2 in [0, 10, 20]:
            name = f'E15: v2={v2},pi2={pi2}'
            S[name] = Params(**common, Cf=20, pi1=10, pi2=pi2,
                             c1=5, c2=10, v2=v2)
            G[g].append(name)

    # ── E16. Degenerate / boundary cases ──
    g = 'E16_degenerate'
    G[g] = []
    configs = [
        ('All zero costs',
         dict(Cf=0, cu=0, h=0, pi1=0, pi2=0, c1=0, c2=0, v2=0)),
        ('Only Cf=20',
         dict(Cf=20, cu=0, h=0, pi1=0, pi2=0, c1=0, c2=0, v2=0)),
        ('Only pi1=10,pi2=10',
         dict(Cf=0, cu=0, h=0, pi1=10, pi2=10, c1=0, c2=0, v2=0)),
        ('Only c1=10,c2=10',
         dict(Cf=0, cu=0, h=0, pi1=0, pi2=0, c1=10, c2=10, v2=0)),
        ('Only h=0.1,v2=5',
         dict(Cf=0, cu=0, h=0.1, pi1=0, pi2=0, c1=0, c2=5, v2=5)),
        ('Balanced baseline',
         dict(Cf=20, cu=1, h=0.1, pi1=10, pi2=10, c1=5, c2=5, v2=1)),
        ('All high',
         dict(Cf=50, cu=3, h=1.0, pi1=30, pi2=20, c1=20, c2=15, v2=5)),
    ]
    for label, kw in configs:
        name = f'E16: {label}'
        base = {**common}
        base.update(kw)
        S[name] = Params(**base)
        G[g].append(name)

    return S, G


# ═══════════════════════════════════════════════════════════════════════
#  Part 1 — Solve
# ═══════════════════════════════════════════════════════════════════════
def solve_all(scenarios):
    solvers = {}
    total = len(scenarios)
    for i, (name, params) in enumerate(scenarios.items()):
        print(f"  [{i+1}/{total}] {name}")
        dp = TransshipmentDP(params)
        dp.solve(store_V=True, verbose=False)
        solvers[name] = dp
    return solvers


# ═══════════════════════════════════════════════════════════════════════
#  Part 2 — Validation
# ═══════════════════════════════════════════════════════════════════════
def run_validation(scenarios, solvers, groups, I2_ref=15, b1_ref=0, n_sims=5000):
    rows = []
    for name, dp in solvers.items():
        p = scenarios[name]
        r = validate(dp, I2_ref, b1_ref, n_sims=n_sims)
        rows.append({
            'Scenario': name,
            'Cf': p.Cf, 'cu': p.cu, 'h': p.h,
            'pi1': p.pi1, 'pi2': p.pi2,
            'c1': p.c1, 'c2': p.c2, 'v2': p.v2,
            'DP Cost':   round(r['dp_cost'], 2),
            'Sim Mean':  round(r['mean'], 2),
            'Sim Std':   round(r['std'], 2),
            '95% CI Lo': round(r['ci95_lo'], 2),
            '95% CI Hi': round(r['ci95_hi'], 2),
            'Gap(%)':    round(r['gap_pct'], 3),
            'In CI':     'Y' if r['in_ci'] else 'N',
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT, 'validation_table.csv'), index=False)

    print("\n" + "=" * 120)
    print("VALIDATION:  DP Cost vs Monte Carlo Simulation")
    print("=" * 120)
    print(df.to_string(index=False))
    n_ok = (df['In CI'] == 'Y').sum()
    print(f"\nResult: {n_ok}/{len(df)} scenarios pass (DP cost within 95% CI)")
    print("=" * 120)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════
def _safe(name: str) -> str:
    for old, new in [(' ', '_'), (':', ''), ('=', ''), (',', '_'),
                      ('(', ''), (')', ''), ('+', 'p'), ('>', 'gt')]:
        name = name.replace(old, new)
    return name

_COLORS = [
    '#2563eb', '#dc2626', '#059669', '#9333ea', '#ea580c',
    '#0891b2', '#be185d', '#78716c', '#4f46e5', '#16a34a',
    '#d97706', '#7c3aed', '#0d9488', '#e11d48', '#475569',
    '#c026d3', '#65a30d', '#0284c7', '#b91c1c', '#047857',
]

def _color(idx):
    return _COLORS[idx % len(_COLORS)]

def _short(name):
    """Strip the 'EX: ' prefix for legend labels."""
    return name.split(': ', 1)[-1] if ': ' in name else name


# ═══════════════════════════════════════════════════════════════════════
#  Part 3 — Q1: Threshold / quantity / safety vs TIME (per group)
# ═══════════════════════════════════════════════════════════════════════
def _extract_time_profiles(dp, I2_values):
    p = dp.p
    out = {}
    for I2 in I2_values:
        b_bar  = np.full(p.N + 1, np.nan)
        q_bar  = np.full(p.N + 1, np.nan)
        safety = np.full(p.N + 1, np.nan)
        for n in range(1, p.N + 1):
            for b1 in range(1, p.b1_max + 1):
                q = dp.get_policy(n, I2, b1)
                if q > 0:
                    b_bar[n]  = b1
                    q_bar[n]  = q
                    safety[n] = I2 - q
                    break
        out[I2] = {'b_bar': b_bar, 'q_bar': q_bar, 'safety': safety}
    return out


def plot_time_analysis_by_group(scenarios, solvers, groups):
    I2_values = [5, 10, 15, 20]

    for gname, names in groups.items():
        active = [n for n in names if n in solvers]
        if not active:
            continue

        profiles = {}
        for name in active:
            profiles[name] = _extract_time_profiles(solvers[name], I2_values)

        for metric, ylabel, suffix, key in [
            ('Dispatch threshold $\\bar{b}_1$', '$\\bar{b}_1$',
             'threshold', 'b_bar'),
            ('Dispatch quantity $q^*$', '$q^*$ at threshold',
             'quantity', 'q_bar'),
            ('Safety stock $I_2 - q^*$', 'Retained $I_2 - q^*$',
             'safety', 'safety'),
        ]:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            for ax_i, I2 in enumerate(I2_values):
                ax = axes.flat[ax_i]
                has_data = False
                for ci, name in enumerate(active):
                    p   = scenarios[name]
                    tau = np.arange(p.N + 1) * p.dt
                    y   = profiles[name][I2][key]
                    mask = ~np.isnan(y)
                    if mask.any():
                        ax.plot(tau[mask], y[mask], '-', color=_color(ci),
                                label=_short(name), linewidth=1.5, alpha=0.85)
                        has_data = True

                ax.set_xlabel('Remaining time $\\tau$')
                ax.set_ylabel(ylabel)
                ax.set_title(f'$I_2 = {I2}$')
                ax.invert_xaxis()
                ax.grid(True, alpha=0.3)
                if has_data:
                    ax.legend(fontsize=7, loc='best')

            fig.suptitle(f'{gname}: {metric} vs Time', fontsize=13, y=1.01)
            plt.tight_layout()
            fname = f'Q1_{_safe(gname)}_{suffix}.png'
            fig.savefig(os.path.join(OUTPUT, fname))
            plt.close()
            print(f"    {fname}")


# ═══════════════════════════════════════════════════════════════════════
#  Part 4 — Q2: Two thresholds in I₂ (per group)
# ═══════════════════════════════════════════════════════════════════════
def plot_inventory_analysis_by_group(scenarios, solvers, groups):
    N = 200
    n_mid = N // 2
    b1_panels = [3, 5, 8, 12]
    b1_sweep  = np.arange(1, 21)

    for gname, names in groups.items():
        active = [n for n in names if n in solvers]
        if not active:
            continue

        # ── Q2a: q* vs I2 at mid-horizon ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax_i, b1 in enumerate(b1_panels):
            ax = axes.flat[ax_i]
            I2r = np.arange(1, scenarios[active[0]].I2_max + 1)

            for ci, name in enumerate(active):
                dp = solvers[name]
                qs = np.array([dp.get_policy(n_mid, int(I2), b1) for I2 in I2r])
                ax.plot(I2r, qs, 'o-', color=_color(ci), label=_short(name),
                        markersize=3, linewidth=1.5)

            ax.plot(I2r, np.minimum(I2r, b1), '--', color='gray',
                    alpha=0.4, label='Full clear')
            ax.set_xlabel('$I_2$')
            ax.set_ylabel('$q^*$')
            ax.set_title(f'$b_1={b1}$')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{gname}: $q^*$ vs $I_2$ (mid-horizon)', fontsize=13, y=1.01)
        plt.tight_layout()
        fname = f'Q2a_{_safe(gname)}_qstar.png'
        fig.savefig(os.path.join(OUTPUT, fname))
        plt.close()
        print(f"    {fname}")

        # ── Q2b: Dispatch trigger + Safety floor vs b1 ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        for ci, name in enumerate(active):
            dp = solvers[name]
            p  = scenarios[name]
            I2_thresh    = []
            safety_floor = []

            for b1 in b1_sweep:
                found = False
                for I2 in range(1, p.I2_max + 1):
                    q = dp.get_policy(n_mid, I2, b1)
                    if q > 0:
                        I2_thresh.append(I2)
                        safety_floor.append(I2 - q)
                        found = True
                        break
                if not found:
                    I2_thresh.append(np.nan)
                    safety_floor.append(np.nan)

            c = _color(ci)
            I2_arr = np.array(I2_thresh, dtype=float)
            sf_arr = np.array(safety_floor, dtype=float)

            mask = ~np.isnan(I2_arr)
            if mask.any():
                axes[0].plot(b1_sweep[mask], I2_arr[mask], 'o-', color=c,
                             label=_short(name), markersize=4)
            mask2 = ~np.isnan(sf_arr)
            if mask2.any():
                axes[1].plot(b1_sweep[mask2], sf_arr[mask2], 's-', color=c,
                             label=_short(name), markersize=4)

        axes[0].set_xlabel('$b_1$')
        axes[0].set_ylabel('Min $I_2$ to dispatch')
        axes[0].set_title('Threshold 1: Dispatch Trigger')
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('$b_1$')
        axes[1].set_ylabel('$I_2 - q^*$ at trigger')
        axes[1].set_title('Threshold 2: Safety Floor')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f'{gname}: Two Thresholds (mid-horizon)', fontsize=13, y=1.03)
        plt.tight_layout()
        fname = f'Q2b_{_safe(gname)}_thresholds.png'
        fig.savefig(os.path.join(OUTPUT, fname))
        plt.close()
        print(f"    {fname}")


# ═══════════════════════════════════════════════════════════════════════
#  Part 5 — Heatmaps: contrast within each group
# ═══════════════════════════════════════════════════════════════════════
def plot_heatmaps(scenarios, solvers, groups):
    N = 200
    snaps = OrderedDict([('Start (n=N)', N), ('Mid (n=N/2)', N // 2)])
    I2_range = np.arange(1, 21)
    b1_range = np.arange(1, 21)

    for gname, names in groups.items():
        active = [n for n in names if n in solvers]
        if len(active) < 2:
            continue

        # Pick contrasting scenarios: first, middle, last
        picks = [active[0], active[-1]]
        if len(active) >= 4:
            picks.insert(1, active[len(active) // 2])

        n_picks = len(picks)
        fig, axes = plt.subplots(n_picks, 2, figsize=(12, 5 * n_picks))
        if n_picks == 1:
            axes = axes.reshape(1, -1)

        for row, name in enumerate(picks):
            dp = solvers[name]
            for col, (tname, n) in enumerate(snaps.items()):
                ax = axes[row, col]
                heatmap = np.zeros((len(b1_range), len(I2_range)))
                for i, b1 in enumerate(b1_range):
                    for j, I2 in enumerate(I2_range):
                        heatmap[i, j] = dp.get_policy(n, int(I2), int(b1))

                im = ax.imshow(heatmap, aspect='auto', cmap='Blues',
                               origin='lower',
                               extent=[I2_range[0] - 0.5, I2_range[-1] + 0.5,
                                       b1_range[0] - 0.5, b1_range[-1] + 0.5])
                ax.set_xlabel('$I_2$')
                ax.set_ylabel('$b_1$')
                ax.set_title(f'{_short(name)}  |  {tname}')
                plt.colorbar(im, ax=ax, label='$q^*$', shrink=0.85)

        fig.suptitle(f'{gname}: Policy Heatmaps', fontsize=14, y=1.01)
        plt.tight_layout()
        fname = f'heatmap_{_safe(gname)}.png'
        fig.savefig(os.path.join(OUTPUT, fname))
        plt.close()
        print(f"    {fname}")


# ═══════════════════════════════════════════════════════════════════════
#  Part 6 — Cost summary bar charts
# ═══════════════════════════════════════════════════════════════════════
def plot_cost_summary(scenarios, solvers, groups, I2_ref=15, b1_ref=0):
    for gname, names in groups.items():
        active = [n for n in names if n in solvers]
        if not active:
            continue

        labels = [_short(n) for n in active]
        costs  = [solvers[n].get_value(scenarios[n].N, I2_ref, b1_ref)
                  for n in active]

        fig, ax = plt.subplots(figsize=(max(8, len(active) * 1.5), 5))
        bars = ax.bar(range(len(active)), costs,
                      color=[_color(i) for i in range(len(active))],
                      edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(active)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Expected Total Cost $V^N(15,0)$')
        ax.set_title(f'{gname}: Cost Comparison')
        ax.grid(True, axis='y', alpha=0.3)

        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{cost:.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fname = f'cost_{_safe(gname)}.png'
        fig.savefig(os.path.join(OUTPUT, fname))
        plt.close()
        print(f"    {fname}")


# ═══════════════════════════════════════════════════════════════════════
#  Part 7 — Full-clear rate analysis
# ═══════════════════════════════════════════════════════════════════════
def compute_full_clear_rates(scenarios, solvers, groups):
    N = 200
    n_mid = N // 2
    rows = []

    for name, dp in solvers.items():
        p = scenarios[name]
        total_disp = 0
        full_clear = 0
        partial_qs = []

        for I2 in range(1, min(21, p.I2_max + 1)):
            for b1 in range(1, min(21, p.b1_max + 1)):
                q = dp.get_policy(n_mid, I2, b1)
                if q > 0:
                    total_disp += 1
                    q_full = min(I2, b1)
                    if q == q_full:
                        full_clear += 1
                    else:
                        partial_qs.append(q_full - q)

        fc_rate = full_clear / total_disp if total_disp > 0 else np.nan
        avg_shortfall = np.mean(partial_qs) if partial_qs else 0.0

        rows.append({
            'Scenario':        name,
            'Dispatch states':  total_disp,
            'Full clear':       full_clear,
            'Partial':          total_disp - full_clear,
            'FC rate':          round(fc_rate, 4) if not np.isnan(fc_rate) else 'N/A',
            'Avg shortfall':    round(avg_shortfall, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT, 'full_clear_rates.csv'), index=False)

    print("\n" + "=" * 100)
    print("FULL-CLEAR RATE (mid-horizon, 20x20 state grid)")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Part 8 — Per-scenario policy export: one CSV per scenario
# ═══════════════════════════════════════════════════════════════════════
def export_policy_detail(scenarios, solvers, groups,
                         I2_range=None, b1_range=None, time_snapshots=None):
    """
    Export one CSV per scenario into outputs/policies/.

    Each CSV has one row per (time_snapshot, I2, b1) with columns:
        snap, n, tau, I2, b1,
        q_star, safety_stock, is_dispatch, is_full_clear, q_max_feasible

    A header comment line records the scenario parameters.
    """
    if I2_range is None:
        I2_range = list(range(1, 21))
    if b1_range is None:
        b1_range = list(range(0, 21))

    N = 200
    if time_snapshots is None:
        time_snapshots = OrderedDict([
            ('start',   N),
            ('mid',     N // 2),
            ('late',    N // 4),
            ('near_T',  N // 10),
        ])

    policy_dir = os.path.join(OUTPUT, 'policies')
    os.makedirs(policy_dir, exist_ok=True)

    total = len(solvers)
    for idx, (name, dp) in enumerate(solvers.items()):
        if (idx + 1) % 20 == 0 or idx + 1 == total or idx == 0:
            print(f"    [{idx+1}/{total}] {name}")

        p = scenarios[name]
        rows = []
        for snap_label, n in time_snapshots.items():
            tau = round(n * p.dt, 4)
            for I2 in I2_range:
                for b1 in b1_range:
                    q = dp.get_policy(n, I2, b1)
                    q_max = max(0, min(I2, b1)) if (I2 > 0 and b1 > 0) else 0
                    rows.append({
                        'snap':           snap_label,
                        'n':              n,
                        'tau':            tau,
                        'I2':             I2,
                        'b1':             b1,
                        'q_star':         q,
                        'safety_stock':   I2 - q,
                        'is_dispatch':    int(q > 0),
                        'is_full_clear':  int(q > 0 and q == q_max),
                        'q_max_feasible': q_max,
                    })

        df = pd.DataFrame(rows)

        # Build safe filename
        fname = _safe(name) + '.csv'
        fpath = os.path.join(policy_dir, fname)

        # Write parameter header as comment, then CSV
        with open(fpath, 'w') as f:
            f.write(f"# scenario: {name}\n")
            f.write(f"# Cf={p.Cf} cu={p.cu} h={p.h} "
                    f"pi1={p.pi1} pi2={p.pi2} "
                    f"c1={p.c1} c2={p.c2} v2={p.v2}\n")
            f.write(f"# T={p.T} N={p.N} lam1={p.lam1} lam2={p.lam2}\n")
            df.to_csv(f, index=False)

    print(f"    Saved {total} policy CSVs to {policy_dir}/")


# ═══════════════════════════════════════════════════════════════════════
#  Part 9 — Scenario-level summary: one row per scenario
# ═══════════════════════════════════════════════════════════════════════
def export_scenario_summary(scenarios, solvers, groups,
                            I2_ref=15, b1_ref=0):
    """
    One row per scenario with:
      - All parameter values
      - DP cost at reference state V^N(I2_ref, b1_ref)
      - At each time snapshot (start, mid, late, near_T):
          - dispatch threshold b_bar(I2=10)
          - full-clear rate over 20x20 grid
          - average safety floor at dispatch trigger (b1=1..20)
    """
    N = 200
    snaps = OrderedDict([
        ('start', N), ('mid', N // 2), ('late', N // 4), ('near_T', N // 10),
    ])

    scen_to_group = {}
    for gname, names in groups.items():
        for n in names:
            scen_to_group[n] = gname

    rows = []
    for name, dp in solvers.items():
        p = scenarios[name]
        row = {
            'scenario':  name,
            'group':     scen_to_group.get(name, ''),
            'Cf': p.Cf, 'cu': p.cu, 'h': p.h,
            'pi1': p.pi1, 'pi2': p.pi2,
            'c1': p.c1, 'c2': p.c2, 'v2': p.v2,
            'dp_cost':   round(dp.get_value(p.N, I2_ref, b1_ref), 4),
        }

        for snap_label, n in snaps.items():
            # Dispatch threshold at I2=10
            b_bar = np.nan
            for b1 in range(1, p.b1_max + 1):
                if dp.get_policy(n, 10, b1) > 0:
                    b_bar = b1
                    break
            row[f'b_bar_I2eq10_{snap_label}'] = b_bar

            # Full-clear rate over 20x20 grid
            n_disp = 0
            n_fc   = 0
            for I2 in range(1, 21):
                for b1 in range(1, 21):
                    q = dp.get_policy(n, I2, b1)
                    if q > 0:
                        n_disp += 1
                        if q == min(I2, b1):
                            n_fc += 1
            row[f'fc_rate_{snap_label}'] = round(n_fc / n_disp, 4) if n_disp > 0 else np.nan
            row[f'n_dispatch_states_{snap_label}'] = n_disp

            # Average safety floor at dispatch trigger (sweep b1=1..20)
            floors = []
            for b1 in range(1, 21):
                for I2 in range(1, p.I2_max + 1):
                    q = dp.get_policy(n, I2, b1)
                    if q > 0:
                        floors.append(I2 - q)
                        break
            row[f'avg_safety_floor_{snap_label}'] = (
                round(np.mean(floors), 2) if floors else np.nan
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    fpath = os.path.join(OUTPUT, 'scenario_summary.csv')
    df.to_csv(fpath, index=False)
    print(f"    Saved scenario_summary.csv  ({len(df)} rows)")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_start = _time.time()

    print("=" * 70)
    print("  TRANSSHIPMENT DP - COMPREHENSIVE EXPERIMENTAL TEST BED")
    print("  Extended Model with End-of-Horizon Clean-up")
    print("=" * 70)

    # Part 0
    scenarios, groups = build_scenarios()
    print(f"\n{len(scenarios)} scenarios in {len(groups)} groups:")
    for gname, names in groups.items():
        print(f"  {gname}: {len(names)} scenarios")

    # Part 1: Solve
    print("\n" + "-" * 70)
    print("PART 1: Solving all scenarios")
    print("-" * 70)
    solvers = solve_all(scenarios)

    # Part 2: Validation
    print("\n" + "-" * 70)
    print("PART 2: Simulation Validation")
    print("-" * 70)
    df_val = run_validation(scenarios, solvers, groups, n_sims=5000)

    # Part 3: Q1 - Time thresholds
    print("\n" + "-" * 70)
    print("PART 3: Q1 - Time-dimension analysis (per group)")
    print("-" * 70)
    plot_time_analysis_by_group(scenarios, solvers, groups)

    # Part 4: Q2 - Inventory thresholds
    print("\n" + "-" * 70)
    print("PART 4: Q2 - Inventory-dimension analysis (per group)")
    print("-" * 70)
    plot_inventory_analysis_by_group(scenarios, solvers, groups)

    # Part 5: Heatmaps
    print("\n" + "-" * 70)
    print("PART 5: Policy Heatmaps (per group)")
    print("-" * 70)
    plot_heatmaps(scenarios, solvers, groups)

    # Part 6: Cost summary
    print("\n" + "-" * 70)
    print("PART 6: Cost Summary (per group)")
    print("-" * 70)
    plot_cost_summary(scenarios, solvers, groups)

    # Part 7: Full-clear rates
    print("\n" + "-" * 70)
    print("PART 7: Full-Clear Rate Analysis")
    print("-" * 70)
    df_fc = compute_full_clear_rates(scenarios, solvers, groups)

    # Part 8: Full policy detail CSV
    print("\n" + "-" * 70)
    print("PART 8: Policy Detail Export (q* for every scenario/state)")
    print("-" * 70)
    df_pol = export_policy_detail(scenarios, solvers, groups)

    # Part 9: Scenario-level summary CSV
    print("\n" + "-" * 70)
    print("PART 9: Scenario Summary Export")
    print("-" * 70)
    df_sum = export_scenario_summary(scenarios, solvers, groups)

    elapsed = _time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE - {len(scenarios)} scenarios, {elapsed:.1f}s total")
    print(f"  All outputs in {OUTPUT}/")
    print(f"{'=' * 70}")