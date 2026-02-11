"""
simulation.py — Monte Carlo Simulation for the Transshipment Model
==================================================================

Two modes:
  1. validate():   run n_sims paths, return aggregate cost statistics
                   for comparison with DP expected cost.
  2. trace():      run a single path, record the full trajectory
                   {time, I₂, b₁, q*, event} for detailed inspection.

Both follow the EXACT same dynamics as the DP:
  - At each period n (counting down from N to 1):
      (a) observe state (I₂, b₁)
      (b) look up action q* = policy(n, I₂, b₁)
      (c) pay one-period cost g(I₂, b₁, q*)
      (d) update state: I₂ ← I₂−q, b₁ ← b₁−q
      (e) stochastic transition with (p₀, p₁, p₂)
  - After period 1, pay terminal cost V⁰(I₂, b₁)

Usage:
    from solver     import Params, TransshipmentDP
    from simulation import validate, trace, validate_batch

    dp = TransshipmentDP(Params(...))
    dp.solve()

    # aggregate validation
    result = validate(dp, I2_init=15, b1_init=0, n_sims=5000)

    # single trajectory
    traj = trace(dp, I2_init=15, b1_init=0, seed=123)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from solver import Params, TransshipmentDP


# ═══════════════════════════════════════════════════════════════════════
#  Aggregate validation
# ═══════════════════════════════════════════════════════════════════════
def validate(dp: TransshipmentDP,
             I2_init: int,
             b1_init: int,
             n_sims: int = 5000,
             seed: int = 42) -> Dict:
    """
    Run n_sims independent sample paths under the DP policy.

    Returns
    -------
    dict with keys:
        mean, std, ci95_lo, ci95_hi, n_sims,
        dp_cost      (V^N at the initial state),
        gap_pct      (relative gap: (sim_mean - dp_cost) / dp_cost × 100),
        in_ci        (bool: is dp_cost inside the 95% CI?)
    """
    rng = np.random.default_rng(seed)
    p   = dp.p
    costs = np.zeros(n_sims)

    for s in range(n_sims):
        I2, b1 = I2_init, b1_init
        total  = 0.0

        for n in range(p.N, 0, -1):
            q = dp.get_policy(n, I2, b1)
            total += dp.g(I2, b1, q)
            I2 -= q
            b1 -= q

            # stochastic transition
            u = rng.random()
            if u < p.p1:
                b1 += 1
            elif u < p.p1 + p.p2:
                I2 -= 1

            I2 = dp._clip_I2(I2)
            b1 = dp._clip_b1(b1)

        # terminal clean-up cost
        total += dp.terminal(I2, b1)
        costs[s] = total

    m  = np.mean(costs)
    se = np.std(costs) / np.sqrt(n_sims)
    ci_lo, ci_hi = m - 1.96 * se, m + 1.96 * se

    dp_cost = dp.get_value(p.N, I2_init, b1_init)
    gap     = (m - dp_cost) / dp_cost * 100

    return {
        'dp_cost':  dp_cost,
        'mean':     m,
        'std':      np.std(costs),
        'ci95_lo':  ci_lo,
        'ci95_hi':  ci_hi,
        'gap_pct':  gap,
        'in_ci':    ci_lo <= dp_cost <= ci_hi,
        'n_sims':   n_sims,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Single trajectory trace
# ═══════════════════════════════════════════════════════════════════════
def trace(dp: TransshipmentDP,
          I2_init: int,
          b1_init: int,
          seed: int = 0) -> pd.DataFrame:
    """
    Simulate one sample path and record full trajectory.

    Returns a DataFrame with columns:
        period, tau, I2_pre, b1_pre, q_star, safety_stock,
        event, I2_post, b1_post, period_cost, cum_cost
    """
    rng = np.random.default_rng(seed)
    p   = dp.p
    rows = []
    I2, b1 = I2_init, b1_init
    cum_cost = 0.0

    for n in range(p.N, 0, -1):
        tau = n * p.dt                          # remaining time
        q   = dp.get_policy(n, I2, b1)
        ss  = I2 - q                            # safety stock after dispatch
        gc  = dp.g(I2, b1, q)
        cum_cost += gc

        I2_post = I2 - q
        b1_post = b1 - q

        # stochastic transition
        u = rng.random()
        if u < p.p1:
            event = 'demand_R1'
            b1_post += 1
        elif u < p.p1 + p.p2:
            event = 'demand_R2'
            I2_post -= 1
        else:
            event = 'none'

        I2_post = dp._clip_I2(I2_post)
        b1_post = dp._clip_b1(b1_post)

        rows.append({
            'period':       n,
            'tau':          round(tau, 4),
            'I2_pre':       I2,
            'b1_pre':       b1,
            'q_star':       q,
            'safety_stock': ss,
            'event':        event,
            'I2_post':      I2_post,
            'b1_post':      b1_post,
            'period_cost':  round(gc, 4),
            'cum_cost':     round(cum_cost, 4),
        })

        I2, b1 = I2_post, b1_post

    # terminal
    term_cost = dp.terminal(I2, b1)
    cum_cost += term_cost
    rows.append({
        'period':       0,
        'tau':          0.0,
        'I2_pre':       I2,
        'b1_pre':       b1,
        'q_star':       0,
        'safety_stock': I2,
        'event':        'terminal',
        'I2_post':      I2,
        'b1_post':      b1,
        'period_cost':  round(term_cost, 4),
        'cum_cost':     round(cum_cost, 4),
    })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  Batch validation across multiple initial states
# ═══════════════════════════════════════════════════════════════════════
def validate_batch(dp: TransshipmentDP,
                   init_states: List[Tuple[int, int]],
                   n_sims: int = 5000,
                   seed: int = 42) -> pd.DataFrame:
    """
    Run validate() for each (I2_init, b1_init) in init_states.
    Returns a summary DataFrame.
    """
    rows = []
    for I2_0, b1_0 in init_states:
        r = validate(dp, I2_0, b1_0, n_sims=n_sims, seed=seed)
        rows.append({
            'I2_init':   I2_0,
            'b1_init':   b1_0,
            'dp_cost':   r['dp_cost'],
            'sim_mean':  r['mean'],
            'sim_std':   r['std'],
            'ci95_lo':   r['ci95_lo'],
            'ci95_hi':   r['ci95_hi'],
            'gap_pct':   r['gap_pct'],
            'in_ci':     '✓' if r['in_ci'] else '✗',
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  Batch validation across multiple SCENARIOS
# ═══════════════════════════════════════════════════════════════════════
def validate_scenarios(scenarios: Dict[str, Params],
                       I2_init: int = 15,
                       b1_init: int = 0,
                       n_sims: int = 5000) -> pd.DataFrame:
    """
    Solve each scenario and validate with simulation.
    Returns summary DataFrame.
    """
    rows = []
    for name, params in scenarios.items():
        print(f"  [{name}] solving...", end='', flush=True)
        dp = TransshipmentDP(params)
        dp.solve(store_V=True, verbose=False)
        r = validate(dp, I2_init, b1_init, n_sims=n_sims)
        print(f"  DP={r['dp_cost']:.2f}  Sim={r['mean']:.2f}  "
              f"Gap={r['gap_pct']:+.3f}%  CI={'✓' if r['in_ci'] else '✗'}")
        rows.append({
            'scenario':  name,
            'Cf': params.Cf, 'pi1': params.pi1, 'pi2': params.pi2,
            'c1': params.c1, 'c2': params.c2, 'v2': params.v2,
            'dp_cost':   round(r['dp_cost'], 2),
            'sim_mean':  round(r['mean'], 2),
            'sim_std':   round(r['std'], 2),
            'ci95_lo':   round(r['ci95_lo'], 2),
            'ci95_hi':   round(r['ci95_hi'], 2),
            'gap_pct':   round(r['gap_pct'], 3),
            'in_ci':     '✓' if r['in_ci'] else '✗',
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from solver import Params, TransshipmentDP

    print("=== Simulation self-test ===")
    p  = Params(N=100, Cf=20, pi1=10, pi2=10, c1=5, c2=5, v2=1)
    dp = TransshipmentDP(p)
    dp.solve(store_V=True)

    # Aggregate validation
    r = validate(dp, 15, 0, n_sims=3000)
    print(f"\nDP cost  = {r['dp_cost']:.2f}")
    print(f"Sim mean = {r['mean']:.2f}  ± {r['std']:.2f}")
    print(f"95% CI   = ({r['ci95_lo']:.2f}, {r['ci95_hi']:.2f})")
    print(f"In CI?     {r['in_ci']}")

    # Trajectory trace
    traj = trace(dp, 15, 0, seed=7)
    dispatches = traj[traj['q_star'] > 0]
    print(f"\nTrajectory: {len(traj)} periods, {len(dispatches)} dispatches")
    print(dispatches[['period', 'tau', 'I2_pre', 'b1_pre',
                      'q_star', 'safety_stock']].to_string(index=False))