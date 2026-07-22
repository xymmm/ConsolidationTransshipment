"""
Policy_main.py — Orchestration for the benchmark-policy computational study.

Current scope (by agreement): solve and simulate the two benchmark
policies only.  The commented blocks sketch where future components slot
in without touching the two runners.

    call SDP solver            - exact optimal solution        [future]
    call Q-policy              - save results                  [now]
    call T-policy              - save results                  [now]
    ...... other policies / comparison                         [future]
    simulation for all policies with their saved results       [now, CRN]

Note on imports: Config performs the sys.path bootstrap that makes the
parent folder's solver.py importable, so `import Config` must come before
any `from solver import ...` in this package.
"""

import Config
from simulator import make_uniforms
from Run_q_policy import sweep_q, save_records as save_q
from Run_t_policy import sweep_t, save_records as save_t


def main():
    p = Config.PARAMS
    p.validate()
    print("=" * 70)
    print("Benchmark policy study")
    print("  instance : %s" % p.summary())
    print("  initial  : I2=%d  b1=%d" % (Config.I2_INIT, Config.B1_INIT))
    print("  sim      : reps=%d  seed=%d" % (Config.N_REPS, Config.SEED))
    print("=" * 70)

    # ── shared CRN array: every policy and parameter uses the same paths ──
    U = make_uniforms(Config.N_REPS, p.N, Config.SEED)

    # ── [future] SDP exact optimal solution ──
    # from solver import TransshipmentDP
    # dp = TransshipmentDP(p); dp.solve(store_V=True)
    # V_opt = dp.get_value(p.N, Config.I2_INIT, Config.B1_INIT)
    # A wrapper policy reading dp.policy[n_remaining, ·, ·] can then be
    # simulated on the same U as a validation of the simulator.

    # ── Q-policy sweep ──
    print("\n[1/2] Q-policy sweep")
    q_records = sweep_q(p, U, Config.Q_GRID, Config.I2_INIT, Config.B1_INIT)
    save_q(q_records, Config.Q_RESULTS_JSON)

    # ── T-policy sweep ──
    print("\n[2/2] T-policy sweep")
    t_records = sweep_t(p, U, Config.DELTA_GRID, Config.I2_INIT, Config.B1_INIT)
    save_t(t_records, Config.T_RESULTS_JSON)

    # ── [future] Zhou2023.py, Glazebrook2015.py, comparison table ──

    print("\nDone.  Plot with:  python Plot_results.py")


if __name__ == "__main__":
    main()