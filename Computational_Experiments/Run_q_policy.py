"""
Run_q_policy.py — Sweep the lot-size grid for the Q-policy.

Importable (sweep_q) and runnable standalone:
    python Run_q_policy.py
"""

import json
import os

import Config
from simulator import make_uniforms, simulate, summarise
from policy_Q import QPolicy


def sweep_q(params, U, q_grid, I2_init, b1_init, verbose=True):
    """Evaluate the Q-policy for every Q on the grid.  Returns a list of records."""
    records = []
    for Q in q_grid:
        pol = QPolicy(Q)
        raw = simulate(pol, params, U, I2_init, b1_init)
        rec = {"Q": int(Q), "policy": pol.label()}
        rec.update(summarise(raw))
        records.append(rec)
        if verbose:
            t = rec["total"]
            print("  Q=%3d   cost = %9.3f +/- %.3f   dispatches/run = %.2f"
                  % (Q, t["mean"], t["ci95"], rec["n_dispatch"]["mean"]))
    return records


def save_records(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print("  saved -> %s" % path)


if __name__ == "__main__":
    p = Config.PARAMS
    p.validate()
    print("Q-policy sweep   instance: %s" % p.summary())
    print("  reps=%d  seed=%d  grid=%s" % (Config.N_REPS, Config.SEED, Config.Q_GRID))

    U = make_uniforms(Config.N_REPS, p.N, Config.SEED)
    records = sweep_q(p, U, Config.Q_GRID, Config.I2_INIT, Config.B1_INIT)
    save_records(records, Config.Q_RESULTS_JSON)