"""
Run_t_policy.py — Sweep the review-interval grid for the T-policy.

Importable (sweep_t) and runnable standalone:
    python Run_t_policy.py
"""

import json
import os

import Config
from simulator import make_uniforms, simulate, summarise
from Policy_T import TPolicy


def sweep_t(params, U, delta_grid, I2_init, b1_init, verbose=True):
    """Evaluate the T-policy for every Delta on the grid.  Returns a list of records."""
    records = []
    for delta in delta_grid:
        pol = TPolicy(delta, params)
        raw = simulate(pol, params, U, I2_init, b1_init)
        rec = {
            "delta": float(delta),
            "n_reviews": len(pol.review_steps),
            "policy": pol.label(),
        }
        rec.update(summarise(raw))
        records.append(rec)
        if verbose:
            t = rec["total"]
            print("  Delta=%7.4f  reviews=%3d   cost = %9.3f +/- %.3f   "
                  "dispatches/run = %.2f"
                  % (delta, rec["n_reviews"], t["mean"], t["ci95"],
                     rec["n_dispatch"]["mean"]))
    return records


def save_records(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print("  saved -> %s" % path)


if __name__ == "__main__":
    p = Config.PARAMS
    p.validate()
    print("T-policy sweep   instance: %s" % p.summary())
    print("  reps=%d  seed=%d  grid: Delta = T/m for m in %d..%d"
          % (Config.N_REPS, Config.SEED, Config.M_GRID[0], Config.M_GRID[-1]))

    U = make_uniforms(Config.N_REPS, p.N, Config.SEED)
    records = sweep_t(p, U, Config.DELTA_GRID, Config.I2_INIT, Config.B1_INIT)
    save_records(records, Config.T_RESULTS_JSON)