# DPmain.py (runner only)

from minimalSolver import (
    Instance,
    solveDP_AMO_Bpriority_dynamic,
    get_optimal_expected_cost_user_t,
    run_simulations,
    append_sim_results,)



# presentation helpers live outside the solver:

from present_epoch import present_epoch
from present_policy import present_policy
from present_surface import present_surface

# import verify_theory functions
from minimalSolver import Instance, solveDP_AMO_Bpriority_dynamic
from verify_theory import run_full_check

if __name__ == "__main__":
    # knobs

    RUN_LABEL = "Baseline"
    EPOCH_OUTFILE = "epoch.csv"
    POLICY_OUTFILE = "policy.csv"
    SIMS_OUTFILE = "sims.csv"
    SEED = 2025

    # instance
    inst = Instance(
    N=20, T=2.0,
    lambdaA=8, lambdaB=5,
    h=0.1, pA=10.0, pB=10.0,
    cf=20.0, cu=1.0,
    minIB=-20, maxIB=20, maxbA=10,
    IB0=20,
    salvage_v=5.0  # <--- ADD THIS LINE. 0.0 = No Salvage, >0.0 = With Salvage
    )



    # solve
    solution = solveDP_AMO_Bpriority_dynamic(inst)



    # optimal expected cost at start (user t=0 => r=N)
    opt_cost = get_optimal_expected_cost_user_t(solution, inst, t_user=0, IB=inst.IB0, bA=0)
    print(f"[Exact] Optimal expected cost at t=0 from (IB0={inst.IB0}, bA0=0): {opt_cost:.4f}")

    # === Single trajectory presentation (append-only CSV) ===
    # present_epoch(inst, solution, IB0=inst.IB0, bA0=0, seed=SEED,outfile=EPOCH_OUTFILE, label=RUN_LABEL)

    # Append N matrices (r = N..1), never overwrite, with blank line separators
    # present_policy(inst, solution, outfile="policy.csv", include_r0=False)

    # 3D surface of full policy (one PNG)
    # present_surface(inst, solution, outdir="figures", label=RUN_LABEL, dpi=180)

    # simulations (append-only CSV with per-run and one summary row)
    simN = 2000
    base_seed = 3260
    mean, std, lo, hi, costs = run_simulations(inst, solution, simN, base_seed, IB0=inst.IB0, bA0=0)
    print(f"[Tally over {simN} sims] mean={mean:.4f}, std={std:.4f}, 95% CI=({lo:.4f}, {hi:.4f})")
    # append_sim_results(SIMS_OUTFILE, RUN_LABEL, base_seed, costs)

