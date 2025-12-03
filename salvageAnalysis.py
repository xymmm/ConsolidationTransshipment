import csv
import numpy as np
from TransshipmentInstance import TransshipmentInstance
from TransshipmentSolver import TransshipmentSolver

def compute_policy_metrics(solver, K_early=10, K_late=10):
    """
    从 solver.Policy_tables 里计算:
      - early_ship_mean
      - late_ship_mean
      - laziness_index
      - last_dispatch_time
    只在 I2>0 且 b1>0 的状态上统计。
    """
    policies = solver.Policy_tables
    if not policies:
        raise ValueError("Solver has no policy tables. Did you call solve()?")

    N = len(policies) - 1          # time steps 0..N
    n_I2, n_b1 = policies[0].shape

    real_I2 = np.arange(solver.inst.i_min, solver.inst.i_max + 1)
    real_b1 = np.arange(0, solver.inst.b_max + 1)

    # 利用 solver 里的网格
    I2_grid = solver.I2_grid
    b1_grid = solver.b1_grid
    active_mask = (I2_grid > 0) & (b1_grid > 0)
    active_count = active_mask.sum()
    if active_count == 0:
        raise ValueError("No active states with I2>0 and b1>0")

    ship_ratio = np.zeros(N + 1)
    wait_ratio = np.zeros(N + 1)

    last_dispatch_time = -1

    for t in range(N + 1):
        policy_t = policies[t]  # shape (n_I2, n_b1)

        ship_mask = (policy_t > 0) & active_mask
        wait_mask = (policy_t == 0) & active_mask

        ship_ratio[t] = ship_mask.sum() / active_count
        wait_ratio[t] = wait_mask.sum() / active_count

        if ship_mask.any():
            last_dispatch_time = t

    K_early = min(K_early, N + 1)
    K_late = min(K_late, N + 1)

    early_indices = np.arange(0, K_early)
    late_indices = np.arange(max(0, N - K_late + 1), N + 1)

    early_ship_mean = ship_ratio[early_indices].mean()
    late_ship_mean = ship_ratio[late_indices].mean()
    laziness_index = early_ship_mean - late_ship_mean

    return {
        "early_ship_mean": early_ship_mean,
        "late_ship_mean": late_ship_mean,
        "laziness_index": laziness_index,
        "last_dispatch_time": last_dispatch_time
    }


def run_experiment(base_params, param_grid, csv_path="boundary_sensitivity.csv",
                   K_early=10, K_late=10):
    """
    base_params: dict，不含 c2, pi_end 的公共参数
    param_grid: list of tuples (case_name, c2, pi_end)
    结果写到 csv_path
    """
    fieldnames = [
        "case_name",
        "c2",
        "pi_end",
        "early_ship_mean",
        "late_ship_mean",
        "laziness_index",
        "last_dispatch_time"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case_name, c2, pi_end in param_grid:
            print(f"\n=== Solving {case_name} (c2={c2}, pi_end={pi_end}) ===")
            inst = TransshipmentInstance(**base_params, c2=c2, pi_end=pi_end)
            solver = TransshipmentSolver(inst)
            solver.solve()

            metrics = compute_policy_metrics(solver, K_early=K_early, K_late=K_late)

            row = {
                "case_name": case_name,
                "c2": c2,
                "pi_end": pi_end,
                **metrics
            }
            writer.writerow(row)
            print("  ->", row)


def main():
    base_params = dict(
        T=2.0, N=300,
        lambda_1=8.0, lambda_2=5.0,
        h=0.1, pi_1=10.0, pi_2=10.0,
        Cf=50.0, cu=1.0,
        i_min=-50, i_max=50, b_max=20
    )

    param_grid = [
        ("case1_no_salvage",   0.0,   0.0),
        ("case2_w_salv_weakP", 50.0, 20.0),
        ("case3_w_salv_midP",  50.0, 50.0),
        ("case4_w_salv_highP", 50.0, 100.0),
        ("case5_w_salv_vHigh", 50.0, 200.0),
    ]

    run_experiment(base_params, param_grid,
                   csv_path="boundary_sensitivity.csv",
                   K_early=10, K_late=10)


if __name__ == "__main__":
    main()
