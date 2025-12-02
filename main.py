from TransshipmentInstance import TransshipmentInstance
from TransshipmentSolver import TransshipmentSolver


# Assuming TransshipmentInstance and TransshipmentSolver are defined/imported above

def main():
    # 1. Setup Instance
    inst = TransshipmentInstance(
        T=1.0, N=300,
        lambda_1=8.0, lambda_2=5.0,
        h=0.1, pi_1=10.0, pi_2=10.0,
        Cf=20.0, cu=1,
        c2=0.0,
        i_min=-10, i_max=20, b_max=20
    )

    # 2. Solve
    solver = TransshipmentSolver(inst)
    solver.solve()

    # 3. Generate CSVs
    solver.save_policy_to_csv(time_idx=0, filename="optimal_policy_matrix_t0.csv")
    solver.save_thresholds_to_csv(filename="threshold_evolution.csv")

    # 4. Generate Graphs
    print("\n--- Plotting Switching Curve (2D) ---")
    solver.plot_switching_curve(time_idx=0)

    # print("\n--- Plotting Policy Surface (3D) ---")
    # solver.plot_3d_policy_surface(time_idx=0)

    # Plot B: Time = 150 (Middle of horizon)
    # The policy might be looser here as less time remains to hold inventory
    print("\n--- Plotting Switching Curve at t=150 ---")
    solver.plot_switching_curve(time_idx=150)

    # Plot C: Time = 290 (Near end of horizon)
    # The policy typically collapses (thresholds drop) because there is no future to save for
    print("\n--- Plotting Switching Curve at t=290 ---")
    solver.plot_switching_curve(time_idx=290)




if __name__ == "__main__":
    main()