from TransshipmentInstance import TransshipmentInstance
from TransshipmentSimulator import TransshipmentSimulator
from TransshipmentSolver import TransshipmentSolver


# Assuming TransshipmentInstance and TransshipmentSolver are defined/imported above

def main():
    # 1. Setup Instance
    inst = TransshipmentInstance(
        T=2.0, N=300,
        lambda_1=8.0, lambda_2=5.0,
        h=0.1, pi_1=10.0, pi_2=10.0,
        Cf=50.0, cu=1,
        c2=50.0, pi_end=100,
        i_min=-50, i_max=50, b_max=20
    )

    # 2. Solve
    solver = TransshipmentSolver(inst)
    solver.solve()

    # Get MDP Expected Cost at t=0 for the specific starting state
    start_i_idx = inst.I2_0 - inst.i_min
    start_b_idx = inst.b1_0
    mdp_cost = solver.V_tables[0][start_i_idx, start_b_idx]

    # 3. Generate CSVs
    solver.save_policy_to_csv(time_idx=0, filename="optimal_policy_matrix_t0.csv")
    solver.save_thresholds_to_csv(filename="threshold_evolution.csv")

    print("\n--- 3. Validate with Continuous-Time Simulation ---")
    sim = TransshipmentSimulator(inst, solver)
    sim_mean, sim_ci = sim.run_monte_carlo(num_simulations=2000)

    print(f"[Sim Result] Mean Cost: {sim_mean:.4f} +/- {sim_ci:.4f}")

    # Validation Check
    error_pct = abs(mdp_cost - sim_mean) / mdp_cost * 100
    print(f"Discrepancy: {error_pct:.2f}%")
    if error_pct < 1.0:
        print(">> VALIDATION SUCCESSFUL (Gap < 1%)")
    else:
        print(">> GAP > 1%. (Expected due to Discrete vs Continuous time approximation)")


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

    solver.plot_3d_policy_surface(time_idx=0)


if __name__ == "__main__":
    main()