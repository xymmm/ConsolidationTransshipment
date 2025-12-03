import numpy as np
import matplotlib.pyplot as plt

from TransshipmentInstance import TransshipmentInstance
from TransshipmentSolver import TransshipmentSolver


def plot_switching_curve_on_ax(solver, time_idx, ax):
    """
    External version of `plot_switching_curve` that draws on a
    provided matplotlib Axes, without modifying TransshipmentSolver.
    """
    if not solver.Policy_tables:
        print("Error: Model not solved.")
        return

    print(f"Generating Switching Curve for t={time_idx}...")

    policy = solver.Policy_tables[time_idx]

    real_I2 = np.arange(solver.inst.i_min, solver.inst.i_max + 1)
    real_b1 = np.arange(0, solver.inst.b_max + 1)

    # Arrays for plotting lines
    x_b1 = []
    y_switch_I2 = []
    y_full_I2 = []

    # Arrays for plotting Wait Nodes (Grey Dots)
    wait_x = []
    wait_y = []

    for j, b1_val in enumerate(real_b1):
        q_col = policy[:, j]

        # --- 1. CAPTURE ALL WAIT NODES ---
        wait_indices = np.where(q_col == 0)[0]
        for idx in wait_indices:
            i2_val = real_I2[idx]
            if i2_val >= 0:
                wait_x.append(b1_val)
                wait_y.append(i2_val)

        # --- 2. SWITCHING BOUNDARY (q > 0) ---
        shipping_indices = np.where(q_col > 0)[0]
        if shipping_indices.size > 0:
            y_switch_I2.append(real_I2[shipping_indices[0]])
        else:
            y_switch_I2.append(np.nan)

        # --- 3. FULL SATISFACTION (q == b1) ---
        full_sat_indices = np.where(q_col == b1_val)[0]
        if full_sat_indices.size > 0:
            y_full_I2.append(real_I2[full_sat_indices[0]])
        else:
            y_full_I2.append(np.nan)

        x_b1.append(b1_val)

    # A. Plot the Grey Dots (Background Grid)
    ax.scatter(wait_x, wait_y, color='lightgray', marker='.', s=50,
               label='Wait ($q=0$)')

    # B. Plot the Lines
    ax.plot(x_b1, y_switch_I2, 'o-', color='navy', linewidth=2,
            label='Start Shipping ($q>0$)')
    ax.plot(x_b1, y_full_I2, 's--', color='green', linewidth=2,
            label='Clear Backlog ($q=b_1$)')

    # C. Fill Regions
    valid_mask = np.isfinite(y_switch_I2) & np.isfinite(y_full_I2)
    if np.any(valid_mask):
        x_arr = np.array(x_b1)[valid_mask]
        y_sw = np.array(y_switch_I2)[valid_mask]
        y_fl = np.array(y_full_I2)[valid_mask]

        # Rationing Region (Orange)
        ax.fill_between(x_arr, y_sw, y_fl, color='orange', alpha=0.2,
                        label='Rationing Region')

        # Wait Region (Red Tint)
        # ax.fill_between(x_arr, 0, y_sw, color='red', alpha=0.1,
                        # label='Wait Region')

    ax.set_xlabel('Retailer 1 Backlog ($b_1$)')
    ax.set_ylabel('Retailer 2 Inventory ($I_2$)')
    ax.set_title(f'Switching & Full-Satisfaction Thresholds (t={time_idx})')

    ax.set_xlim(0, solver.inst.b_max)
    ax.set_ylim(0, solver.inst.i_max)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)


def plot_comparison(solvers_dict, time_idx):
    """
    For a given time_idx, plot the switching curves of all solvers
    side by side in one figure as horizontal subplots.
    """
    print(f"\n==============================================")
    print(f"   Visualizing Policy at Time Step t={time_idx}")
    print(f"   (Steps Remaining: {300 - time_idx})")
    print(f"==============================================\n")

    n_cases = len(solvers_dict)
    fig, axes = plt.subplots(
        1, n_cases,
        figsize=(5 * n_cases, 4),
        sharey=True  # share Y axis across cases
    )

    if n_cases == 1:
        axes = [axes]

    for ax, (name, solver) in zip(axes, solvers_dict.items()):
        clean_name = name.replace("\n", " ")
        print(f"---> Plotting: {clean_name}")

        plot_switching_curve_on_ax(solver, time_idx, ax)
        ax.set_title(clean_name, fontsize=9)

    fig.suptitle(f"Switching Curves at t = {time_idx}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def main():
    print("--- 1. Setup Parameters (Curve-Generating Set) ---")
    # Base parameters that generate the nice "Band" curve
    base_params = dict(
        T=2.0, N=300,
        lambda_1=8.0, lambda_2=5.0,
        h=0.1, pi_1=10.0, pi_2=10.0,
        Cf=50.0, cu=1.0,
        i_min=-50, i_max=50, b_max=20
    )

    print("--- 2. Solving Three Cases ---")

    # Case 1: Nagihan (Standard)
    print("\nSolving Case 1: no salvage or end penalty")
    inst1 = TransshipmentInstance(**base_params, c2=0.0, pi_end=0.0)
    s1 = TransshipmentSolver(inst1)
    s1.solve()

    # Case 2: The Trap (Hoarding)
    print("\nSolving Case 2: Hoarding only salvage")
    inst2 = TransshipmentInstance(**base_params, c2=50.0, pi_end=0.0)
    s2 = TransshipmentSolver(inst2)
    s2.solve()

    # Case 3: Bo (Incentive Compatible)
    print("\nSolving Case 3: salvage and high end penalty")
    inst3 = TransshipmentInstance(**base_params, c2=50.0, pi_end=100.0)
    s3 = TransshipmentSolver(inst3)
    s3.solve()

    solvers = {
        "Case 1: no salvage or end penalty": s1,
        "Case 2: only salvage": s2,
        "Case 3: salvage and high end penalty": s3
    }

    print("\n--- 3. Plotting Comparisons ---")

    # Snapshot A: The Beginning (t=0)
    plot_comparison(solvers, time_idx=0)

    # Snapshot B: The Middle (t=150)
    plot_comparison(solvers, time_idx=150)

    # Snapshot C: The End (t=290)
    plot_comparison(solvers, time_idx=290)


if __name__ == "__main__":
    main()
