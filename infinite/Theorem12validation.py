import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from infinite.infiniteInstance import TransshipmentInstance
from infinite.infiniteSolver import InfiniteSolver


# --- 2. Helper Functions for Theoretical Calculations ---

def calc_theoretical_b1_bar(inst, I2):
    """Calculates b1_bar based on Theorem 1 ."""
    h_pi = inst.h + inst.pi1
    term_A = 2 * I2 + 1 - (2 * inst.l2 * inst.cu) / h_pi
    term_B = (8 * inst.l2 * inst.Cf) / h_pi

    discriminant = term_A ** 2 - term_B
    if discriminant < 0:
        return None  # Always Wait

    b1_bar = 0.5 * term_A - 0.5 * np.sqrt(discriminant)
    return b1_bar


def calc_theoretical_I2_bar(inst):
    """Calculates I2_bar based on Theorem 1 ."""
    h_pi = inst.h + inst.pi1
    ratio = (2 * inst.l2 * inst.cu) / h_pi
    term_inner = (1 - ratio)
    term_fixed = (8 * inst.l2 * inst.Cf) / h_pi

    discriminant = term_inner ** 2 + term_fixed
    I2_bar = 0.5 * (np.sqrt(discriminant) - term_inner)
    return I2_bar


def save_policy_to_csv(filename, policy_map, solver, tau):
    """Writes the policy map to a CSV file including V_wait, Action, q*, and V_final."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["I2", "b1", "tau", "V_wait", "Action", "q_star", "V_final"])

        # Sort keys for consistent output
        for (I2, b1) in sorted(policy_map.keys()):
            action, q_star = policy_map[(I2, b1)]

            # Recalculate values for the CSV report
            v_wait = solver.get_waiting_value(I2, b1, tau)
            if action == "DISPATCH":
                # V_d = Cf + cu*q + V_w(I2-q, b1-q, tau)
                v_final = solver.inst.Cf + solver.inst.cu * q_star + \
                          solver.get_waiting_value(I2 - q_star, b1 - q_star, tau)
            else:
                v_final = v_wait

            writer.writerow([I2, b1, tau, f"{v_wait:.2f}", action, q_star, f"{v_final:.2f}"])
    print(f"-> Policy data saved to {filename}")


def plot_policy(filename, title="Optimal Transshipment Policy"):
    """Reads the CSV policy file and plots the state space."""
    df = pd.read_csv(filename)

    # Separate states by action
    wait_states = df[df['Action'] == 'WAIT']
    dispatch_states = df[df['Action'] == 'DISPATCH']

    plt.figure(figsize=(10, 6))
    plt.scatter(wait_states['I2'], wait_states['b1'], c='blue', label='WAIT', alpha=0.5, marker='o')
    plt.scatter(dispatch_states['I2'], dispatch_states['b1'], c='red', label='DISPATCH', marker='x')

    plt.title(f"{title} (State Space)")
    plt.xlabel("Inventory at Retailer 2 (I2)")
    plt.ylabel("Backorders at Retailer 1 (b1)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- 3. Experiment: Verify Tau Independence ---

def experiment_tau_independence():
    print("\n=== Experiment 1: Testing Tau Independence (Special Case 4.1) ===")
    # Parameters where pi1=pi2 and c1=c2
    inst = TransshipmentInstance(l1=8, l2=5, h=1, pi1=5, pi2=5,
                                 Cf=20, cu=1, c1=5, c2=5, v2=2, T=10)
    solver = InfiniteSolver(inst)

    tau_values = [0.1, 3.0, 5.0, 7.0, 9.9]  # Series of tau values
    policies = []

    I2_range = range(-50, 50)
    b1_range = range(0, 20)

    for tau in tau_values:
        pol = solver.solve_policy(I2_range, b1_range, tau)
        policies.append(pol)
        # Save each tau's result to CSV
        csv_name = f"policy_exp1_tau_{tau}.csv"
        save_policy_to_csv(csv_name, pol, solver, tau)
        print(f"Solved for tau={tau}")

    # Compare policies
    match = True
    for key in policies[0]:
        act0, q0 = policies[0][key]
        for i in range(1, len(tau_values)):
            acti, qi = policies[i][key]
            if act0 != acti or q0 != qi:
                match = False
                print(f"Mismatch at State {key}: tau={tau_values[0]}->{act0} vs tau={tau_values[i]}->{acti}")

    if match:
        print("SUCCESS: Optimal policy is identical across all tau values.")
        print("Conclusion: As stated in Section 4, the special case removes tau from the decision boundary.")
        # Plot one of the policies to visualize the tau-independent structure
        for tau in tau_values:
            plot_policy(f"policy_exp1_tau_{tau}.csv", title=f"Tau Independence (Tau={tau})")
    else:
        print("FAILURE: Policies differ by tau.")


# --- 4. Experiment: Verify Theorem 1 (Case 4.1) ---

def experiment_theorem_1():
    print("\n=== Experiment 2: Verifying Theorem 1 (Case 4.1) ===")
    # [cite_start]Case 4.1 Condition: (l2 * cu) / (h + pi) <= 0.5 [cite: 133]
    # Parameters: h=1, pi=10 -> h+pi=11. l2=2, cu=1 -> l2*cu=2. Ratio = 2/11 = 0.18 <= 0.5. Holds.
    inst = TransshipmentInstance(l1=2, l2=2, h=1, pi1=10, pi2=10,
                                 Cf=20, cu=1, c1=5, c2=5, v2=2, T=10)
    solver = InfiniteSolver(inst)

    tau = 5.0
    # Wide range to catch thresholds
    I2_range = range(0, 20)
    b1_range = range(0, 20)
    policy = solver.solve_policy(I2_range, b1_range, tau)

    # Save results to CSV
    csv_filename = "policy_exp2_theorem1.csv"
    save_policy_to_csv(csv_filename, policy, solver, tau)

    # 1. Check q* = min(I2, b1)
    q_check = True
    for (I2, b1), (act, q) in policy.items():
        if act == "DISPATCH":
            if q != min(I2, b1):
                q_check = False
                print(f"q* Mismatch at {I2, b1}: Got {q}, Expected {min(I2, b1)}")
    if q_check:
        print("CHECK 1 PASSED: q* = min(I2, b1) for all dispatch states.")

    # 2. Check Threshold I2_bar (Eq 172)
    # This applies when b1 >= I2 + 1. We look for the I2 where behavior switches from WAIT to DISPATCH.
    theo_I2_bar = calc_theoretical_I2_bar(inst)
    print(f"Theoretical I2_bar: {theo_I2_bar:.4f}")

    # Scan simulation for transition
    sim_transition = None
    for I2 in I2_range:
        # Check a high b1 (e.g., b1 = I2 + 5) to ensure we are in Case 2 of Theorem 1
        b1_test = I2 + 5
        if b1_test not in b1_range: continue

        act, _ = policy[(I2, b1_test)]
        if act == "DISPATCH":
            sim_transition = I2
            break

    print(f"Simulation: Waits for I2 < {sim_transition}, Dispatches for I2 >= {sim_transition}")
    if sim_transition is not None and (sim_transition - 1 <= theo_I2_bar <= sim_transition):
        print("CHECK 2 PASSED: Simulation transition aligns with theoretical I2_bar.")
    else:
        print("CHECK 2 FAILED: Mismatch in I2 threshold.")

    # Plot the policy for verification
    plot_policy(csv_filename, title="Theorem 1 Verification (Case 4.1)")


# --- 5. Experiment: Verify Theorem 2 (Case 4.2) ---

def experiment_theorem_2():
    print("\n=== Experiment 3: Verifying Theorem 2 (Case 4.2) ===")
    # [cite_start]Case 4.2 Condition: (l2 * cu) / (h + pi) > 0.5 [cite: 174]
    # Parameters: Need high cu or low pi.
    # Let h=1, pi=2 (h+pi=3). l2=2. Need 2*cu/3 > 0.5 -> cu > 0.75. Let cu=2. Ratio = 4/3 = 1.33.
    inst = TransshipmentInstance(l1=2, l2=2, h=1, pi1=2, pi2=2,
                                 Cf=20, cu=2, c1=5, c2=5, v2=2, T=10)
    solver = InfiniteSolver(inst)

    tau = 5.0
    policy = solver.solve_policy(range(0, 20), range(0, 20), tau)

    # Save results to CSV
    csv_filename = "policy_exp3_theorem2.csv"
    save_policy_to_csv(csv_filename, policy, solver, tau)

    # 1. Check if q* is capped (not always min(I2, b1))
    capped_q_found = False
    for (I2, b1), (act, q) in policy.items():
        if act == "DISPATCH" and q < min(I2, b1):
            capped_q_found = True
            print(f"Found capped dispatch: State({I2},{b1}) -> q*={q} (vs min={min(I2, b1)})")
            break

    if capped_q_found:
        print("CHECK 1 PASSED: Found states where q* < min(I2, b1), satisfying Theorem 2.")

    # 2. Check Waiting condition for I2 <= ratio
    # Ratio = lambda2 * cu / (h + pi) = 2 * 2 / 3 = 1.333
    # Theory says: If I2 <= 1.333 (i.e., I2=0, 1), we should WAIT regardless of b1.
    ratio = (inst.l2 * inst.cu) / (inst.h + inst.pi1)
    print(f"Theoretical Wait Limit (Ratio): {ratio:.4f}")

    violation = False
    for I2 in [0, 1]:  # Integers <= 1.33
        for b1 in range(1, 20):
            if (I2, b1) in policy:
                act, _ = policy[(I2, b1)]
                if act == "DISPATCH":
                    violation = True
                    print(f"Violation at {I2, b1}: Should WAIT but got DISPATCH.")

    if not violation:
        print(f"CHECK 2 PASSED: System always waits for I2 <= {ratio:.2f}.")

    # Plot the policy for verification
    plot_policy(csv_filename, title="Theorem 2 Verification (Case 4.2)")


if __name__ == "__main__":
    experiment_tau_independence()
    # experiment_theorem_1()
    # experiment_theorem_2()