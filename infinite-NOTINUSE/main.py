from infinite import infiniteInstance, infiniteSolver
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import csv


def main():
    # Setup parameters: Case 4.1 (pi1=pi2, c1=c2)
    # h=1, pi=10, Cf=20, cu=1, c1=5, c2=5, v2=2, T=10
    inst = infiniteInstance.TransshipmentInstance(l1=2, l2=2, h=1, pi1=5, pi2=5,
                                 Cf=20, cu=1, c1=5, c2=5, v2=2, T=10)
    solver = infiniteSolver.InfiniteSolver(inst)

    tau = 5.0  # Time remaining until end of horizon
    I2_max = 30
    b1_max = 30

    filename = "optimal_policy_results.csv"

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header definition
        writer.writerow(["I2", "b1", "tau", "V_wait", "Action", "q_star", "V_final"])

        for I2 in range(I2_max + 1):
            for b1 in range(b1_max + 1):
                # Calculate Waiting Value
                v_wait = solver.get_waiting_value(I2, b1, tau)

                # Default to waiting if no inventory or no backorders
                if I2 <= 0 or b1 <= 0:
                    writer.writerow([I2, b1, tau, v_wait, "WAIT", 0, v_wait])
                    continue

                # Determine optimal dispatch quantity q*
                q_star = solver.get_optimal_q(I2, b1, tau)

                # Calculate Dispatch Value
                # V_d = Cf + cu*q + V_w(I2-q, b1-q, tau)
                v_dispatch = (inst.Cf + inst.cu * q_star +
                              solver.get_waiting_value(I2 - q_star, b1 - q_star, tau))

                # Decision Rule: Transship if V_d < V_w
                if v_dispatch < v_wait:
                    action = "DISPATCH"
                    v_final = v_dispatch
                else:
                    action = "WAIT"
                    q_star = 0
                    v_final = v_wait

                writer.writerow([I2, b1, tau, f"{v_wait:.2f}", action, q_star, f"{v_final:.2f}"])

    print(f"Full optimal policy for tau={tau} recorded in {filename}")
    plot_policy(filename)

def plot_policy(filename):
    df = pd.read_csv(filename)

    # Separate states by action
    wait_states = df[df['Action'] == 'WAIT']
    dispatch_states = df[df['Action'] == 'DISPATCH']

    plt.figure(figsize=(10, 6))
    plt.scatter(wait_states['I2'], wait_states['b1'], c='blue', label='WAIT', alpha=0.5)
    plt.scatter(dispatch_states['I2'], dispatch_states['b1'], c='red', label='DISPATCH', marker='x')

    plt.title("Optimal Transshipment Policy (State Space)")
    plt.xlabel("Inventory at Retailer 2 (I2)")
    plt.ylabel("Backorders at Retailer 1 (b1)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()