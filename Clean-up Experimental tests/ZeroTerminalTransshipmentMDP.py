import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CleanUpTransshipmentMDP:
    def __init__(self, T, N, lam1, lam2, h, Cf, cu, pi1, pi2, c1, c2, v2, max_I2=20, min_I2=-5, max_b1=25):
        self.T, self.N, self.dt = T, N, T / N
        self.lam1, self.lam2 = lam1, lam2
        self.h, self.Cf, self.cu = h, Cf, cu
        self.pi1, self.pi2 = pi1, pi2
        self.c1, self.c2, self.v2 = c1, c2, v2

        self.p1 = lam1 * self.dt
        self.p2 = lam2 * self.dt
        self.p0 = 1 - self.p1 - self.p2

        self.max_I2, self.min_I2, self.max_b1 = max_I2, min_I2, max_b1
        self.I2_range = np.arange(self.min_I2, self.max_I2 + 1)
        self.b1_range = np.arange(0, self.max_b1 + 1)
        self.n_I2, self.n_b1 = len(self.I2_range), len(self.b1_range)

        self.V = np.zeros((N + 1, self.n_I2, self.n_b1))
        self.Policy = np.zeros((N + 1, self.n_I2, self.n_b1), dtype=int)

    def _get_idx(self, i2, b1):
        return int(np.clip(i2 - self.min_I2, 0, self.n_I2 - 1)), int(np.clip(b1, 0, self.n_b1 - 1))

    def solve(self):
        # Terminal Condition
        for i_idx, i2 in enumerate(self.I2_range):
            for b_idx, b1 in enumerate(self.b1_range):
                term_cost = self.c1 * b1 + (self.c2 * abs(i2) if i2 < 0 else -self.v2 * i2)
                self.V[0, i_idx, b_idx] = term_cost

        # Backward Induction
        for n in range(1, self.N + 1):
            for i_idx, i2 in enumerate(self.I2_range):
                for b_idx, b1 in enumerate(self.b1_range):
                    max_q = min(i2, b1) if (i2 > 0 and b1 > 0) else 0
                    best_val, best_q = float('inf'), 0

                    # Optimization loop
                    for q in range(max_q + 1):
                        cost_imm = (self.Cf if q > 0 else 0) + (self.cu * q)
                        i2_p, b1_p = i2 - q, b1 - q
                        cost_step = self.dt * (self.h * max(0, i2_p) + self.pi1 * b1_p + self.pi2 * max(0, -i2_p))

                        idx_same = self._get_idx(i2_p, b1_p)
                        idx_dem1 = self._get_idx(i2_p, b1_p + 1)
                        idx_dem2 = self._get_idx(i2_p - 1, b1_p)

                        val_future = self.p0 * self.V[n - 1][idx_same] + self.p1 * self.V[n - 1][idx_dem1] + self.p2 * \
                                     self.V[n - 1][idx_dem2]
                        if cost_imm + cost_step + val_future < best_val:
                            best_val, best_q = cost_imm + cost_step + val_future, q

                    self.V[n, i_idx, b_idx] = best_val
                    self.Policy[n, i_idx, b_idx] = best_q

    def export_policy_csv(self, filename):
        data = []
        for n in range(self.N, -1, -1):  # Time 0 to T
            t = self.T - (n * self.dt)
            for i_idx, i2 in enumerate(self.I2_range):
                for b_idx, b1 in enumerate(self.b1_range):
                    q = self.Policy[n, i_idx, b_idx]
                    data.append({
                        'Time_t': round(t, 2),
                        'I2': i2,
                        'b1': b1,
                        'Optimal_q': q,
                        'delta_t': round(self.dt, 3)  # Added delta_t as requested previously
                    })
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return df

    def simulate(self, initial_I2, initial_b1=0):
        # Returns single trace
        times, i2_hist, b1_hist, events = [0], [initial_I2], [initial_b1], []
        accumulated_cost = [0]
        curr_I2, curr_b1 = initial_I2, initial_b1
        curr_total_cost = 0

        np.random.seed(42)

        for step in range(self.N, 0, -1):
            t = self.T - (step * self.dt)
            i_idx, b_idx = self._get_idx(curr_I2, curr_b1)

            # Decision
            q = self.Policy[step, i_idx, b_idx]
            step_trans_cost = 0
            if q > 0:
                events.append((t, q, curr_b1, curr_I2))
                step_trans_cost = self.Cf + self.cu * q
                curr_I2 -= q
                curr_b1 -= q

            # Holding/Penalty Cost
            step_hold_cost = self.dt * (self.h * max(0, curr_I2) + self.pi1 * curr_b1 + self.pi2 * max(0, -curr_I2))

            curr_total_cost += step_trans_cost + step_hold_cost

            # Transitions
            rand = np.random.rand()
            if rand < self.p1:
                curr_b1 += 1
            elif rand < self.p1 + self.p2:
                curr_I2 -= 1

            # Terminal Cost check at step 1 -> 0?
            # Usually simulation just runs to T.
            # If we want "Total Cost" including terminal, we add it at the end.

            times.append(t + self.dt)
            i2_hist.append(curr_I2)
            b1_hist.append(curr_b1)
            accumulated_cost.append(curr_total_cost)

        # Add terminal cost at t=T
        term_cost = self.c1 * curr_b1 + (self.c2 * abs(curr_I2) if curr_I2 < 0 else -self.v2 * curr_I2)
        accumulated_cost[-1] += term_cost

        return times, i2_hist, b1_hist, events, accumulated_cost

    def get_average_cost_curve(self, initial_I2, initial_b1, num_simulations=200):
        # Runs multiple simulations and averages the accumulated cost curve
        all_costs = np.zeros((num_simulations, self.N + 1))

        for sim_idx in range(num_simulations):
            curr_I2, curr_b1 = initial_I2, initial_b1
            curr_total_cost = 0
            all_costs[sim_idx, 0] = 0

            # Random seed must vary
            np.random.seed(sim_idx)

            for step_idx, step in enumerate(range(self.N, 0, -1)):  # 0 to N-1 index for steps
                t = self.T - (step * self.dt)
                i_idx, b_idx = self._get_idx(curr_I2, curr_b1)

                q = self.Policy[step, i_idx, b_idx]
                step_trans_cost = (self.Cf + self.cu * q) if q > 0 else 0
                curr_I2 -= q
                curr_b1 -= q

                step_hold_cost = self.dt * (self.h * max(0, curr_I2) + self.pi1 * curr_b1 + self.pi2 * max(0, -curr_I2))

                curr_total_cost += step_trans_cost + step_hold_cost

                rand = np.random.rand()
                if rand < self.p1:
                    curr_b1 += 1
                elif rand < self.p1 + self.p2:
                    curr_I2 -= 1

                all_costs[sim_idx, step_idx + 1] = curr_total_cost

            # Add terminal cost
            term_cost = self.c1 * curr_b1 + (self.c2 * abs(curr_I2) if curr_I2 < 0 else -self.v2 * curr_I2)
            all_costs[sim_idx, -1] += term_cost

        avg_costs = np.mean(all_costs, axis=0)
        times = np.linspace(0, self.T, self.N + 1)
        return times, avg_costs


# --- CONFIGURATION ---
common_params = {
    'T': 5.0, 'N': 50,
    'lam1': 5, 'lam2': 3,
    'h': 0.1, 'Cf': 15, 'cu': 1.0,
    'c1': 10, 'c2': 10, 'v2': 0  # Fixed c1=c2=c
}

experiments = [
    {'name': 'Exp1_Equal', 'pi1': 10, 'pi2': 10},
    {'name': 'Exp2_HighPi1', 'pi1': 15, 'pi2': 5},
    {'name': 'Exp3_HighPi2', 'pi1': 5, 'pi2': 15}
]

# Run Experiments
for exp in experiments:
    print(f"Running {exp['name']}...")
    params = common_params.copy()
    params.update({'pi1': exp['pi1'], 'pi2': exp['pi2']})
    model = CleanUpTransshipmentMDP(**params)
    model.solve()

    # 2. Export CSV
    csv_name = f"{exp['name']}_policy.csv"
    model.export_policy_csv(csv_name)

    # 3. Figure 1: Simulation Traces (Backlog & Inventory)
    times_sim, i2s_sim, b1s_sim, events_sim, costs_sim = model.simulate(initial_I2=20, initial_b1=0)

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Subplot 1: Backlog
    ax1.step(times_sim, b1s_sim, 'k-', where='post', label='$b_1$')
    for (t, q, b_pre, i_pre) in events_sim:
        ax1.vlines(t, 0, b_pre, colors='r', linestyles='--')
        ax1.plot(t, b_pre, 'ro')
        ax1.text(t, b_pre + 0.5, f'q={q}', color='r', ha='center', fontsize=8)
    ax1.set_ylabel('Backlog ($b_1$)')
    ax1.set_title(f"{exp['name']}: Simulation Trace (Backlog & Inventory)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Inventory
    ax2.step(times_sim, i2s_sim, 'b-', where='post', label='$I_2$')
    for (t, q, b_pre, i_pre) in events_sim:
        ax2.vlines(t, i_pre - q, i_pre, colors='r', linestyles='-')
        ax2.text(t, i_pre + 0.5, f'-{q}', color='r', ha='center', fontsize=8)
    ax2.set_ylabel('Inventory ($I_2$)')
    ax2.set_xlabel('Time')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{exp['name']}_simulation.png")
    plt.close()

    # 4. Figure 2: Cost & Value Analysis
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot A: Average Accumulated Cost vs Time (Monotonically Increasing)
    # Using I2=15, b1=5 as starting point for this "Expected Cost" curve?
    # Or start from 0 like simulation?
    # Usually "Expected Total Cost" implies expectation from t=0 state.
    # Let's use the same starting state as simulation (20, 0) or the (15, 5) state?
    # User said "optimal expected total cost随时间变化的曲线".
    # If it must rise, it's accumulated.
    # Let's use start state (20, 0) to be consistent with "Evolution of the system".
    t_avg, c_avg = model.get_average_cost_curve(initial_I2=20, initial_b1=0, num_simulations=200)

    ax3.plot(t_avg, c_avg, 'g-', linewidth=2)
    ax3.set_title(f"Accumulated Cost vs Time\n(Avg of 200 Sim, Start: $I_2=20, b_1=0$)")
    ax3.set_xlabel("Time ($t$)")
    ax3.set_ylabel("Avg Accumulated Cost")
    ax3.grid(True, alpha=0.3)

    # Subplot B: V vs I2 (Fixed Time, Fixed b1) - Remains Value Function
    fixed_b1_for_v = 5
    n_mid = int(model.N / 2)
    i_idx, b_idx_v = model._get_idx(0, fixed_b1_for_v)
    v_i2 = [model.V[n_mid, i, b_idx_v] for i in range(model.n_I2)]

    ax4.plot(model.I2_range, v_i2, 'm-', linewidth=2)
    ax4.set_title(f"Future Cost (V) vs Inventory $I_2$\n(Time={model.T / 2}, $b_1={fixed_b1_for_v}$)")
    ax4.set_xlabel("Inventory $I_2$")
    ax4.set_ylabel("Expected Future Cost ($V$)")
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{exp['name']}_cost_analysis.png")
    plt.close()

    # 5. Figure 3: Dispatch Quantity vs Time
    fig3, ax5 = plt.subplots(figsize=(8, 6))

    state_high = 15  # High Inventory -> Dispatch Threshold Logic
    state_low = 3  # Low Inventory -> Safety Threshold Logic

    idx_h, _ = model._get_idx(state_high, 8)
    idx_l, _ = model._get_idx(state_low, 8)

    q_high = [model.Policy[n, idx_h, model._get_idx(0, 8)[1]] for n in range(model.N, -1, -1)]
    q_low = [model.Policy[n, idx_l, model._get_idx(0, 8)[1]] for n in range(model.N, -1, -1)]

    t_axis = np.linspace(0, model.T, model.N + 1)

    ax5.step(t_axis, q_high, where='post', label=f'High Inventory ($I_2={state_high}$)', linewidth=2)
    ax5.step(t_axis, q_low, where='post', label=f'Low Inventory ($I_2={state_low}$)', linewidth=2, linestyle='--')

    ax5.set_title(f"{exp['name']}: Optimal Dispatch Quantity over Time\n(Fixed Backlog $b_1=8$)")
    ax5.set_xlabel("Time ($t$)")
    ax5.set_ylabel("Dispatch Quantity ($q^*$)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{exp['name']}_action_dynamics.png")
    plt.close()

print("All experiments completed.")