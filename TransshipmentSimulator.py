import numpy as np
import scipy.stats as stats


class TransshipmentSimulator:
    """
    Validates the MDP policy using Continuous-Time Discrete-Event Simulation (DES).
    """

    def __init__(self, instance, solver):
        self.inst = instance
        self.solver = solver
        self.policy_tables = solver.Policy_tables

    def simulate_single_path(self, seed=None):
        """
        Runs one sample path of the horizon [0, T] using Next-Event Logic.
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Initialize State
        t = 0.0
        I2 = self.inst.I2_0
        b1 = self.inst.b1_0
        total_cost = 0.0

        # Helper to map state to grid index
        def get_indices(curr_I2, curr_b1):
            # Clamp to grid boundaries for policy lookup
            i_idx = np.clip(curr_I2 - self.inst.i_min, 0, self.solver.n_I2 - 1)
            b_idx = np.clip(curr_b1, 0, self.solver.n_b1 - 1)
            return i_idx, b_idx

        # --- SIMULATION LOOP ---
        while t < self.inst.T:

            # A. Check Policy at CURRENT moment
            # We map continuous time t to the nearest discrete step index n
            # Time 0 -> Index 0. Time T -> Index N.
            time_idx = int((t / self.inst.T) * self.inst.N)
            time_idx = min(time_idx, self.inst.N - 1)

            i_idx, b_idx = get_indices(I2, b1)

            # Get optimal quantity from MDP result
            q = self.policy_tables[time_idx][i_idx, b_idx]

            # If policy says ship, do it INSTANTLY (impulse control)
            if q > 0:
                # Ensure we actually have stock/backlog (simulation ground truth)
                real_q = min(q, max(0, I2), b1)

                if real_q > 0:
                    # 1. Pay Transshipment Cost
                    total_cost += self.inst.Cf + (self.inst.cu * real_q)

                    # 2. Update State
                    I2 -= real_q
                    b1 -= real_q

                    # 3. Continue loop without advancing time (we might need to ship again or re-eval)
                    # However, strictly, the policy usually clears enough to wait.
                    continue

            # B. Generate Time to Next Event (Exponential)
            # Rate of events = lambda1 + lambda2
            total_rate = self.inst.lambda_1 + self.inst.lambda_2

            # Time until next arrival
            dt_next = np.random.exponential(1.0 / total_rate)

            # C. Advance Time
            t_next = t + dt_next

            # If the next event is beyond Horizon T, clamp it
            if t_next > self.inst.T:
                dt_step = self.inst.T - t
                t = self.inst.T
                # Add holding cost for this final segment
                step_cost = self._calc_holding_rate(I2, b1) * dt_step
                total_cost += step_cost
                break  # End of Horizon

            # D. Add Holding/Penalty Cost for the duration dt_next
            # Cost = Rate * Time
            step_cost = self._calc_holding_rate(I2, b1) * dt_next
            total_cost += step_cost

            # E. Execute Event (Demand Arrival)
            # Determine which event happened: R1 Demand or R2 Demand?
            # Prob(R1) = lambda1 / total_rate
            rand_check = np.random.rand()
            p_r1 = self.inst.lambda_1 / total_rate

            if rand_check < p_r1:
                # Event: Demand at Retailer 1 -> Backlog increases
                b1 += 1
            else:
                # Event: Demand at Retailer 2 -> Inventory decreases
                # Strict Priority: If I2 > 0, satisfy demand. If I2 <= 0, backorder?
                # Note: Minimal model says I2 can be negative.
                I2 -= 1

            # Update clock
            t = t_next

        # F. Terminal Cost (Salvage)
        if self.inst.c2 > 0:
            # We recover value for remaining inventory
            # Note: We subtract from cost (gain)
            total_cost -= (self.inst.c2 * I2)

        return total_cost

    def _calc_holding_rate(self, I2, b1):
        """
        Calculates cost rate per unit time: h*I2+ + pi1*b1 + pi2*I2-
        """
        pos_I2 = max(0, I2)
        neg_I2 = max(0, -I2)

        cost_rate = (self.inst.h * pos_I2) + \
                    (self.inst.pi_1 * b1) + \
                    (self.inst.pi_2 * neg_I2)
        return cost_rate

    def run_monte_carlo(self, num_simulations=1000):
        """
        Runs multiple paths and returns statistics.
        """
        costs = []
        print(f"Running {num_simulations} Continuous-Time simulations...")

        for i in range(num_simulations):
            c = self.simulate_single_path(seed=i)
            costs.append(c)

        costs = np.array(costs)
        mean_cost = np.mean(costs)
        std_error = np.std(costs, ddof=1) / np.sqrt(num_simulations)
        conf_interval = 1.96 * std_error  # 95% CI

        return mean_cost, conf_interval