import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
# The following import is required for 3D plotting
from mpl_toolkits.mplot3d import Axes3D


class TransshipmentSolver:
    def __init__(self, instance):
        self.inst = instance
        self.n_I2, self.n_b1 = self.inst.get_state_dims()

        # Grid: Rows=I2, Cols=b1
        self.I2_grid, self.b1_grid = np.meshgrid(
            np.arange(self.inst.i_min, self.inst.i_max + 1),
            np.arange(0, self.inst.b_max + 1),
            indexing='ij'
        )

        self.step_holding_cost = self.inst.dt * (
                self.inst.h * np.maximum(0, self.I2_grid) +
                self.inst.pi_1 * self.b1_grid +
                self.inst.pi_2 * np.maximum(0, -self.I2_grid)
        )

        self.V_tables = []
        self.Policy_tables = []

    def solve(self):
        start_t = time.time()

        # Terminal Condition
        V_next = -self.inst.c2 * self.I2_grid.astype(float)

        V_list = [V_next.copy()]
        Policy_list = [np.zeros_like(self.I2_grid, dtype=int)]

        print(f"Solving {self.inst.N} steps...")

        for n in range(1, self.inst.N + 1):
            V_curr = np.full(V_next.shape, np.inf)
            P_curr = np.zeros(V_next.shape, dtype=int)

            max_q_global = min(self.inst.i_max, self.inst.b_max)

            for q in range(max_q_global + 1):
                if q == 0:
                    valid_mask = np.ones(V_curr.shape, dtype=bool)
                else:
                    valid_mask = (self.I2_grid >= q) & (self.b1_grid >= q)

                if not np.any(valid_mask): continue

                trans_cost = self.inst.Cf + (self.inst.cu * q) if q > 0 else 0.0

                I2_post = self.I2_grid - q
                b1_post = self.b1_grid - q

                # Fast Vectorized Expectation
                idx_I2_base = (I2_post - self.inst.i_min).astype(int)
                idx_b1_base = b1_post.astype(int)

                def clamp(idx, max_d):
                    return np.clip(idx, 0, max_d - 1)

                idx_I2_0 = clamp(idx_I2_base, self.n_I2)
                idx_b1_0 = clamp(idx_b1_base, self.n_b1)
                idx_b1_p1 = clamp(idx_b1_base + 1, self.n_b1)
                idx_I2_p2 = clamp(idx_I2_base - 1, self.n_I2)

                v_stay = V_next[idx_I2_0, idx_b1_0]
                v_dem1 = V_next[idx_I2_0, idx_b1_p1]
                v_dem2 = V_next[idx_I2_p2, idx_b1_0]

                expected_future = (self.inst.p0 * v_stay) + \
                                  (self.inst.p1 * v_dem1) + \
                                  (self.inst.p2 * v_dem2)

                total_cost_q = trans_cost + self.step_holding_cost + expected_future

                # Update Min
                update_mask = valid_mask & (total_cost_q < V_curr)
                V_curr[update_mask] = total_cost_q[update_mask]
                P_curr[update_mask] = q

            V_list.append(V_curr)
            Policy_list.append(P_curr)
            V_next = V_curr

        print(f"Solved in {time.time() - start_t:.4f}s.")
        self.V_tables = list(reversed(V_list))
        self.Policy_tables = list(reversed(Policy_list))
        return self.V_tables, self.Policy_tables

    def save_policy_to_csv(self, time_idx, filename):
        if not self.Policy_tables: return
        print(f"Saving policy matrix for t={time_idx} to {filename}...")

        policy_matrix = self.Policy_tables[time_idx]
        policy_T = policy_matrix.T  # Transpose for I2 (Horz) vs b1 (Vert)

        real_I2 = np.arange(self.inst.i_min, self.inst.i_max + 1)
        real_b1 = np.arange(0, self.inst.b_max + 1)

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["b1_down_I2_right"] + [f"I2={i}" for i in real_I2]
            writer.writerow(header)
            for r_idx, b1_val in enumerate(real_b1):
                row_data = policy_T[r_idx, :]
                row_str = [f"b1={b1_val}"] + [str(q) for q in row_data]
                writer.writerow(row_str)
        print("Done.")

    def save_thresholds_to_csv(self, filename):
        if not self.Policy_tables: return
        print(f"Refreshing {filename}...")
        real_b1 = np.arange(0, self.inst.b_max + 1)
        real_I2 = np.arange(self.inst.i_min, self.inst.i_max + 1)

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Time_Step", "Backlog_Level_b1", "Trigger_Inventory_I2"])
            for t, policy in enumerate(self.Policy_tables):
                for col_idx, b1_val in enumerate(real_b1):
                    q_col = policy[:, col_idx]
                    indices = np.where(q_col > 0)[0]
                    if indices.size > 0:
                        writer.writerow([t, b1_val, real_I2[indices[0]]])
                    else:
                        writer.writerow([t, b1_val, "No_Transshipment"])
        print("Done.")

    # ==========================
    # PLOTTING
    # ==========================
    def plot_switching_curve(self, time_idx=0):
        """
        Generates Switching Curve with explicit GREY DOTS for Wait Zones.
        """
        if not self.Policy_tables:
            print("Error: Model not solved.")
            return

        print(f"Generating Switching Curve for t={time_idx}...")

        policy = self.Policy_tables[time_idx]

        real_I2 = np.arange(self.inst.i_min, self.inst.i_max + 1)
        real_b1 = np.arange(0, self.inst.b_max + 1)

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
            # Find every index in this column where q == 0
            wait_indices = np.where(q_col == 0)[0]
            for idx in wait_indices:
                i2_val = real_I2[idx]
                # Only plot dots within the requested view (non-negative I2)
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

        # Plotting
        plt.figure(figsize=(10, 6))

        # A. Plot the Grey Dots (Background Grid)
        # This fills the "empty" areas
        plt.scatter(wait_x, wait_y, color='lightgray', marker='.', s=50, label='Wait ($q=0$)')

        # B. Plot the Lines
        plt.plot(x_b1, y_switch_I2, 'o-', color='navy', linewidth=2, label='Start Shipping ($q>0$)')
        plt.plot(x_b1, y_full_I2, 's--', color='green', linewidth=2, label='Clear Backlog ($q=b_1$)')

        # C. Fill Regions
        valid_mask = np.isfinite(y_switch_I2) & np.isfinite(y_full_I2)
        if np.any(valid_mask):
            x_arr = np.array(x_b1)[valid_mask]
            y_sw = np.array(y_switch_I2)[valid_mask]
            y_fl = np.array(y_full_I2)[valid_mask]

            # Rationing Region (Orange)
            plt.fill_between(x_arr, y_sw, y_fl, color='orange', alpha=0.2, label='Rationing Region')

            # Wait Region (Red Tint)
            # We explicitly fill from 0 up to the blue line
            plt.fill_between(x_arr, 0, y_sw, color='red', alpha=0.1, label='Wait Region')

        plt.xlabel('Retailer 1 Backlog ($b_1$)')
        plt.ylabel('Retailer 2 Inventory ($I_2$)')
        plt.title(f'Switching & Full-Satisfaction Thresholds (t={time_idx})')

        # Scale Adjustment
        plt.xlim(0, self.inst.b_max)
        plt.ylim(0, self.inst.i_max)

        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.show()




    def plot_3d_policy_surface(self, time_idx=0):
        """
        Plots 3D Surface of optimal Quantity q vs (I2, b1).
        """
        if not self.Policy_tables: return
        print(f"Generating 3D Policy Surface for t={time_idx}...")

        policy = self.Policy_tables[time_idx]
        real_I2 = np.arange(self.inst.i_min, self.inst.i_max + 1)
        real_b1 = np.arange(0, self.inst.b_max + 1)

        X, Y = np.meshgrid(real_I2, real_b1)
        Z = policy.T

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                               rstride=1, cstride=1,
                               linewidth=0.5, edgecolor='k', alpha=0.85)

        # Floor Contour
        floor_val = np.min(Z)
        ax.contourf(X, Y, Z, zdir='z', offset=floor_val, cmap=cm.viridis, alpha=0.4)

        ax.set_xlabel('Inventory ($I_2$)')
        ax.set_ylabel('Backlog ($b_1$)')
        ax.set_zlabel('Quantity ($q$)')
        ax.set_title(f'Optimal Policy Surface (t={time_idx})')
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Transshipment Quantity')

        ax.set_zlim(floor_val, np.max(Z))
        ax.view_init(elev=30, azim=-125)
        plt.show()