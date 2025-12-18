import numpy as np


class InfiniteSolver:
    def __init__(self, instance):
        self.inst = instance

    def get_waiting_value(self, I2, b1, tau):
        # Implementation of Eq (6) / Eq (86) [cite: 86]
        term1 = ((self.inst.h + self.inst.pi2) / (2 * self.inst.l2)) * (I2 ** 2 + I2)
        term2 = -self.inst.pi2 * tau * I2
        term3 = self.inst.pi1 * tau * b1
        term4 = 0.5 * (self.inst.pi1 * self.inst.l1 + self.inst.pi2 * self.inst.l2) * (tau ** 2)
        term5 = self.inst.c1 * (b1 + self.inst.l1 * tau) + self.inst.c2 * (self.inst.l2 * tau - I2)
        return term1 + term2 + term3 + term4 + term5

    def get_optimal_q(self, I2, b1, tau):
        # Implementation of the axis of symmetry from page 5 [cite: 111, 114]
        numerator = self.inst.l2 * ((self.inst.pi1 - self.inst.pi2) * tau - self.inst.cu + self.inst.c1 - self.inst.c2)
        denominator = self.inst.h + self.inst.pi2
        symmetry_axis = (numerator / denominator) + I2 + 0.5

        limit = min(I2, b1)
        if limit <= symmetry_axis:
            return limit  # [cite: 112]
        else:
            return max(1, int(round(symmetry_axis)))  # [cite: 116]

    def solve_policy(self, I2_range, b1_range, tau):
        policy_map = {}
        for I2 in I2_range:
            for b1 in b1_range:
                if I2 <= 0 or b1 <= 0:
                    policy_map[(I2, b1)] = ("WAIT", 0)  # [cite: 14]
                    continue

                V_w = self.get_waiting_value(I2, b1, tau)
                q_star = self.get_optimal_q(I2, b1, tau)
                # Dispatch value formula [cite: 100, 108]
                V_d = self.inst.Cf + self.inst.cu * q_star + self.get_waiting_value(I2 - q_star, b1 - q_star, tau)

                if V_d < V_w:  #
                    policy_map[(I2, b1)] = ("DISPATCH", q_star)
                else:
                    policy_map[(I2, b1)] = ("WAIT", 0)
        return policy_map