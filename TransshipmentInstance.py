import numpy as np


class TransshipmentInstance:
    """
    Stores parameters and pre-calculated probabilities for the minimal
    transshipment model (Model I and Model II).

    Attributes:
        T (float): Planning horizon length (e.g., 1.0).
        N (int): Number of time steps.
        dt (float): Time step length (T/N).
        lambda_1 (float): Demand rate at Retailer 1 (backlog generator).
        lambda_2 (float): Demand rate at Retailer 2 (inventory holder).
        h (float): Unit holding cost.
        pi_1 (float): Unit penalty cost for Retailer 1 backlog.
        pi_2 (float): Unit penalty cost for Retailer 2 backlog.
        Cf (float): Fixed transshipment cost.
        cu (float): Unit transshipment cost.
        c2 (float): Salvage value per unit.
                    Set c2=0 for Model I (No Salvage).
                    et c2>0 for Model II (With Salvage).

        # Derived Probabilities
        p1 (float): Probability of demand at Retailer 1 (lambda_1 * dt).
        p2 (float): Probability of demand at Retailer 2 (lambda_2 * dt).
        p0 (float): Probability of no event (1 - p1 - p2).

        # State Space Boundaries
        i_min (int): Min inventory for Retailer 2 (e.g., -20).
        i_max (int): Max inventory for Retailer 2 (e.g., 20).
        b_max (int): Max backlog for Retailer 1 (e.g., 20).

        # Initial inventory level of Retailer 2
        I2_0 (int): Initial inventory level at Retailer 2.
    """

    def __init__(self,
                 T: float, N: int,
                 lambda_1: float, lambda_2: float,
                 h: float, pi_1: float, pi_2: float,
                 Cf: float, cu: float,
                 c2: float = 0.0,
                 i_min: int = -20, i_max: int = 20, b_max: int = 20,
                 I2_0: int = None,
                 b1_0: int = 0):
        # System Parameters
        self.T = T
        self.N = N
        self.dt = T / N

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # Costs
        self.h = h
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.Cf = Cf
        self.cu = cu
        self.c2 = c2
        # c2=0 -> Model I;
        # c2>0 -> Model II

        # Grid Boundaries
        self.i_min = i_min
        self.i_max = i_max
        self.b_max = b_max

        # Initial States
        if I2_0 is None:
            self.I2_0 = self.i_max
        else:
            self.I2_0 = I2_0
        self.b1_0 = b1_0


        # Probability Calculations
        self.p1 = self.lambda_1 * self.dt
        self.p2 = self.lambda_2 * self.dt
        self.p0 = 1.0 - (self.p1 + self.p2)

        # Verify Model Stability immediately upon creation
        self._validate_stability()

    def _validate_stability(self):
        """
        Ensures delta_t is small enough that p0 >= 0.
        If p1 + p2 > 1, the discrete approximation fails.
        """
        if self.p0 < 0:
            raise ValueError(
                f"Unstable parameters: Total event prob ({self.p1 + self.p2:.4f}) > 1. "
                f"Increase N (current: {self.N}) to reduce dt."
            )

    def get_state_dims(self):
        """Returns grid dimensions for (I2, b1)."""
        n_I2 = self.i_max - self.i_min + 1
        n_b1 = self.b_max + 1
        return n_I2, n_b1

    def __repr__(self):
        model_type = "Model II (With Salvage -c_2*I_2)" if self.c2 > 0 else "Model I (No Salvage)"
        return (f"Instance: Start State=({self.I2_0}, {self.b1_0}) | "
                f"T={self.T}, N={self.N}, dt={self.dt:.4f} | "
                f"Cf={self.Cf}, cu={self.cu:.1f}, h={self.h:.1f}, (pi_1,pi_2)=({self.pi_1:.1f},{self.pi_2:.1f}) | "
                f"Model={'Salvage' if self.c2>0 else 'No Salvage'}")

'''
# --- TEST EXAMPLE ---
if __name__ == "__main__":
    print("--- Test 1: Model I (No Salvage) Setup ---")
    try:
        # Create instance for Model I (c2=0)
        inst_1 = TransshipmentInstance(
            T=1.0, N=200,
            lambda_1=8.0, lambda_2=10.0,
            h=0.1, pi_1=10.0, pi_2=10.0,
            Cf=20.0, cu=1,
            c2=0.0,  # Model I
            i_min=-30, i_max=30, b_max=30
        )
        print("Successfully created:", inst_1)
        print(f"Probabilities -> p1: {inst_1.p1}, p2: {inst_1.p2}, p0: {inst_1.p0}")

    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test 2: Model II (With Salvage) Setup ---")
    inst_2 = TransshipmentInstance(
        T=1.0, N=200,
        lambda_1=8.0, lambda_2=10.0,
        h=0.1, pi_1=10.0, pi_2=10.0,
        Cf=20.0, cu=1,
        c2=5.0,  # Model II salvage value
        i_min=-30, i_max=30, b_max=30
    )
    print("Successfully created:", inst_2)
'''