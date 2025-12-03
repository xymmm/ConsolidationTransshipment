class TransshipmentInstance:
    """
    Stores parameters and pre-calculated probabilities for the minimal
    transshipment model (Model I and Model II).
    Supports generalized terminal conditions.
    """

    def __init__(self,
                 T: float, N: int,
                 lambda_1: float, lambda_2: float,
                 h: float, pi_1: float, pi_2: float,
                 Cf: float, cu: float,
                 # Generalized Terminal Parameters
                 # c2: Salvage value per unit of inventory (Benefit)
                 # pi_end: Penalty per unit of backlog at time T (Cost)
                 c2: float = 0.0,
                 pi_end: float = 0.0,

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

        # Terminal Condition Parameters
        self.c2 = c2
        self.pi_end = pi_end

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

        # Verify Model Stability
        self._validate_stability()

    def _validate_stability(self):
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
                f"Model={'Salvage' if self.c2>0 else 'No Salvage'} | "
                f"Terminal=(c2={self.c2}, pi_end={self.pi_end})"
                )


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
            c2=0.0,  pi_end = 0,# Model I
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
        c2=5.0,  pi_end = 2,# Model II salvage value
        i_min=-30, i_max=30, b_max=30
    )
    print("Successfully created:", inst_2)
