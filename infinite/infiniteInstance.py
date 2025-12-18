class TransshipmentInstance:
    def __init__(self, l1, l2, h, pi1, pi2, Cf, cu, c1, c2, v2, T):
        self.l1 = l1  # Arrival rate retailer 1 [cite: 31]
        self.l2 = l2  # Arrival rate retailer 2 [cite: 31]
        self.h = h    # Holding cost retailer 2 [cite: 36]
        self.pi1 = pi1 # Backorder cost retailer 1 [cite: 36]
        self.pi2 = pi2 # Backorder cost retailer 2 [cite: 36]
        self.Cf = Cf  # Fixed transshipment cost [cite: 65, 69]
        self.cu = cu  # Unit transshipment cost [cite: 65, 69]
        self.c1 = c1  # Replenishment cost r1 at T [cite: 19]
        self.c2 = c2  # Replenishment cost r2 at T [cite: 19]
        self.v2 = v2  # Salvage value r2 at T [cite: 21]
        self.T = T    # Time horizon [cite: 4]