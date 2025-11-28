# minimalSolver.py
# Core finite-horizon MDP with AMO (NONE/A_ONLY/B_ONLY) and B-priority.
# Pure solver + simulation utilities. No plotting/printing here.

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import csv
import os


# -------------------------
# Problem instance
# -------------------------
@dataclass(frozen=True)
class Instance:
    N: int
    T: float
    lambdaA: float
    lambdaB: float
    h: float
    pA: float
    pB: float
    cf: float
    cu: float
    minIB: int
    maxIB: int
    maxbA: int
    IB0: int = 0  # initial inventory at B (user input)
    tail: float = 0.0  # unused in AMO
    salvage_v: float = 0.0  # ADDED: Salvage Value (default 0.0 = No Salvage)

    def dt(self) -> float:
        return self.T / self.N


# -------------------------
# Global toggles
# -------------------------
PAY_FIXED_ON_REALIZED: bool = True  # fixed cost charged when q_realized>0 (else on planned q>0)
CLAMP_TO_GRID: bool = True  # clamp next states into grid bounds


# -------------------------
# Demand probabilities (AMO)
# -------------------------
def computeDemandProbability_AMO(inst: Instance) -> Tuple[float, float, float]:
    # At most one event in a small dt: NONE, A_ONLY, B_ONLY (renormalized if needed)
    dt = inst.dt()
    piA = inst.lambdaA * dt
    piB = inst.lambdaB * dt
    pi0 = max(0.0, 1.0 - (piA + piB))
    s = pi0 + piA + piB
    if s <= 0.0:
        return 1.0, 0.0, 0.0
    return pi0 / s, piA / s, piB / s


# -------------------------
# Costs
# -------------------------
def computeImmediateCost_closing(IB_end: int, bA_end: int, inst: Instance) -> float:
    # End-of-period cost: dt * [ h*max(IB_end,0) + pB*max(-IB_end,0) + pA*bA_end ]
    dt = inst.dt()
    holding = inst.h * max(IB_end, 0)
    backlogB = inst.pB * max(-IB_end, 0)
    backlogA = inst.pA * bA_end
    return dt * (holding + backlogB + backlogA)


def computeTransshipmentCost(q_planned: int, q_realized: int, inst: Instance) -> float:
    # Dispatch cost = fixed + variable
    pay_fixed = (q_realized > 0) if PAY_FIXED_ON_REALIZED else (q_planned > 0)
    fixed = inst.cf if pay_fixed else 0.0
    variable = inst.cu * q_realized
    return fixed + variable


# -------------------------
# State grid
# -------------------------
def buildStateGrid(inst: Instance):
    IB_vals = np.arange(inst.minIB, inst.maxIB + 1, dtype=int)
    bA_vals = np.arange(0, inst.maxbA + 1, dtype=int)
    IB2i = {v: i for i, v in enumerate(IB_vals)}
    bA2j = {v: j for j, v in enumerate(bA_vals)}
    return IB_vals, bA_vals, IB2i, bA2j


# -------------------------
# DP solver (FORWARD in r = periods remaining)
# -------------------------
def solveDP_AMO_Bpriority_dynamic(inst: Instance) -> Dict[str, Any]:
    pi0, piA, piB = computeDemandProbability_AMO(inst)
    pis = np.array([pi0, piA, piB], dtype=float)

    IB_vals, bA_vals, IB2i, bA2j = buildStateGrid(inst)
    nI, nA = len(IB_vals), len(bA_vals)

    # --- MODIFICATION START ---
    # Initialization of Terminal Condition (V at r=0, i.e., t=T)
    V_terminal = np.zeros((nI, nA), dtype=float)

    # Factor for Theorem 1 approximation (Base Terminal Cost)
    factor = (inst.h + inst.pB) / (2 * inst.lambdaB) if inst.lambdaB > 0 else 0

    for i, IB in enumerate(IB_vals):
        for j, bA in enumerate(bA_vals):
            # 1. Calculate Base Terminal Cost (Theorem 1 Proxy)
            val_proxy = 0.0
            if IB >= 0:
                term_I2 = factor * (IB ** 2 + IB)
                term_b1 = inst.cu * bA
                term_fixed = inst.cf if bA > 0 else 0.0
                val_proxy = term_I2 + term_b1 + term_fixed

            # 2. Calculate Salvage Reward (Negative Cost)
            # If salvage_v=0, this term is 0.
            val_salvage = inst.salvage_v * max(0, IB)

            # 3. Combine: Cost = Proxy - Reward
            V_terminal[i, j] = val_proxy - val_salvage

    # Initialize V list with this terminal condition at index 0 (which is r=0)
    V = [np.zeros((nI, nA), dtype=float) for _ in range(inst.N + 1)]
    V[0] = V_terminal
    # --- MODIFICATION END ---

    # PI[r] = optimal action when r periods remain
    PI = [np.zeros((nI, nA), dtype=int) for _ in range(inst.N + 1)]

    for r in range(1, inst.N + 1):
        Vprev = V[r - 1]  # future value after this period
        for i, IB in enumerate(IB_vals):
            for j, bA in enumerate(bA_vals):
                max_feasible_q = max(0, min(IB, bA))
                best_cost = float("inf")
                best_q = 0

                for q in range(0, max_feasible_q + 1):
                    total_expected = 0.0

                    # NONE
                    q_eff = q
                    IB_end = IB - q_eff
                    bA_end = max(0, bA - q_eff)
                    if CLAMP_TO_GRID:
                        IBc = max(inst.minIB, min(inst.maxIB, IB_end))
                        bAc = max(0, min(inst.maxbA, bA_end))
                    else:
                        IBc, bAc = IB_end, bA_end
                    dc = computeTransshipmentCost(q_planned=q, q_realized=q_eff, inst=inst)
                    cc = computeImmediateCost_closing(IBc, bAc, inst)
                    total_expected += pis[0] * (dc + cc + Vprev[IB2i[IBc], bA2j[bAc]])

                    # A_ONLY
                    q_eff = q
                    IB_end = IB - q_eff
                    bA_end = max(0, bA - q_eff + 1)
                    if CLAMP_TO_GRID:
                        IBc = max(inst.minIB, min(inst.maxIB, IB_end))
                        bAc = max(0, min(inst.maxbA, bA_end))
                    else:
                        IBc, bAc = IB_end, bA_end
                    dc = computeTransshipmentCost(q_planned=q, q_realized=q_eff, inst=inst)
                    cc = computeImmediateCost_closing(IBc, bAc, inst)
                    total_expected += pis[1] * (dc + cc + Vprev[IB2i[IBc], bA2j[bAc]])

                    # B_ONLY with B priority (CORRECTED LOGIC)
                    # 1. Demand at B always reduces IB (even if < 0)
                    IB_post_demand = IB - 1

                    # 2. Can only ship if we still have positive inventory after demand
                    available_for_ship = max(0, IB_post_demand)
                    q_eff = min(q, available_for_ship)

                    # 3. Final state
                    IB_end = IB_post_demand - q_eff
                    bA_end = max(0, bA - q_eff)

                    if CLAMP_TO_GRID:
                        IBc = max(inst.minIB, min(inst.maxIB, IB_end))
                        bAc = max(0, min(inst.maxbA, bA_end))
                    else:
                        IBc, bAc = IB_end, bA_end
                    dc = computeTransshipmentCost(q_planned=q, q_realized=q_eff, inst=inst)
                    cc = computeImmediateCost_closing(IBc, bAc, inst)
                    total_expected += pis[2] * (dc + cc + Vprev[IB2i[IBc], bA2j[bAc]])

                    if total_expected < best_cost:
                        best_cost = total_expected
                        best_q = q

                PI[r][i, j] = best_q
                V[r][i, j] = best_cost

    return {
        "V": V,  # list of (nI x nA), r=0..N
        "PI": PI,  # list of (nI x nA), r=0..N
        "IB_vals": IB_vals,
        "bA_vals": bA_vals,
        "pi": (pi0, piA, piB),
    }


# -------------------------
# Cost helpers (exact and simulation)
# -------------------------
def get_optimal_expected_cost_user_t(solution, inst: Instance, t_user: int, IB: int, bA: int) -> float:
    """
    User t=0 (start) corresponds to internal r=N.
    Return V[r, IB, bA] with r = N - t_user.
    """
    r = inst.N - t_user
    if not (0 <= r <= inst.N):
        raise ValueError(f"user t out of range: t_user={t_user} -> r={r}")
    IB_vals, bA_vals = solution["IB_vals"], solution["bA_vals"]
    IBc = max(inst.minIB, min(inst.maxIB, IB))
    bAc = max(0, min(inst.maxbA, bA))
    i0 = int(np.where(IB_vals == IBc)[0][0])
    j0 = int(np.where(bA_vals == bAc)[0][0])
    return float(solution["V"][r][i0, j0])


def simulate_realized_cost(inst: Instance, solution, IB0: int, bA0: int, seed: int = 2025) -> float:
    """
    Simulate one path under the optimal policy and return the realized total cost.
    Period cost = dispatch + closing + TERMINAL COST.
    """
    rng = np.random.default_rng(seed)
    IB_vals, bA_vals = solution["IB_vals"], solution["bA_vals"]
    PI = solution["PI"]
    pi0, piA, piB = solution["pi"]

    IB = IB0
    bA = bA0
    total_cost = 0.0

    # 1. Loop through N periods
    for k in range(inst.N):
        r = inst.N - k  # periods remaining now
        IBc = max(inst.minIB, min(inst.maxIB, IB))
        bAc = max(0, min(inst.maxbA, bA))
        i = int(np.where(IB_vals == IBc)[0][0])
        j = int(np.where(bA_vals == bAc)[0][0])
        q = int(PI[r][i, j])

        # draw scenario
        u = rng.random()
        if u < pi0:
            scen = 0
        elif u < pi0 + piA:
            scen = 1
        else:
            scen = 2

        # evolve (B priority)
        if scen == 0:
            q_eff = q
            IB_next = IB - q_eff
            bA_next = max(0, bA - q_eff)
        elif scen == 1:
            q_eff = q
            IB_next = IB - q_eff
            bA_next = max(0, bA - q_eff + 1)
        else:  # scen 2 (B_ONLY) - CORRECTED LOGIC
            IB_post_demand = IB - 1
            available_for_ship = max(0, IB_post_demand)
            q_eff = min(q, available_for_ship)
            IB_next = IB_post_demand - q_eff
            bA_next = max(0, bA - q_eff)

        if CLAMP_TO_GRID:
            IB_next = max(inst.minIB, min(inst.maxIB, IB_next))
            bA_next = max(0, min(inst.maxbA, bA_next))

        dispatch_cost = computeTransshipmentCost(q_planned=q, q_realized=q_eff, inst=inst)
        closing_cost = computeImmediateCost_closing(IB_next, bA_next, inst)
        total_cost += (dispatch_cost + closing_cost)

        IB, bA = IB_next, bA_next

    # 2. TERMINAL COST LOGIC (Matching DP)
    if inst.lambdaB > 0:
        factor = (inst.h + inst.pB) / (2 * inst.lambdaB)
    else:
        factor = 0.0

    # Base Terminal Cost (Theorem 1 Proxy)
    terminal_cost = 0.0
    if IB >= 0:
        term_I2 = factor * (IB ** 2 + IB)
        term_b1 = inst.cu * bA
        term_fixed = inst.cf if bA > 0 else 0.0
        terminal_cost = term_I2 + term_b1 + term_fixed

    # Subtract Salvage Value (if salvage_v > 0, this reduces cost)
    terminal_cost -= inst.salvage_v * max(0, IB)

    total_cost += terminal_cost

    return float(total_cost)


def run_simulations(inst: Instance, solution, simN: int, base_seed: int, IB0: int, bA0: int):
    """
    Run simN independent paths with seeds base_seed + r.
    Return (mean, std, ci_low, ci_high, all_costs).
    CI uses normal approx (z=1.96).
    """
    costs = []
    for r in range(simN):
        seed = base_seed + r
        c = simulate_realized_cost(inst, solution, IB0=IB0, bA0=bA0, seed=seed)
        costs.append(c)

    costs = np.array(costs, dtype=float)
    mean = float(costs.mean())
    std = float(costs.std(ddof=1)) if simN > 1 else 0.0
    z = 1.96  # 95% normal approx
    half_width = z * std / (simN ** 0.5) if simN > 1 else 0.0
    ci_low, ci_high = mean - half_width, mean + half_width
    return mean, std, ci_low, ci_high, costs


# -------------------------
# Persist simulation results (refresh-only)
# -------------------------
def append_sim_results(outfile: str, label: str, base_seed: int, costs: np.ndarray):
    """
    Append each simulation result (seed, cost) to a CSV, and also append one summary row.
    Schema:
        label, kind, run_idx, seed, cost, mean, std, ci_low, ci_high
    """
    file_exists = os.path.exists(outfile)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["label", "kind", "run_idx", "seed", "cost", "mean", "std", "ci_low", "ci_high"])
        # per-run rows
        for idx, c in enumerate(costs):
            w.writerow([label, "run", idx, base_seed + idx, f"{float(c):.6f}", "", "", "", ""])
        # summary row
        if len(costs) > 0:
            mean = float(costs.mean())
            std = float(costs.std(ddof=1)) if len(costs) > 1 else 0.0
            z = 1.96
            half = z * std / (len(costs) ** 0.5) if len(costs) > 1 else 0.0
            lo, hi = mean - half, mean + half
            w.writerow([label, "summary", "", "", "", f"{mean:.6f}", f"{std:.6f}", f"{lo:.6f}", f"{hi:.6f}"])


__all__ = [
    "Instance",
    "PAY_FIXED_ON_REALIZED",
    "CLAMP_TO_GRID",
    "solveDP_AMO_Bpriority_dynamic",
    "get_optimal_expected_cost_user_t",
    "simulate_realized_cost",
    "run_simulations",
    "append_sim_results",
]