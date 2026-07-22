"""
simulator.py — Monte Carlo evaluation engine.

Design decision (cost alignment with the SDP solver):
  The simulation does NOT generate exact continuous-time Poisson paths.
  It simulates the IDENTICAL at-most-one-event discretised chain that the
  backward-induction DP in ../solver.py solves:

      transition:   p1 = lam1·dt → b1+1,   p2 = lam2·dt → I2−1,   p0 → stay
      period cost:  g(I2, b1, q) = Cf·1{q>0} + cu·q
                      + dt·( h·(I2−q)^+ + pi1·(b1−q) + pi2·(I2−q)^− )
                    with the flow cost on the POST-dispatch state,
      terminal:     c1·b1 + c2·(−I2)^+ − v2·I2^+,
      clipping:     b1 ∈ [0, b1_max],  I2 ∈ [I2_min, I2_max],

  so that a policy's simulated expected cost and the DP value V^N are
  expectations of the same functional on the same probability space.
  Any gap between a benchmark and V^N is then attributable to the policy
  alone, with no discretisation mismatch.

  This module takes the Params object as an argument and never imports
  solver itself, so it has no dependency on the parent-folder path setup.

Common random numbers:
  make_uniforms() draws one (n_reps × N) array of U(0,1) variates.  The
  SAME array is passed to every policy and every parameter value, so the
  estimated cost curves C(Q) and C(Delta) are smooth and directly
  comparable across parameters and across policies.
"""

import numpy as np


def make_uniforms(n_reps, n_steps, seed):
    """One shared CRN array for the whole study."""
    rng = np.random.default_rng(seed)
    return rng.random((n_reps, n_steps))


def simulate(policy, p, U, I2_init, b1_init):
    """
    Evaluate one policy on the shared CRN array.

    Parameters
    ----------
    policy  : base.Policy
    p       : solver.Params            (same instance as the DP)
    U       : (n_reps, N) uniforms     (from make_uniforms)
    I2_init, b1_init : initial state

    Returns
    -------
    dict with per-replication arrays:
      total, fixed, variable, holding, penalty1, penalty2, terminal,
      n_dispatch
    """
    n_reps, n_steps = U.shape
    assert n_steps == p.N, "CRN array must have exactly N columns."

    I2 = np.full(n_reps, I2_init, dtype=np.int64)
    b1 = np.full(n_reps, b1_init, dtype=np.int64)

    cost_fixed = np.zeros(n_reps)
    cost_var   = np.zeros(n_reps)
    cost_hold  = np.zeros(n_reps)
    cost_pen1  = np.zeros(n_reps)
    cost_pen2  = np.zeros(n_reps)
    n_disp     = np.zeros(n_reps, dtype=np.int64)

    for step in range(p.N):
        n_rem = p.N - step

        # ── dispatch decision at period start ──
        q = policy.decide(step, n_rem, I2, b1)
        # Feasibility clip (safety net; mirrors the DP's action set
        # q ∈ {0,…,min(I2,b1)} when I2>0 and b1>0, else q=0):
        q = np.clip(q, 0, None)
        q = np.minimum(q, np.minimum(np.maximum(I2, 0), np.maximum(b1, 0)))

        I2a = I2 - q                     # post-dispatch state
        b1a = b1 - q

        # ── period cost, identical to TransshipmentDP.g ──
        pos = q > 0
        cost_fixed += p.Cf * pos
        cost_var   += p.cu * q
        cost_hold  += p.dt * p.h   * np.maximum(I2a, 0)
        cost_pen1  += p.dt * p.pi1 * b1a
        cost_pen2  += p.dt * p.pi2 * np.maximum(-I2a, 0)
        n_disp     += pos

        # ── transition: at most one event, same p1/p2/p0 as the DP ──
        u    = U[:, step]
        arr1 = u < p.p1                              # retailer-1 arrival
        arr2 = (~arr1) & (u < p.p1 + p.p2)           # retailer-2 arrival
        b1 = b1a + arr1
        I2 = I2a - arr2

        # ── boundary clipping, identical to the DP's state space ──
        np.clip(b1, 0, p.b1_max, out=b1)
        np.clip(I2, p.I2_min, p.I2_max, out=I2)

    # ── terminal clean-up, identical to TransshipmentDP.terminal ──
    cost_term = (p.c1 * b1
                 + p.c2 * np.maximum(-I2, 0)
                 - p.v2 * np.maximum(I2, 0)).astype(np.float64)

    total = (cost_fixed + cost_var + cost_hold
             + cost_pen1 + cost_pen2 + cost_term)

    return {
        "total":      total,
        "fixed":      cost_fixed,
        "variable":   cost_var,
        "holding":    cost_hold,
        "penalty1":   cost_pen1,
        "penalty2":   cost_pen2,
        "terminal":   cost_term,
        "n_dispatch": n_disp,
    }


def summarise(raw):
    """Mean, std, and 95% CI half-width for every cost component."""
    n = len(raw["total"])
    out = {"n_reps": n}
    for key, arr in raw.items():
        arr = np.asarray(arr, dtype=np.float64)
        mean = float(arr.mean())
        std  = float(arr.std(ddof=1))
        out[key] = {
            "mean": mean,
            "std":  std,
            "ci95": 1.96 * std / np.sqrt(n),
        }
    return out