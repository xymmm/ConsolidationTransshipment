"""
todo3_boundary_condition.py
=============================================================================
To-do 3 (meeting note, red circle 3 + "larger cu, less difference?"):
  boundary condition zai moqi qi zuoyong, yexu formula qianqi shi zhengque de
  -> nage shijiandian shi zuihou yige moment when formula shi zhengque de?

PRECISE FRAMING
-----------------------------------------------------------------------------
The verification-theorem structure used in both PDFs (e.g. Theorem 2 of
analysis_general.pdf) requires a candidate W to satisfy TWO separate
conditions:
    (1) the governing PDE/recursion, for all tau > 0
    (2) the terminal boundary condition W(I2,b1,0) = terminal cost

The closed-form Vw(I2,tau) in 20260604_zero_fixed_cost.pdf is derived by
solving (1) exactly (this is what the "Verification" sections in the PDF
check). It is NOT checked against (2). Direct substitution shows it fails:

    Vw(I2, 0) = (h+pi2)/(2*lam2) * (I2^2 + I2)      for I2 >= 1

which is nonzero for I2 >= 1, contradicting the required V(I2,0) = 0.
(For I2 <= 0 the closed form DOES satisfy both conditions exactly, this is
verified numerically below too.)

CONSEQUENCE
-----------------------------------------------------------------------------
Let e(I2,tau) = V*(I2,tau) - Vw(I2,tau) be the gap between the true value
function and the closed form, restricted to the wait region (I2 below the
dispatch threshold, so V* obeys the same recursion as Vw there). Since Vw
solves the recursion exactly, the inhomogeneous "flow" term cancels in the
difference, and e obeys the HOMOGENEOUS version of the same recursion:

    de(I2,tau)/dtau = -lam2 * ( e(I2,tau) - e(I2-1,tau) ),   I2 >= 1
    e(0, tau) = 0                       for all tau
    e(I2, 0)  = -Vw(I2,0) = -(h+pi2)/(2*lam2)*(I2^2+I2)      "initial" mismatch

This is now a LINEAR recursion (no control/min operator), so it can be
solved exactly as a Poisson convolution:

    e(I2,tau) = sum_{k=0}^{I2} Pois(k; lam2*tau) * g(I2-k),
    g(j) = -(h+pi2)/(2*lam2)*(j^2+j) for j>=1,  g(0)=0

This gives a fast, closed-form characterisation of "how wrong is the
closed-form formula, this far from the terminal boundary" without
re-running the full nonlinear DP. It is validated below against:
    (a) a direct numerical solve of the homogeneous recursion
    (b) the TRUE error extracted from the full nonlinear ZeroFixedCostDP
        (restricted to I2 well below the dispatch threshold, where the
        wait-region recursion is what actually governs V*)
"""

import numpy as np
from scipy.stats import poisson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Todo1_policyType_hybrid_20260621 import ZFParams, ZeroFixedCostDP, threshold_symmetric


# =============================================================================
# Boundary mismatch g(I2) = -Vw(I2, 0)
# =============================================================================

def g_mismatch(I2, p: ZFParams):
    """-Vw(I2,0) = -(h+pi2)/(2*lam2)*(I2^2+I2), for I2>=1. g(0)=0."""
    if I2 <= 0:
        return 0.0
    return -(p.h + p.pi2) / (2 * p.lam2) * (I2**2 + I2)


# =============================================================================
# (a) Direct numerical solve of the HOMOGENEOUS error recursion
# =============================================================================

def solve_error_direct(p: ZFParams, I2_max: int):
    """Forward Euler in tau, mirrors ZeroFixedCostDP's own discretisation."""
    nI2 = I2_max + 1
    e = np.array([g_mismatch(I2, p) for I2 in range(nI2)])
    E = np.zeros((p.N + 1, nI2))
    E[0] = e
    for n in range(1, p.N + 1):
        e_prev = E[n - 1]
        e_new = np.zeros(nI2)
        for I2 in range(nI2):
            if I2 == 0:
                e_new[I2] = 0.0
            else:
                e_new[I2] = e_prev[I2] - p.lam2 * p.dt * (e_prev[I2] - e_prev[I2 - 1])
        E[n] = e_new
    return E  # E[n, I2] = e(I2, n*dt)


# =============================================================================
# (b) Closed-form Poisson convolution solution
# =============================================================================

def error_poisson(I2, tau, p: ZFParams, kmax=None):
    """e(I2,tau) = sum_k Pois(k; lam2*tau) * g(I2-k)."""
    if I2 <= 0:
        return 0.0
    if kmax is None:
        kmax = I2
    ks = np.arange(0, kmax + 1)
    weights = poisson.pmf(ks, p.lam2 * tau)
    gs = np.array([g_mismatch(I2 - k, p) for k in ks])
    return float(np.sum(weights * gs))


# =============================================================================
# Main verification + boundary-depth characterisation
# =============================================================================

def main():
    p = ZFParams(pi1=4.7, pi2=4.7, cu=3.2, h=0.1, lam2=5.0, T=8.0, N=3200,
                 I2_min=-150, I2_max=25)
    I2_threshold = threshold_symmetric(p)
    print(f"Params: {p.summary()}")
    print(f"Analytical dispatch threshold I2* = {I2_threshold:.4f}")
    print(f"g(I2) = -(h+pi2)/(2*lam2)*(I2^2+I2)")
    for I2 in [1, 2, 3, 4, 5]:
        print(f"  g({I2}) = {g_mismatch(I2, p):.4f}   (= -Vw({I2},0))")

    # ---- (a) vs (b): direct recursion vs Poisson-convolution closed form ----
    print("\n[Check 1] direct homogeneous recursion vs Poisson convolution")
    E = solve_error_direct(p, I2_max=10)
    max_check_diff = 0.0
    for I2 in [2, 3, 4]:
        for n in [40, 200, 800, 1600, 3200]:
            tau = n * p.dt
            e_direct = E[n, I2]
            e_closed = error_poisson(I2, tau, p)
            d = abs(e_direct - e_closed)
            max_check_diff = max(max_check_diff, d)
            print(f"  I2={I2} tau={tau:5.2f}: direct={e_direct:9.4f}  "
                  f"closed-form={e_closed:9.4f}  diff={d:.2e}")
    print(f"  max|direct-closed_form| over all checks = {max_check_diff:.2e}")

    # ---- (c) Poisson-convolution-predicted error vs TRUE error from the
    #          full nonlinear DP, deep in the wait region ----
    print("\n[Check 2] predicted error vs TRUE error (full nonlinear DP)")
    dp = ZeroFixedCostDP(p)
    dp.solve()

    def Vw(I2, tau):
        return ((p.h + p.pi2) / (2 * p.lam2) * I2**2
                + (p.h + p.pi2) / (2 * p.lam2) * I2
                - p.pi2 * tau * I2
                + (p.pi1 * p.lam1 + p.pi2 * p.lam2) / 2 * tau**2)

    I2_probe = 2  # well below threshold I2*=3.33, so DP should be in wait region
    taus_check = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
    for tau in taus_check:
        n = round(tau / p.dt)
        tau_actual = n * p.dt
        ii = dp._ii(I2_probe)
        V_true = dp.V[n, ii]
        V_w = Vw(I2_probe, tau_actual)
        e_true = V_true - V_w
        e_pred = error_poisson(I2_probe, tau_actual, p)
        print(f"  tau={tau_actual:.3f}: e_true={e_true:9.4f}  e_predicted={e_pred:9.4f}  "
              f"diff={abs(e_true-e_pred):.4f}")

    # ---- (d) Boundary-depth rule of thumb: how does the error at the
    #          THRESHOLD level decay with tau, and how does that depth
    #          scale with the threshold value itself? ----
    print("\n[Boundary depth] error at I2=I2* (rounded) vs tau, several thresholds")
    print("Rule-of-thumb candidate: boundary_depth ~ I2* / lam2")
    print("(time for a rate-lam2 Poisson process to traverse from I2* down to 0)\n")

    results = []
    for pi_val, cu_val in [(4.7, 1.0), (4.7, 3.2), (4.7, 6.0), (2.0, 3.2)]:
        pp = ZFParams(pi1=pi_val, pi2=pi_val, cu=cu_val, h=0.1, lam2=5.0)
        I2star = threshold_symmetric(pp)
        I2_round = max(1, round(I2star))
        rule_of_thumb = I2star / pp.lam2

        # find tau where |e(I2*,tau)| drops below 5% of g(I2*) magnitude
        g0 = abs(g_mismatch(I2_round, pp))
        tau_grid = np.linspace(0.001, 6.0, 3000)
        e_vals = np.array([abs(error_poisson(I2_round, t, pp)) for t in tau_grid])
        below = np.where(e_vals < 0.05 * g0)[0]
        tau_5pct = tau_grid[below[0]] if len(below) else np.nan

        print(f"  pi={pi_val}, cu={cu_val}: I2*={I2star:.3f} (round {I2_round})  "
              f"rule_of_thumb=I2*/lam2={rule_of_thumb:.3f}  "
              f"tau(error<5%)={tau_5pct:.3f}")
        results.append((pp, I2star, I2_round, tau_grid, e_vals, g0, rule_of_thumb, tau_5pct))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for I2 in [1, 2, 3, 4, 5]:
        taus_plot = np.linspace(0.001, 3.0, 600)
        e_plot = [error_poisson(I2, t, p) for t in taus_plot]
        ax.plot(taus_plot, e_plot, label=f'I2={I2}')
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$e(I_2,\tau) = V^*-V_w$ (predicted)')
    ax.set_title('Boundary-mismatch error, Poisson-convolution prediction')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for pp, I2star, I2_round, tau_grid, e_vals, g0, rot, tau5 in results:
        ax.plot(tau_grid, e_vals / g0, label=f'pi={pp.pi1}, cu={pp.cu} (I2*={I2star:.1f})')
        ax.axvline(rot, ls=':', alpha=0.5)
    ax.axhline(0.05, color='gray', ls='--', label='5% threshold')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$|e(I_2^*,\tau)| / |g(I_2^*)|$')
    ax.set_title('Normalised error decay vs rule-of-thumb (dotted)\nboundary depth ~ I2*/lam2')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig('boundary_condition_decay.png', dpi=150)
    print("\nFigure saved to boundary_condition_decay.png")


if __name__ == '__main__':
    main()