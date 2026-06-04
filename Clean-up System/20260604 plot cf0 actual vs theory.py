"""
20260604_plot_Cf0_actual_vs_theory.py
======================================
For each of the five Cf=0 cases (Theorems 1-5 of zero_fixed_cost.pdf), overlay
the DP-extracted optimal dispatch threshold I2*(tau) against the paper's
analytical threshold, and show the gap (DP - analytical) explicitly.

This script only CALLS the existing 3-state backward-induction solver
(solver.py); it does not re-implement the DP.

Why b1 = 1
----------
The threshold is the smallest I2 at which dispatching the marginal unit becomes
optimal, so it is read off at b1 = 1.  That slice sits dozens of units away from
the b1_max / I2_min grid boundaries, so the truncation artefact discussed
separately does NOT touch these curves.

Analytical threshold (paper, Cf=0):
    I2*(tau) = lam2*cu/(h+pi2) - lam2*(pi1-pi2)/(h+pi2) * tau
    pi1 = pi2 -> constant ;  pi1 > pi2 -> decreasing ;  pi1 < pi2 -> increasing

To match the paper's terminal condition V(I2,0)=0, all clean-up costs are 0
(c1 = c2 = v2 = 0) and Cf = 0.

Output: five PNGs in ./20260604/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from solver import Params, TransshipmentDP

# ----------------------------------------------------------------------
# shared settings
# ----------------------------------------------------------------------
H              = 1.0
LAM1, LAM2     = 5.0, 3.0
T, N           = 2.0, 400
I2_MAX, I2_MIN = 35, -15        # I2_min kept well below the threshold band
B1_MAX         = 30             # b1=1 query is ~29 units from this boundary
TAU            = np.linspace(T / N, T, 200)   # remaining-time grid
OUTDIR         = "20260604"

# colours
C_DP   = "#C0432A"   # actual (DP)
C_ANA  = "#1F4E79"   # theory (paper)
C_GAP  = "#6A2C91"   # gap
C_BAND = "#CFE8CF"   # rounding band
C_CENS = "#F2D5CE"   # censored band

# ----------------------------------------------------------------------
# five cases
# ----------------------------------------------------------------------
CASES = [
    dict(key="case1_Th1", thm="Theorem 1", pi1=6.0,  pi2=6.0,  cu=1.0,
         cond=r"$\pi_1=\pi_2$, $\alpha\leq1$",
         note="degenerate: always dispatch"),
    dict(key="case2_Th2", thm="Theorem 2", pi1=6.0,  pi2=6.0,  cu=6.0,
         cond=r"$\pi_1=\pi_2$, $\alpha>1$",
         note="constant threshold"),
    dict(key="case3_Th3", thm="Theorem 3", pi1=10.0, pi2=6.0,  cu=1.0,
         cond=r"$\pi_1>\pi_2$, $\alpha\leq1$",
         note="degenerate: always dispatch"),
    dict(key="case4_Th4", thm="Theorem 4", pi1=10.0, pi2=6.0,  cu=10.0,
         cond=r"$\pi_1>\pi_2$, $\alpha>1$",
         note="decreasing threshold"),
    dict(key="case5_Th5", thm="Theorem 5", pi1=2.0,  pi2=6.0,  cu=3.0,
         cond=r"$\pi_1<\pi_2$",
         note="increasing threshold"),
]


def analytical(tau, c):
    """Paper's threshold curve (real-valued)."""
    return (LAM2 * c["cu"] / (H + c["pi2"])
            - LAM2 * (c["pi1"] - c["pi2"]) / (H + c["pi2"]) * tau)


def build_and_solve(c):
    p = Params(T=T, N=N, lam1=LAM1, lam2=LAM2, h=H, Cf=0.0, cu=c["cu"],
               pi1=c["pi1"], pi2=c["pi2"], c1=0.0, c2=0.0, v2=0.0,
               I2_max=I2_MAX, I2_min=I2_MIN, b1_max=B1_MAX)
    dp = TransshipmentDP(p)
    dp.solve(store_V=False, verbose=False)
    return dp


def dp_threshold(dp, tau):
    """Smallest I2 (at b1=1) where dispatch is optimal; None if none in grid."""
    dt = dp.p.T / dp.p.N
    n = min(dp.p.N, max(1, round(tau / dt)))
    for I2 in range(1, dp.p.I2_max + 1):
        if dp.get_policy(n, I2, 1) > 0:
            return I2
    return None


def make_figure(c):
    dp = build_and_solve(c)

    ana   = analytical(TAU, c)
    raw   = [dp_threshold(dp, t) for t in TAU]
    dp_th = np.array([np.nan if v is None else float(v) for v in raw])
    cens  = np.isnan(dp_th)

    # Dispatch requires I2>=1, so a formula value below 1 just means
    # "always dispatch".  Floor the analytical curve at 1 before forming the
    # gap, otherwise the degenerate cases report a spurious gap against a
    # sub-1 (even negative) prediction that has no operational meaning.
    ana_cmp = np.maximum(1.0, ana)
    gap     = dp_th - ana_cmp                 # nan where DP did not dispatch

    a0    = analytical(0.0, c)               # analytical value at tau=0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.3, 1.0]))

    # ---- "wait-everywhere" band: DP dispatches for no I2 in this tau range ----
    # This is still inside the horizon; it marks where the optimal policy is to
    # wait regardless of I2 (no finite dispatch threshold exists).
    if cens.any():
        tcens = TAU[cens]
        ax1.axvspan(tcens.min(), tcens.max(), color=C_CENS, alpha=0.7, zorder=0)
        ax2.axvspan(tcens.min(), tcens.max(), color=C_CENS, alpha=0.7, zorder=0)
        xc = 0.5 * (tcens.min() + tcens.max())
        ax1.text(xc, 0.94,
                 "DP optimal: wait for every $I_2$\n"
                 "(no finite threshold; still inside horizon)",
                 transform=ax1.get_xaxis_transform(),
                 ha="center", va="top", fontsize=8.5, style="italic",
                 color="#8A3324")

    # ---- top panel: overlay ----
    ax1.plot(TAU, ana, color=C_ANA, lw=2.2, ls="--",
             label="paper analytical (theory)")
    ax1.plot(TAU, dp_th, color=C_DP, lw=2.4,
             label="DP optimal (actual)")
    ax1.axhline(1, color="grey", lw=0.9, ls=":")
    ax1.text(T * 0.985, 1.06, r"dispatch floor ($I_2\geq1$)",
             fontsize=8.5, color="grey", va="bottom")
    ax1.set_ylabel(r"dispatch threshold  $I_2^*(\tau)$", fontsize=12)
    top = max(np.nanmax(dp_th) if not np.all(cens) else 1.0, a0, 1.0) + 1.2
    ax1.set_ylim(-0.6, top)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=10)

    # annotate the analytical intercept at tau=0
    ax1.annotate(f"theory @ $\\tau$=0:  {a0:.2f}",
                 xy=(0, a0), xytext=(T * 0.18, a0 + 0.6),
                 fontsize=9, color=C_ANA,
                 arrowprops=dict(arrowstyle="->", color=C_ANA, lw=0.8))

    # ---- bottom panel: the gap ----
    ax2.axhspan(-0.5, 0.5, color=C_BAND, alpha=0.6,
                label="$\\pm0.5$ (integer rounding)")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.plot(TAU, gap, color=C_GAP, lw=2.0)
    ax2.set_ylabel("gap\nDP $-$ analytical$^{*}$", fontsize=11)
    ax2.grid(True, alpha=0.3)
    finite = gap[np.isfinite(gap)]
    if finite.size:
        ax2.set_ylim(min(-0.8, finite.min() - 0.4), max(1.5, finite.max() + 0.4))

    # interior diagnostic (deep interior = large tau half)
    interior = (TAU >= 0.5 * T) & np.isfinite(gap)
    med = np.nanmedian(gap[interior]) if interior.any() else np.nan
    cens_frac = 100.0 * cens.mean()

    # ---- x axis: tau from T (start) on the left to 0 (end) on the right ----
    ax2.set_xlim(T, 0)
    ax2.set_xlabel(r"$\tau$ (remaining time)   $\longleftarrow$   end of horizon",
                   fontsize=11)

    axtop = ax1.twiny()
    axtop.set_xlim(ax1.get_xlim())
    axtop.set_xticks([T, T * 0.5, TAU[0]])
    axtop.set_xticklabels(["start", "mid", "end"], fontsize=9)
    axtop.set_xlabel("planning horizon", fontsize=9)

    title = (f"{c['thm']}  ({c['note']})   |   {c['cond']}\n"
             f"$C_f=0$,  $h={H:g}$,  $\\pi_1={c['pi1']:g}$,  "
             f"$\\pi_2={c['pi2']:g}$,  $c_u={c['cu']:g}$,  "
             f"$\\lambda_1={LAM1:g}$,  $\\lambda_2={LAM2:g}$")
    fig.suptitle(title, fontsize=11.5, y=0.99)

    # confine the axes to the upper part of the figure, leaving a clear band
    # at the bottom for the diagnostic footer (no overlap with any plot element)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])

    footer = (f"interior (large $\\tau$) median gap = {med:+.2f}      |      "
              f"shaded band = {cens_frac:.0f}% of horizon where DP waits for "
              f"every $I_2$ (still inside horizon)\n"
              f"green band = $\\pm0.5$ integer-rounding tolerance      |      "
              f"$^{{*}}$analytical floored at the $I_2\\geq1$ dispatch level")
    fig.text(0.5, 0.035, footer, ha="center", va="center", fontsize=8.5,
             color="0.25")

    out = os.path.join(OUTDIR, f"{c['key']}_actual_vs_theory.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, med, cens_frac


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("=" * 70)
    print("Cf=0  actual (DP) vs theory (analytical) threshold comparison")
    print("=" * 70)
    for c in CASES:
        print(f"  solving {c['key']} ({c['thm']}) ...", end=" ", flush=True)
        out, med, cf = make_figure(c)
        print(f"done.  interior gap median={med:+.2f}, "
              f"censored={cf:.0f}%  ->  {out}")
    print("=" * 70)
    print(f"All five PNGs written to ./{OUTDIR}/")


if __name__ == "__main__":
    main()