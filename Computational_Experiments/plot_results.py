"""
Plot_results.py — Cost curves from the saved sweep results.

This is a NEW file (no counterpart in the existing project tree).

Produces, inside Computational_Experiments/results/:
  cost_vs_Q.pdf          C(Q) with 95% CI band + component breakdown
  cost_vs_Delta.pdf      C(Delta) with 95% CI band + component breakdown

Run after Policy_main.py (or after the two standalone runners):
    python Plot_results.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Config

COMPONENTS = ["penalty1", "holding", "fixed", "variable", "penalty2", "terminal"]
COMP_LABELS = {
    "penalty1": r"backlog penalty $\pi_1$",
    "holding":  r"holding $h$",
    "fixed":    r"fixed dispatch $C_f$",
    "variable": r"variable dispatch $c_u$",
    "penalty2": r"retailer-2 penalty $\pi_2$",
    "terminal": r"terminal clean-up",
}


def _load(path):
    with open(path) as f:
        return json.load(f)


def _extract(records, xkey):
    x    = np.array([r[xkey] for r in records], dtype=float)
    mean = np.array([r["total"]["mean"] for r in records])
    ci   = np.array([r["total"]["ci95"] for r in records])
    comps = {c: np.array([r[c]["mean"] for r in records]) for c in COMPONENTS}
    order = np.argsort(x)
    return (x[order], mean[order], ci[order],
            dict((c, v[order]) for c, v in comps.items()))


def _plot_curve(x, mean, ci, comps, xlabel, xsymbol, title, outfile):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ── top: total cost with CI band, minimiser marked ──
    ax1.plot(x, mean, marker="o", ms=3.5, lw=1.2, color="tab:blue",
             label="simulated mean cost")
    ax1.fill_between(x, mean - ci, mean + ci, alpha=0.25, color="tab:blue",
                     label="95% CI")
    i_star = int(np.argmin(mean))
    ax1.axvline(x[i_star], ls="--", lw=0.8, color="grey")
    ax1.annotate(
        "min at %s = %g\ncost = %.2f" % (xsymbol, x[i_star], mean[i_star]),
        xy=(x[i_star], mean[i_star]), xytext=(8, 12),
        textcoords="offset points", fontsize=8,
    )
    ax1.set_ylabel("expected total cost")
    ax1.set_title(title)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── bottom: mean cost components ──
    for c in COMPONENTS:
        ax2.plot(x, comps[c], lw=1.0, label=COMP_LABELS[c])
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("component mean")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> %s" % outfile)


def plot_q():
    records = _load(Config.Q_RESULTS_JSON)
    x, mean, ci, comps = _extract(records, "Q")
    _plot_curve(
        x, mean, ci, comps,
        xlabel=r"$Q$  (dispatch lot size)",
        xsymbol="Q",
        title=r"Q-policy: cost vs $Q$",
        outfile=os.path.join(Config.RESULTS_DIR, "cost_vs_Q.pdf"),
    )


def plot_t():
    records = _load(Config.T_RESULTS_JSON)
    x, mean, ci, comps = _extract(records, "delta")
    _plot_curve(
        x, mean, ci, comps,
        xlabel=r"$\Delta$  (review interval)",
        xsymbol="Delta",
        title=r"T-policy: cost vs $\Delta$",
        outfile=os.path.join(Config.RESULTS_DIR, "cost_vs_Delta.pdf"),
    )


if __name__ == "__main__":
    plot_q()
    plot_t()