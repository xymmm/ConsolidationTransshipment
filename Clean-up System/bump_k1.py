"""
bump_k1.py — dispatch threshold when only one dispatch is allowed (K = 1)
==========================================================================

Uses KDispatchDP from k_dispatch_solver.py with K = 1, so the policy is the
optimal policy with at most one dispatch over the horizon: the model decides
when to dispatch and how much, and after that dispatch no other is allowed.

For each instance in INSTANCES it reports
  1. the threshold b1bar(I2, tau) at the chosen tau values, where b1bar is
     the smallest backlog b1 at which the policy dispatches (inf = never);
  2. how the threshold moves with I2 at every tau: how often it goes up,
     stays the same, or goes down when I2 increases by one;
  3. every place where it goes down, with its tau, I2 and the two
     threshold values, and the shape of the threshold in each period:
     rises only, falls only, falls then rises, or rises then falls; the
     last one is the shape of the bump of the unrestricted DP;
  4. the same count in the tau direction;
and saves two figures per instance in OUT_DIR.

Run from PyCharm without arguments.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solver import Params
from k_dispatch_solver import KDispatchDP

BASE = dict(T=5.0, N=400, lam1=5, lam2=3, h=1, cu=1, c1=0, c2=0, v2=0,
            I2_max=40, I2_min=-31, b1_max=140)
INSTANCES = {
    "sec62":   dict(Cf=8,  pi1=6,  pi2=6),
    "meeting": dict(Cf=12, pi1=6,  pi2=8),
    "pi1_gt":  dict(Cf=20, pi1=10, pi2=6),
    # an instance in which the K = 1 threshold first falls and then rises
    "falls_then_rises": dict(Cf=33.4, pi1=5.1, pi2=4.5, lam1=2.48, lam2=5.16,
                             h=0.22, cu=0.49, T=3.0, I2_min=-21, b1_max=25),
}
TAUS = (1.0, 3.0, 5.0)     # tau values for the printed table and the figure
I2_SHOW = 25               # largest I2 printed and plotted
OUT_DIR = "bump_k1_out"


def steps(a, b):
    """Compare thresholds a (at I2) and b (at I2 + 1), both finite."""
    up = int((b > a).sum()); same = int((b == a).sum()); down = int((b < a).sum())
    return up, same, down


def analyse(name, extra):
    p = Params(**{**BASE, **extra})
    kd = KDispatchDP(p, K=1).solve()
    B = kd.threshold_table(1)                    # rows n = 1..N, cols I2 = 1..I2_max
    taus = np.arange(1, p.N + 1) * p.dt

    print(f"\n=== {name}: {extra} ===")

    # 1. threshold at the chosen tau values
    print("1. threshold b1bar at I2 = 1, 2, ...  (inf = never dispatch)")
    for tau in [t for t in TAUS if t <= p.T]:
        row = B[int(round(tau / p.dt)) - 1][:I2_SHOW]
        print(f"   tau = {tau:4.2f}: " + " ".join("inf" if np.isnan(v) else f"{int(v)}" for v in row))

    # 2. direction in I2 (finite neighbours only)
    a, b = B[:, :-1], B[:, 1:]
    fin = np.isfinite(a) & np.isfinite(b)
    up, same, down = steps(a[fin], b[fin])
    tot = up + same + down
    print(f"2. when I2 increases by one: threshold goes up {up} times "
          f"({100 * up / tot:.1f}%), stays {same} ({100 * same / tot:.1f}%), "
          f"goes down {down} ({100 * down / tot:.2f}%)")
    back_to_inf = int((np.isfinite(a) & np.isnan(b)).sum())
    print(f"   finite threshold followed by inf at the next I2: {back_to_inf}")

    # 3. every place where it goes down
    r, c = np.nonzero(fin & (b < a))
    if r.size == 0:
        print("3. the threshold never goes down in I2: no bump at K = 1")
    else:
        print(f"3. places where the threshold goes down in I2 ({r.size}):")
        for i, j in list(zip(r, c))[:30]:
            print(f"   tau = {taus[i]:.4f}: I2 = {j + 1} -> {j + 2}: "
                  f"{int(B[i, j])} -> {int(B[i, j + 1])}")
        if r.size > 30:
            print(f"   ... {r.size - 30} more")

    # 3b. shape of the threshold in I2 in each period
    shape = dict(rises_only=0, falls_only=0, falls_then_rises=0,
                 rises_then_falls=0, flat=0)
    for row in B[:, :I2_SHOW]:
        f = row[np.isfinite(row)]
        d = np.sign(np.diff(f)); d = d[d != 0]
        if d.size == 0:
            shape["flat"] += 1; continue
        rf = any(d[i] > 0 and (d[i + 1:] < 0).any() for i in range(len(d)))
        fr = any(d[i] < 0 and (d[i + 1:] > 0).any() for i in range(len(d)))
        if rf:
            shape["rises_then_falls"] += 1
        elif fr:
            shape["falls_then_rises"] += 1
        elif (d > 0).all():
            shape["rises_only"] += 1
        else:
            shape["falls_only"] += 1
    print(f"3b. shape in I2, number of periods: {shape}")
    print("    a bump in the sense of the unrestricted DP is 'rises_then_falls'")

    # 4. direction in tau
    a, b = B[:-1, :], B[1:, :]
    fin = np.isfinite(a) & np.isfinite(b)
    up, same, down = steps(a[fin], b[fin])
    tot = up + same + down
    print(f"4. when tau increases by one period: threshold goes up {up} "
          f"({100 * up / tot:.2f}%), stays {same}, goes down {down} "
          f"({100 * down / tot:.2f}%)")

    # figures
    os.makedirs(OUT_DIR, exist_ok=True)
    I2 = np.arange(1, I2_SHOW + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for tau in [t for t in TAUS if t <= p.T]:
        ax.step(I2, B[int(round(tau / p.dt)) - 1][:I2_SHOW], where="mid",
                lw=1.8, label=f"τ = {tau}")
    ax.set_xlabel("I₂"); ax.set_ylabel("b̄₁ (missing = never)")
    ax.set_title(f"{name}: threshold with one dispatch allowed", fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"{name}_k1_slices.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#C8C8C8")
    im = ax.imshow(np.ma.masked_invalid(B[:, :I2_SHOW]), aspect="auto", origin="lower",
                   cmap=cmap, interpolation="nearest",
                   extent=[0.5, I2_SHOW + 0.5, taus[0], taus[-1]])
    ax.set_xlabel("I₂"); ax.set_ylabel("τ")
    ax.set_title(f"{name}: b̄₁ with one dispatch allowed, grey = never", fontsize=10)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"{name}_k1_heatmap.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    for name, extra in INSTANCES.items():
        analyse(name, extra)
    print(f"\nFigures saved in {OUT_DIR}/")