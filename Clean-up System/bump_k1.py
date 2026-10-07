"""
bump_k1.py — bump in the (I2, b1) plane when only one dispatch is allowed
==========================================================================

Policy: at most one dispatch over the horizon (KDispatchDP with K = 1).

A bump cell is a cell of the policy table at a fixed tau, rows I2 and
columns b1, that waits while in the same column b1 some smaller I2 and some
larger I2 dispatch. In Section 6.2 of the unrestricted DP these are the
cells (6, 3) and (7, 3).

For every instance and every period the script looks for bump cells. When a
period has bump cells it prints the part of the policy table around them:
    .     wait
    q     dispatch q units
    [.]   bump cell
and, for each bump cell, the cost margin Q(wait) − best dispatch; a margin
of size below 1e-9 is an exact tie, decided only by rounding. When an
instance has no bump cell nothing is printed for it. No files are written.

Run from PyCharm without arguments.
"""

import numpy as np
from solver import Params
from k_dispatch_solver import KDispatchDP

BASE = dict(T=5.0, N=400, lam1=5, lam2=3, h=1, cu=1, c1=0, c2=0, v2=0,
            I2_max=40, I2_min=-31, b1_max=140)
INSTANCES = {
    "sec62":   dict(Cf=8,  pi1=6,  pi2=6),
    "meeting": dict(Cf=12, pi1=6,  pi2=8),
    "pi1_gt":  dict(Cf=20, pi1=10, pi2=6),
    "falls_then_rises": dict(Cf=33.4, pi1=5.1, pi2=4.5, lam1=2.48, lam2=5.16,
                             h=0.22, cu=0.49, T=3.0, I2_min=-21, b1_max=25),
}
MAX_PERIODS = 10     # largest number of periods printed per instance
TIE = 1e-9           # |margin| below this is an exact tie


def bump_cells(D):
    """D: dispatch table, rows I2 = 1, 2, ..., columns b1 = 1, 2, ..."""
    below = np.maximum.accumulate(D, axis=0)
    above = np.maximum.accumulate(D[::-1], axis=0)[::-1]
    return (~D) & below & above


def print_table(P, G, n, tau, margin):
    """P: dispatch quantities, G: bump cells; rows I2 = 1.., cols b1 = 1.."""
    rows, cols = np.nonzero(G)
    r_lo, r_hi = max(rows.min() - 2, 0), min(rows.max() + 2, P.shape[0] - 1)
    c_lo, c_hi = max(cols.min() - 3, 0), min(cols.max() + 3, P.shape[1] - 1)
    print(f"\n  tau = {tau:.4f} (n = {n}), {len(rows)} bump cell(s)")
    print("  I2 \\ b1 " + "".join(f"{j + 1:>5}" for j in range(c_lo, c_hi + 1)))
    for i in range(r_lo, r_hi + 1):
        cells = []
        for j in range(c_lo, c_hi + 1):
            if G[i, j]:
                cells.append("[.]")
            elif P[i, j] > 0:
                cells.append(str(int(P[i, j])))
            else:
                cells.append(".")
        print(f"  {i + 1:>7} " + "".join(f"{c:>5}" for c in cells))
    for i, j in zip(rows, cols):
        m = margin[i, j]
        print(f"  bump cell (I2, b1) = ({i + 1}, {j + 1}): Q(wait) - best dispatch "
              f"= {m:+.3e}" + ("  -> exact tie" if abs(m) < TIE else ""))


def analyse(name, extra):
    p = Params(**{**BASE, **extra})
    kd = KDispatchDP(p, K=1).solve()
    r0 = 1 - p.I2_min                                    # row of I2 = 1
    found = []
    for n in range(1, p.N + 1):
        P = kd.pol[1][n][r0:, 1:]
        G = bump_cells(P > 0)
        if G.any():
            found.append((n, P, G))
    if not found:
        return False
    print(f"\n=== {name}: {extra} ===")
    print(f"periods with bump cells: {len(found)} of {p.N}")
    for n, P, G in found[:MAX_PERIODS]:
        w = kd.Q(0, kd.V[1][n - 1])
        d, _ = kd.best_dispatch(kd.V[0][n - 1])
        print_table(P, G, n, n * p.dt, (w - d)[r0:, 1:])
    if len(found) > MAX_PERIODS:
        print(f"\n  ... {len(found) - MAX_PERIODS} more periods not printed")
    return True


if __name__ == "__main__":
    for name, extra in INSTANCES.items():
        analyse(name, extra)