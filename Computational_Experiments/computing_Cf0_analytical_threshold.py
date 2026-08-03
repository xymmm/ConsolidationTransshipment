"""
computing_Cf0_analytical_threshold.py
=====================================
Analytical optimal policy of the Cf = 0 note, Eq. (20)-(22).

    Ibar(tau) = min{ m >= 1 : M(m, tau) >= g(tau) }          Eq. (20), (22)
    M(n, tau) = E[min(K, n)] = sum_{j=1}^{n} P{K >= j}       Eq. (21)
    g(tau)    = lam2 ( cu + (pi2 - pi1) tau ) / (h + pi2)
    K ~ Poisson(lam2 tau)

Optimal policy: at a Retailer-1 arrival with remaining time tau, dispatch
one unit iff I2 >= Ibar(tau); otherwise reject. Ibar = +inf means never
dispatch; it holds exactly for tau <= tau* = cu / (h + pi1).

Ibar does not depend on lam1: in V(I2,tau) - V(I2-1,tau) the term
(pi1 lam1 + pi2 lam2) tau^2 / 2 is common to all I2 and cancels.

Usage
-----
    python computing_Cf0_analytical_threshold.py
    python computing_Cf0_analytical_threshold.py --lam2 5 --h 1 --cu 5 \
           --pi1 4 --pi2 5.5 --T 5 --ntau 501

Output: Cf0_analytical_threshold/optimal_policy.csv
        columns  tau, Ibar
Pure Python + numpy.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "Cf0_analytical_threshold")


def threshold(tau: float, lam2: float, h: float, cu: float,
              pi1: float, pi2: float, mmax: int = 5000) -> float:
    """Ibar(tau) from Eq. (20)-(22); float('inf') when no finite level works."""
    if tau <= 0:
        return math.inf
    level = lam2 * (cu + (pi2 - pi1) * tau) / (h + pi2)      # g(tau)
    if level <= 0:
        return 1.0
    mu = lam2 * tau
    if level > mu:                    # sup_m M(m,tau) = E[K] = mu
        return math.inf
    total = 0.0
    pmf = math.exp(-mu)               # P{K = j-1}, from j = 1
    cdf = pmf                         # P{K <= j-1}
    for m in range(1, mmax + 1):
        total += 1.0 - cdf            # add P{K >= m}
        if total >= level:
            return float(m)
        pmf *= mu / m
        cdf += pmf
    return math.inf


def main():
    ap = argparse.ArgumentParser(
        description="Analytical optimal threshold Ibar(tau), Cf = 0.")
    ap.add_argument("--lam2", type=float, default=5.0)
    ap.add_argument("--h",    type=float, default=1.0)
    ap.add_argument("--cu",   type=float, default=5.0)
    ap.add_argument("--pi1",  type=float, default=4.0)
    ap.add_argument("--pi2",  type=float, default=5.5)
    ap.add_argument("--T",    type=float, default=5.0)
    ap.add_argument("--ntau", type=int,   default=501,
                    help="number of tau points on (0, T]")
    a = ap.parse_args()

    assert a.lam2 > 0 and a.T > 0 and a.h + a.pi2 > 0

    taus = np.linspace(a.T / a.ntau, a.T, a.ntau)
    ibar = [threshold(float(t), a.lam2, a.h, a.cu, a.pi1, a.pi2)
            for t in taus]

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "optimal_policy.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "Ibar"])
        for t, v in zip(taus, ibar):
            w.writerow([f"{t:.10g}", "inf" if math.isinf(v) else int(v)])

    tau_star = a.cu / (a.h + a.pi1)
    finite = [v for v in ibar if math.isfinite(v)]
    print(f"lam2={a.lam2} h={a.h} cu={a.cu} pi1={a.pi1} pi2={a.pi2} T={a.T}")
    print(f"tau* = cu/(h+pi1) = {tau_star:.6g}   "
          f"(Ibar = +inf for tau <= tau*)")
    if finite:
        print(f"Ibar range on finite part: {int(min(finite))} .. "
              f"{int(max(finite))};   Ibar(T) = {int(ibar[-1])}")
    print(f"wrote {len(taus)} rows -> {path}")


if __name__ == "__main__":
    main()