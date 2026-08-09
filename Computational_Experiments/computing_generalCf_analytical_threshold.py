"""
computing_generalCf_analytical_threshold.py
===========================================
Analytical dispatch policy of the general-Cf note, Sections 4-5, Eq. (18),
(23)-(29). This is the GENERAL procedure; Section 6 (Eq. 30-36) is the
special case pi1 = pi2, obtained here by setting pi1 = pi2.

Marginal value of the m-th retained unit, Eq. (18):

    Dbeta(m, tau) = -pi2 tau + (h + pi2)/lam2 * E[min(K, m)],
    K ~ Poisson(lam2 tau)

Marginal saving of the i-th peeled unit, Eq. (23), peeling from level I2:

    delta_i = Dbeta(I2 - i, tau) - (cu - pi1 tau)
            = (h + pi2)/lam2 * E[min(K, I2 - i)] - cu + (pi1 - pi2) tau

The sequence delta_0 >= delta_1 >= ... is non-increasing, so the cumulative
saving is maximised by keeping exactly the positive terms, Eq. (25):

    N(I2, tau) = #{ m in 1..I2 : delta(m, tau) > 0 }
    q0         = min(b1, N(I2, tau))
    S          = sum_{i=0}^{q0-1} delta_i

Eq. (26):   q* = q0 if S > Cf, else 0
Eq. (27):   Vd = Vw - (S - Cf)^+
Eq. (28)-(29):
    b1bar(I2, tau) = min{ b >= 1 : sum_{i=0}^{b-1} delta_i > Cf },  +inf if
    the saturated saving Smax never exceeds Cf.

Remark (note, after Eq. 29):
    delta_i <= (h + pi1) tau - cu, so b1bar = +inf whenever
    tau <= tau* = cu / (h + pi1); and b1bar does not depend on lam1.

Sign test as a level condition, Eq. (24):
    delta_i > 0  <=>  E[min(K, I2 - i)] > g*(tau) = lam2 (cu + (pi2-pi1) tau)
                                                    / (h + pi2)

Usage
-----
    from computing_generalCf_analytical_threshold import GeneralCfPolicy
    pol = GeneralCfPolicy(lam2=3, h=1, cu=1, pi1=6, pi2=6, Cf=8)
    pol.b1bar(I2=10, tau=5.0)      # threshold
    pol.q_star(I2=10, b1=3, tau=5.0)
    pol.saving(I2=10, b1=3, tau=5.0)

    python computing_generalCf_analytical_threshold.py --Cf 8 --pi1 6 --pi2 6
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "generalCf_analytical_threshold")


class GeneralCfPolicy:
    """Sections 4-5 of the general-Cf note, evaluated exactly."""

    def __init__(self, lam2: float, h: float, cu: float,
                 pi1: float, pi2: float, Cf: float):
        self.lam2, self.h, self.cu = lam2, h, cu
        self.pi1, self.pi2, self.Cf = pi1, pi2, Cf

    # ── E[min(K, m)] for m = 0..mmax, K ~ Poisson(lam2 tau) ──────
    def _Emin(self, tau: float, mmax: int) -> np.ndarray:
        E = np.zeros(mmax + 1)
        if tau <= 0 or mmax < 1:
            return E
        mu = self.lam2 * tau
        pmf = math.exp(-mu)          # P{K = j-1}
        cdf = pmf                    # P{K <= j-1}
        tot = 0.0
        for j in range(1, mmax + 1):
            tot += 1.0 - cdf         # add P{K >= j}
            E[j] = tot
            pmf *= mu / j
            cdf += pmf
        return E

    # ── Eq. (18) and (23) ────────────────────────────────────────
    def delta_levels(self, I2: int, tau: float) -> np.ndarray:
        """delta(m, tau) for m = 0..I2; index m is the level being peeled."""
        E = self._Emin(tau, I2)
        return ((self.h + self.pi2) / self.lam2 * E
                - self.cu + (self.pi1 - self.pi2) * tau)

    def tau_star(self) -> float:
        """b1bar = +inf for tau <= tau*, from the Remark after Eq. (29)."""
        return self.cu / (self.h + self.pi1)

    def g_star(self, tau: float) -> float:
        """Level cutoff of the sign test, Eq. (24)."""
        return (self.lam2 * (self.cu + (self.pi2 - self.pi1) * tau)
                / (self.h + self.pi2))

    # ── Eq. (25): N, q0, S ───────────────────────────────────────
    def n_positive(self, I2: int, tau: float) -> int:
        if I2 < 1 or tau <= 0:
            return 0
        d = self.delta_levels(I2, tau)
        return int((d[1:I2 + 1] > 0).sum())

    def saving(self, I2: int, b1: int, tau: float) -> tuple[int, float]:
        """(q0, S) of Eq. (25): the optimal batch before the Cf test."""
        if I2 < 1 or b1 < 1 or tau <= 0:
            return 0, 0.0
        d = self.delta_levels(I2, tau)
        npos = int((d[1:I2 + 1] > 0).sum())
        if npos == 0:
            return 0, 0.0
        q0 = min(b1, npos)
        S = float(sum(d[I2 - i] for i in range(q0)))
        return q0, S

    # ── Eq. (26): q* ─────────────────────────────────────────────
    def q_star(self, I2: int, b1: int, tau: float) -> int:
        q0, S = self.saving(I2, b1, tau)
        return q0 if (q0 > 0 and S > self.Cf) else 0

    # ── Eq. (27): the drop in value from dispatching ─────────────
    def value_drop(self, I2: int, b1: int, tau: float) -> float:
        """(S - Cf)^+ , so Vd = Vw - value_drop."""
        _, S = self.saving(I2, b1, tau)
        return max(S - self.Cf, 0.0)

    # ── Eq. (28)-(29): the threshold ─────────────────────────────
    def b1bar(self, I2: int, tau: float) -> float:
        """Smallest b1 >= 1 at which dispatching beats waiting; inf if none."""
        if I2 < 1 or tau <= 0:
            return math.inf
        if tau <= self.tau_star():          # Remark after Eq. (29)
            return math.inf
        d = self.delta_levels(I2, tau)
        run = 0.0
        for b in range(1, I2 + 1):
            di = d[I2 - b + 1]
            if di <= 0:                      # saturated: Smax <= Cf
                break
            run += di
            if run > self.Cf:
                return float(b)
        return math.inf

    def b1bar_curve(self, I2_values, tau: float) -> np.ndarray:
        return np.array([self.b1bar(int(i), tau) for i in I2_values])

    def summary(self) -> str:
        return (f"lam2={self.lam2} h={self.h} cu={self.cu} "
                f"pi1={self.pi1} pi2={self.pi2} Cf={self.Cf} "
                f"tau*={self.tau_star():.6g}")


def main():
    ap = argparse.ArgumentParser(
        description="Analytical b1bar and q* of the general-Cf note, Sec 4-5.")
    ap.add_argument("--lam2", type=float, default=3.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--cu", type=float, default=1.0)
    ap.add_argument("--pi1", type=float, default=6.0)
    ap.add_argument("--pi2", type=float, default=6.0)
    ap.add_argument("--Cf", type=float, default=8.0)
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--I2max", type=int, default=30)
    ap.add_argument("--ntau", type=int, default=51)
    a = ap.parse_args()

    pol = GeneralCfPolicy(a.lam2, a.h, a.cu, a.pi1, a.pi2, a.Cf)
    print(pol.summary())

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"b1bar_Cf{a.Cf:g}.csv")
    taus = np.linspace(a.T / a.ntau, a.T, a.ntau)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "I2", "b1bar", "N_positive", "S_max"])
        for t in taus:
            for I2 in range(1, a.I2max + 1):
                bb = pol.b1bar(I2, float(t))
                npos = pol.n_positive(I2, float(t))
                _, smax = pol.saving(I2, 10 ** 6, float(t))
                w.writerow([f"{t:.6g}", I2,
                            "inf" if math.isinf(bb) else int(bb),
                            npos, f"{smax:.6g}"])
    print(f"wrote {path}")

    print(f"\nb1bar(I2, tau=T={a.T}) for I2 = 1..12:")
    row = "  " + "  ".join(
        f"{('inf' if math.isinf(pol.b1bar(i, a.T)) else int(pol.b1bar(i, a.T))):>3}"
        for i in range(1, 13))
    print("  I2 :" + "".join(f"{i:>5}" for i in range(1, 13)))
    print("  b1bar:" + row)


if __name__ == "__main__":
    main()