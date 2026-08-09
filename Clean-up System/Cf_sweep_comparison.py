"""
Cf_sweep_comparison.py — DP versus the analytical policy as Cf grows
=====================================================================
Place in "Computational_Experiments/". It imports

    Clean-up System/solver.py                          3-D DP, V(I2, b1, tau)
    Computational_Experiments/
        computing_generalCf_analytical_threshold.py    Sec 4-5 analytic policy

and runs a continuous-time discrete-event simulation of both policies on one
shared set of sample paths.

Scope. The sweep starts at a SMALL POSITIVE Cf and uses the 3-D backlog model
throughout, so every row is the same model and the rows are comparable. The
Cf = 0 switching model is a different model (no b1 state, per-unit accept or
reject) and is validated separately with solver_cf0_2d.py; it is deliberately
not mixed into this table.

For each Cf the script records
  thresholds     b1bar from the DP and from Eq. (28)-(29), at several I2
  decisions      agreement over a full (I2, b1, tau) grid, and the direction
                 of every disagreement
  quantities     whether q* agrees at the states where both dispatch
  cost, exact    policy evaluation of both policies on one common recursion
  cost, DES      continuous-time simulation on common random numbers, with
                 dispatch counts and realised batch sizes

Usage
-----
    python Cf_sweep_comparison.py
    python Cf_sweep_comparison.py --N 300 --R 100000 --cf 0.5 1 2 4 8 16 32

Output: Cf_sweep_comparison/sweep.csv  plus a printed summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEANUP = os.path.join(os.path.dirname(HERE), "Clean-up System")
for _p in (HERE, CLEANUP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Computational_Experiments.computing_generalCf_analytical_threshold import GeneralCfPolicy  # noqa

try:
    from solver import Params, TransshipmentDP                        # noqa
    HAVE_SOLVER = True
except Exception as _e:                                              # pragma
    HAVE_SOLVER = False
    _SOLVER_ERR = _e

# ── standing example ────────────────────────────────────────────────
LAM1, LAM2, H, PI1, PI2, CU, T = 5.0, 3.0, 1.0, 6.0, 6.0, 1.0, 5.0
I2_MAX, I2_MIN, B1_MAX = 40, -60, 90
START = (30, 2)


# ═══════════════════════════════════════════════════════════════════
#  vectorised 3-D DP, verified against solver.py
# ═══════════════════════════════════════════════════════════════════
class DP3D:
    """Backward induction on the model of solver.py, vectorised over states.
    verify_against_solver() checks it cell by cell against solver.py."""

    def __init__(self, Cf, N, lam1=LAM1, lam2=LAM2, h=H, pi1=PI1, pi2=PI2,
                 cu=CU, T_=T, I2_max=I2_MAX, I2_min=I2_MIN, b1_max=B1_MAX):
        self.__dict__.update(dict(Cf=Cf, N=N, lam1=lam1, lam2=lam2, h=h,
                                  pi1=pi1, pi2=pi2, cu=cu, T=T_,
                                  I2_max=I2_max, I2_min=I2_min,
                                  b1_max=b1_max))
        self.dt = T_ / N
        self.p1, self.p2 = lam1 * self.dt, lam2 * self.dt
        self.p0 = 1 - self.p1 - self.p2
        assert self.p0 >= 0, "dt too large"
        self.I2v = np.arange(I2_min, I2_max + 1)
        self.I2g = self.I2v[:, None]
        self.b1g = np.arange(0, b1_max + 1)[None, :]
        self.sh = (len(self.I2v), b1_max + 1)
        self.flow = np.broadcast_to(
            self.dt * (h * np.maximum(0, self.I2g) + pi1 * self.b1g
                       + pi2 * np.maximum(0, -self.I2g)), self.sh).copy()
        self.policy = None

    def _cI(self, x):
        return np.clip(x, self.I2_min, self.I2_max) - self.I2_min

    def _cB(self, x):
        return np.clip(x, 0, self.b1_max)

    def _cont(self, V, I2a, b1a):
        cI, cB = self._cI, self._cB
        return (self.p0 * V[cI(I2a), cB(b1a)]
                + self.p1 * V[cI(I2a), cB(b1a + 1)]
                + self.p2 * V[cI(I2a - 1), cB(b1a)])

    def solve(self):
        """Optimal policy tensor policy[n, I2idx, b1]."""
        V = np.zeros(self.sh)
        pol = np.zeros((self.N + 1,) + self.sh, np.int16)
        I2g, b1g, sh = self.I2g, self.b1g, self.sh
        for n in range(1, self.N + 1):
            best = self.flow + self._cont(V, I2g, b1g)
            bq = np.zeros(sh, np.int16)
            for q in range(1, self.I2_max + 1):
                feas = (I2g >= q) & (b1g >= q)
                if not feas.any():
                    break
                I2a = np.broadcast_to(I2g - q, sh)
                b1a = np.broadcast_to(b1g - q, sh)
                c = (self.Cf + self.cu * q
                     + self.dt * (self.h * np.maximum(0, I2a)
                                  + self.pi1 * b1a
                                  + self.pi2 * np.maximum(0, -I2a))
                     + self._cont(V, I2a, b1a))
                c = np.where(feas, c, np.inf)
                upd = c < best - 1e-12
                best = np.where(upd, c, best)
                bq = np.where(upd, q, bq)
            V, pol[n] = best, bq
        self.policy, self.V = pol, V
        return self

    def evaluate(self, qfun):
        """Exact policy evaluation: same recursion, actions forced by
        qfun(n) -> integer array of shape sh."""
        V = np.zeros(self.sh)
        for n in range(1, self.N + 1):
            q = qfun(n)
            I2a = np.broadcast_to(self.I2g, self.sh) - q
            b1a = np.broadcast_to(self.b1g, self.sh) - q
            V = (np.where(q > 0, self.Cf + self.cu * q, 0.0)
                 + self.dt * (self.h * np.maximum(0, I2a)
                              + self.pi1 * b1a
                              + self.pi2 * np.maximum(0, -I2a))
                 + self._cont(V, I2a, b1a))
        return V

    def at(self, arr, I2, b1):
        return arr[self._cI(I2), self._cB(b1)]

    def verify_against_solver(self, n_check=3, seed=0):
        """Cell-by-cell check against solver.py on a small instance."""
        if not HAVE_SOLVER:
            return None
        p = Params(T=1.0, N=40, lam1=self.lam1, lam2=self.lam2, h=self.h,
                   Cf=self.Cf, cu=self.cu, pi1=self.pi1, pi2=self.pi2,
                   c1=0, c2=0, v2=0, I2_max=12, I2_min=-8, b1_max=14)
        ref = TransshipmentDP(p)
        ref.solve(store_V=True, verbose=False)
        mine = DP3D(self.Cf, 40, self.lam1, self.lam2, self.h, self.pi1,
                    self.pi2, self.cu, 1.0, 12, -8, 14).solve()
        dv = max(abs(float(mine.at(mine.V, I2, b1))
                     - ref.get_value(p.N, I2, b1))
                 for I2 in range(-8, 13) for b1 in range(0, 15))
        dpol = sum(int(mine.policy[p.N, mine._cI(I2), b1]
                       != ref.get_policy(p.N, I2, b1))
                   for I2 in range(-8, 13) for b1 in range(0, 15))
        return dv, dpol


# ═══════════════════════════════════════════════════════════════════
#  analytic policy tensor from Sections 4-5
# ═══════════════════════════════════════════════════════════════════
def analytic_tensor(pol: GeneralCfPolicy, dp: DP3D):
    """q*_analytic[n, I2idx, b1] using Eq. (25)-(26)."""
    Q = np.zeros((dp.N + 1,) + dp.sh, np.int16)
    for n in range(1, dp.N + 1):
        tau = n * dp.dt
        d = pol.delta_levels(dp.I2_max, tau)          # levels 0..I2_max
        for I2 in range(1, dp.I2_max + 1):
            npos = int((d[1:I2 + 1] > 0).sum())
            if npos == 0:
                continue
            cs = np.cumsum([d[I2 - i] for i in range(npos)])
            ii = dp._cI(I2)
            for b1 in range(1, dp.b1_max + 1):
                q0 = min(b1, npos)
                if cs[q0 - 1] > pol.Cf:
                    Q[n, ii, b1] = q0
    return Q


def b1bar_from_tensor(Q, dp: DP3D, I2, n):
    row = Q[n, dp._cI(I2), 1:dp.b1_max + 1]
    nz = np.nonzero(row)[0]
    return float(nz[0] + 1) if len(nz) else math.inf


# ═══════════════════════════════════════════════════════════════════
#  continuous-time discrete-event simulation, common random numbers
# ═══════════════════════════════════════════════════════════════════
def make_paths(R, KMAX, seed):
    rng = np.random.default_rng(seed)
    lam = LAM1 + LAM2
    times = np.cumsum(rng.exponential(1 / lam, size=(R, KMAX)), axis=1)
    is1 = rng.random((R, KMAX)) < LAM1 / lam
    assert (times[:, -1] > T).all(), "increase KMAX"
    return times, is1


def simulate(Q, dp: DP3D, Cf, times, is1, start=START):
    R, KMAX = times.shape
    I2 = np.full(R, start[0], np.int64)
    b1 = np.full(R, start[1], np.int64)
    cost = np.zeros(R)
    ndisp = np.zeros(R, np.int64)
    totq = np.zeros(R, np.int64)
    t = np.zeros(R)

    def act(tau, trig):
        nonlocal I2, b1, cost, ndisp, totq
        n = np.clip(np.round(tau / dp.dt).astype(np.int64), 1, dp.N)
        q = Q[n, dp._cI(I2), dp._cB(b1)].astype(np.int64)
        q = np.where((I2 > 0) & (b1 > 0) & trig,
                     np.minimum(q, np.minimum(I2, b1)), 0)
        a = q > 0
        cost += np.where(a, Cf + CU * q, 0.0)
        ndisp += a.astype(np.int64)
        totq += q
        I2 -= q
        b1 -= q

    act(np.full(R, T), np.ones(R, bool))
    for k in range(KMAX):
        tk = times[:, k]
        seg = np.maximum(np.minimum(tk, T) - t, 0.0)
        cost += seg * (H * np.maximum(I2, 0) + PI1 * b1
                       + PI2 * np.maximum(-I2, 0))
        t = np.minimum(tk, T)
        alive = tk < T
        if not alive.any():
            break
        b1 += (alive & is1[:, k]).astype(np.int64)
        I2 -= (alive & ~is1[:, k]).astype(np.int64)
        act(T - t, alive)
    seg = np.maximum(T - t, 0.0)
    cost += seg * (H * np.maximum(I2, 0) + PI1 * b1
                   + PI2 * np.maximum(-I2, 0))
    return cost, ndisp, totq


# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", type=float, nargs="+",
                    default=[0.25, 0.5, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--R", type=int, default=50_000)
    ap.add_argument("--KMAX", type=int, default=140)
    ap.add_argument("--seed", type=int, default=20260809)
    a = ap.parse_args()

    os.makedirs(os.path.join(HERE, "Cf_sweep_comparison"), exist_ok=True)
    times, is1 = make_paths(a.R, a.KMAX, a.seed)

    if HAVE_SOLVER:
        chk = DP3D(8.0, a.N).verify_against_solver()
        print(f"[verify] vectorised DP vs solver.py: "
              f"max|V diff| = {chk[0]:.2e}, policy mismatches = {chk[1]}")
    else:
        print(f"[verify] solver.py not importable ({_SOLVER_ERR}); "
              "using the vectorised DP alone")

    I2_probe = [3, 5, 6, 7, 10, 20, 30]
    rows = []
    print(f"\nstart {START}, tau0 = {T}, {a.R:,} shared paths, N = {a.N}")
    print(f"{'Cf':>6} {'DPcost':>9} {'ANcost':>9} {'gap':>8} {'gap%':>6} "
          f"{'DPdisp':>7} {'ANdisp':>7} {'DPbatch':>8} {'ANbatch':>8} "
          f"{'agree%':>7} {'AN early':>8} {'DP early':>8}")

    for Cf in a.cf:
        dp = DP3D(Cf, a.N).solve()
        pol = GeneralCfPolicy(LAM2, H, CU, PI1, PI2, Cf)
        QAN = analytic_tensor(pol, dp)
        QDP = dp.policy

        # exact policy evaluation on the common recursion
        V_dp = dp.evaluate(lambda n: QDP[n])
        V_an = dp.evaluate(lambda n: QAN[n])
        v_dp0 = float(dp.at(V_dp, *START))
        v_an0 = float(dp.at(V_an, *START))

        # decisions over the grid
        ns = np.unique(np.linspace(1, a.N, 60).astype(int))
        I2s = np.arange(1, 26)
        b1s = np.arange(1, 16)
        agree = tot = an_early = dp_early = qsame = qdiff = 0
        for n in ns:
            d1 = QDP[n][np.ix_(dp._cI(I2s), b1s)]
            d2 = QAN[n][np.ix_(dp._cI(I2s), b1s)]
            a1, a2 = d1 > 0, d2 > 0
            tot += a1.size
            agree += int((a1 == a2).sum())
            an_early += int((a2 & ~a1).sum())
            dp_early += int((a1 & ~a2).sum())
            both = a1 & a2
            qsame += int((both & (d1 == d2)).sum())
            qdiff += int((both & (d1 != d2)).sum())

        # thresholds at tau = T
        nT = a.N
        thr = {i: (b1bar_from_tensor(QDP, dp, i, nT),
                   b1bar_from_tensor(QAN, dp, i, nT)) for i in I2_probe}

        # DES
        cdp, ndp, qdp = simulate(QDP, dp, Cf, times, is1)
        can, nan, qan = simulate(QAN, dp, Cf, times, is1)
        d = can - cdp
        ci = 1.96 * d.std() / math.sqrt(len(d))
        bd = qdp.sum() / max(ndp.sum(), 1)
        ba = qan.sum() / max(nan.sum(), 1)

        row = dict(
            Cf=Cf, tau_star=pol.tau_star(),
            V_DP_exact=v_dp0, V_AN_exact=v_an0,
            gap_exact=v_an0 - v_dp0,
            gap_exact_pct=100 * (v_an0 - v_dp0) / v_dp0,
            DES_DP=cdp.mean(), DES_AN=can.mean(),
            DES_gap=d.mean(), DES_gap_ci=ci,
            DES_gap_pct=100 * d.mean() / cdp.mean(),
            DP_dispatches=ndp.mean(), AN_dispatches=nan.mean(),
            DP_batch=bd, AN_batch=ba,
            DP_units=qdp.mean(), AN_units=qan.mean(),
            decision_agree_pct=100 * agree / tot,
            AN_dispatch_DP_wait=an_early,
            DP_dispatch_AN_wait=dp_early,
            q_same_when_both=qsame, q_diff_when_both=qdiff,
        )
        for i in I2_probe:
            row[f"b1bar_DP_I2{i}"] = ("inf" if math.isinf(thr[i][0])
                                      else int(thr[i][0]))
            row[f"b1bar_AN_I2{i}"] = ("inf" if math.isinf(thr[i][1])
                                      else int(thr[i][1]))
        rows.append(row)

        print(f"{Cf:>6g} {cdp.mean():>9.3f} {can.mean():>9.3f} "
              f"{d.mean():>8.3f} {row['DES_gap_pct']:>5.1f}% "
              f"{ndp.mean():>7.2f} {nan.mean():>7.2f} "
              f"{bd:>8.3f} {ba:>8.3f} {row['decision_agree_pct']:>6.2f}% "
              f"{an_early:>8} {dp_early:>8}")

    path = os.path.join(HERE, "Cf_sweep_comparison", "sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nwrote {path}  ({len(rows)} rows, {len(rows[0])} columns)")
    print("\nb1bar at tau = T:")
    print("   Cf  " + "".join(f"  I2={i:<2}(DP/AN)" for i in I2_probe))
    for r in rows:
        print(f" {r['Cf']:>5g}  " + "".join(
            f"  {str(r[f'b1bar_DP_I2{i}']):>3}/{str(r[f'b1bar_AN_I2{i}']):<7}"
            for i in I2_probe))


if __name__ == "__main__":
    main()