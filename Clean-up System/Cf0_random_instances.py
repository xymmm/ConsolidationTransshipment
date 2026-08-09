"""
Cf0_random_instances.py
=======================
Random-instance study of the cu question in the Cf = 0 model.

Target: waiting states that are LATER OVERTAKEN by the falling boundary and
dispatch, because those are exactly the states whose value depends on cu.

Set-up. In the waiting mode I2 only falls, so a waiting state can reach the
dispatch region only if the boundary itself comes down to it. The boundary
falls with the passage of time iff pi2 > pi1, so all instances here are drawn
with pi1 < pi2 (the valley case of Theorem 4). A state (I2, tau0) is

    waiting now       iff  I2 <  Ibar(tau0)
    overtaken later   iff  I2 >= min_{0 < t <= tau0} Ibar(t)      [the floor]

Both can hold at once: the state waits today and dispatches later. Those are
the states reported here. For each one the value V(I2, tau0) is computed by
the exact 2-D DP over a grid of cu, holding every other parameter fixed, and
the span of V over that grid measures the cu-dependence.

The DP is solver_cf0_2d.SwitchingDPCf0, i.e. the note's own 2-D model.
The threshold is Eq. (20)-(22) via computing_Cf0_analytical_threshold.

Output: Cf0_random_instances/random_instances.csv  (one row per cu point)
        Cf0_random_instances/summary.csv           (one row per instance)

Usage
    python Cf0_random_instances.py
    python Cf0_random_instances.py --n-instances 40 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(ROOT, "Clean-up System"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from solver_cf0_2d import ParamsCf0, SwitchingDPCf0           # noqa: E402
from Computational_Experiments.computing_Cf0_analytical_threshold import threshold      # noqa: E402

OUTDIR = os.path.join(HERE, "Cf0_random_instances")
INF = float("inf")


# ─────────────────────────────────────────────────────────────────────
def floor_over_future(tau0, lam2, h, cu, pi1, pi2, grid=400):
    """min_{0 < t <= tau0} Ibar(t): the lowest threshold the state will face."""
    lo = INF
    for k in range(1, grid + 1):
        v = threshold(tau0 * k / grid, lam2, h, cu, pi1, pi2)
        if v < lo:
            lo = v
    return lo


def solve_V(lam1, lam2, h, cu, pi1, pi2, tau0, N=4000):
    """V(., tau0) from the note's 2-D DP; returns (I2 grid, V)."""
    p = ParamsCf0(T=tau0, N=N, lam1=lam1, lam2=lam2, h=h, cu=cu,
                  pi1=pi1, pi2=pi2, c2=0.0, v2=0.0).with_auto_bounds()
    dp = SwitchingDPCf0(p)
    dp.solve(store_V=False, verbose=False)
    return np.arange(p.I2_min, p.I2_max + 1), dp.V_final, p


def draw_instance(rng):
    """A random valley instance: pi1 < pi2."""
    lam1 = float(np.round(float(np.exp(rng.uniform(np.log(1.0), np.log(25.0)))), 2))
    lam2 = float(np.round(float(np.exp(rng.uniform(np.log(0.3), np.log(8.0)))), 2))
    h = float(np.round(rng.uniform(0.2, 3.0), 2))
    pi1 = float(np.round(rng.uniform(0.2, 4.0), 2))
    pi2 = float(np.round(pi1 + rng.uniform(0.5, 8.0), 2))
    T = float(np.round(rng.uniform(3.0, 10.0), 1))
    return lam1, lam2, h, pi1, pi2, T


def study_instance(lam1, lam2, h, pi1, pi2, T, cu_grid, N=4000):
    """
    Find a state that waits at tau0 = T for EVERY cu in the grid but is
    overtaken for at least one of them, then evaluate V across the grid.
    Returns (state, rows) or (None, []) when no such state exists.
    """
    tau0 = T
    thr_now = {cu: threshold(tau0, lam2, h, cu, pi1, pi2) for cu in cu_grid}
    flr = {cu: floor_over_future(tau0, lam2, h, cu, pi1, pi2)
           for cu in cu_grid}

    # waiting at tau0 for all cu  ->  I2 < min_cu Ibar(tau0)
    hi = min(thr_now.values())
    if not math.isfinite(hi) or hi < 2:
        return None, []
    # overtaken for at least one cu  ->  I2 >= min_cu floor
    lo = min(flr.values())
    if not math.isfinite(lo):
        return None, []
    cand = [I2 for I2 in range(int(math.ceil(lo)), int(hi))]
    if not cand:
        return None, []
    # prefer the state overtaken under the most cu values: strongest effect
    def n_over(I2):
        return sum(1 for cu in cu_grid if I2 >= flr[cu])
    I2s = max(cand, key=lambda x: (n_over(x), -x))

    rows = []
    for cu in cu_grid:
        I2v, V, p = solve_V(lam1, lam2, h, cu, pi1, pi2, tau0, N)
        v = float(V[int(np.clip(I2s, p.I2_min, p.I2_max)) - p.I2_min])
        rows.append(dict(
            lam1=lam1, lam2=lam2, h=h, pi1=pi1, pi2=pi2, T=T,
            I2=I2s, tau0=tau0, cu=cu,
            Ibar_now=thr_now[cu],
            floor_future=flr[cu],
            waiting_now=I2s < thr_now[cu],
            overtaken_later=I2s >= flr[cu],
            V=v))
    return I2s, rows


def _fmt(x):
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    if isinstance(x, float):
        return "inf" if math.isinf(x) else f"{x:.10g}"
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-instances", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--cu-points", type=int, default=13)
    ap.add_argument("--N", type=int, default=4000)
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    all_rows, summary = [], []
    tried = kept = 0
    while kept < a.n_instances and tried < a.n_instances * 12:
        tried += 1
        lam1, lam2, h, pi1, pi2, T = draw_instance(rng)
        # cu grid scaled to the instance: tau* = cu/(h+pi1) must leave room
        cu_hi = (h + pi1) * T * 0.85
        if cu_hi < 0.4:
            continue
        cu_grid = [float(np.round(c, 3))
                   for c in np.linspace(cu_hi * 0.15, cu_hi, a.cu_points)]
        I2s, rows = study_instance(lam1, lam2, h, pi1, pi2, T, cu_grid, a.N)
        if I2s is None:
            continue
        over = [r for r in rows if r["overtaken_later"]]
        safe = [r for r in rows if not r["overtaken_later"]]
        if not over:
            continue                       # want the interesting case
        kept += 1
        vs = [r["V"] for r in rows]
        span = max(vs) - min(vs)
        rel = span / max(abs(np.mean(vs)), 1e-12) * 100
        # where the state stops being overtaken
        cu_star = min((r["cu"] for r in safe), default=None)
        span_over = (max(r["V"] for r in over) - min(r["V"] for r in over)
                     if len(over) > 1 else 0.0)
        span_safe = (max(r["V"] for r in safe) - min(r["V"] for r in safe)
                     if len(safe) > 1 else 0.0)
        # dV/dcu on the overtaken side = E[number of future dispatches]
        if len(over) > 1:
            xs = np.array([r["cu"] for r in over])
            ys = np.array([r["V"] for r in over])
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = 0.0
        for r in rows:
            r["instance"] = kept
        all_rows.extend(rows)
        summary.append(dict(
            instance=kept, lam1=lam1, lam2=lam2, h=h, pi1=pi1, pi2=pi2, T=T,
            I2=I2s, tau0=T,
            cu_min=cu_grid[0], cu_max=cu_grid[-1],
            n_cu_overtaken=len(over), n_cu_safe=len(safe),
            cu_star=cu_star if cu_star is not None else INF,
            V_span_all=span, V_span_rel_pct=rel,
            V_span_overtaken=span_over, V_span_safe=span_safe,
            dV_dcu_overtaken=slope))
        print(f"  [{kept:>2}] lam1={lam1:<5} lam2={lam2:<5} h={h:<4} "
              f"pi1={pi1:<4} pi2={pi2:<5} T={T:<4} | state (I2={I2s}, "
              f"tau0={T}) | overtaken at {len(over)}/{len(rows)} cu | "
              f"V span {span:.4f}  dV/dcu {slope:.4f}  "
              f"safe-side span {span_safe:.6f}")

    if not all_rows:
        print("no qualifying instance found; widen the draw ranges")
        return

    p1 = os.path.join(OUTDIR, "random_instances.csv")
    with open(p1, "w", newline="") as f:
        keys = list(all_rows[0].keys())
        w = csv.writer(f)
        w.writerow(keys)
        for r in all_rows:
            w.writerow([_fmt(r[k]) for k in keys])
    p2 = os.path.join(OUTDIR, "summary.csv")
    with open(p2, "w", newline="") as f:
        keys = list(summary[0].keys())
        w = csv.writer(f)
        w.writerow(keys)
        for r in summary:
            w.writerow([_fmt(r[k]) for k in keys])

    sp = [s["V_span_all"] for s in summary]
    sf = [s["V_span_safe"] for s in summary]
    print(f"\n{kept} qualifying instances out of {tried} draws")
    print(f"  V span over the cu grid: min {min(sp):.4f}  median "
          f"{float(np.median(sp)):.4f}  max {max(sp):.4f}")
    print(f"  span on the SAFE side (state never overtaken): max {max(sf):.6f}")
    sl = [s["dV_dcu_overtaken"] for s in summary]
    print(f"  dV/dcu on the overtaken side: min {min(sl):.4f}  max {max(sl):.4f}"
          "   (= E[number of future dispatches])")
    print("  Reading: V moves with cu exactly while the state is overtaken,")
    print("  and is flat once the floor rises above it.")
    print(f"\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()