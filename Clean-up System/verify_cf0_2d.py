"""
verify_cf0_2d.py — Does the new 2-D Cf=0 DP close the staircase gap?

Collaborator's case (from the app screenshot):
    lam2 = 5, cu = 2, h = 1, Cf = 0, pi1 = 1.5, pi2 = 4, T = 5

Compares, on the SAME case:
    (a) note's analytic staircase          eq. (20)-(22)
    (b) new 2-D Cf=0 DP                    solver_cf0_2d.py   [note's own model]
    (c) existing 3-D (I2, b1) DP           solver.py          [operational model]

Metric: the tau at which each threshold curve steps up from k to k+1.
A gap that shrinks as N grows is a discretisation artifact.
A gap that converges to a non-zero limit is structural.
"""

import math
import numpy as np
from solver_cf0_2d import ParamsCf0, SwitchingDPCf0, analytic_threshold

T, LAM1, LAM2 = 5.0, 3.0, 5.0
H, CU, PI1, PI2 = 1.0, 2.0, 1.5, 4.0
STEPS = [(8, 9), (10, 11), (12, 13)]


# ── analytic transition tau ───────────────────────────────────────────
def analytic_transition(frm, to, lo, hi, p):
    prev = None
    for t in np.arange(lo, hi, 0.0002):
        v = analytic_threshold(float(t), p)
        if v == to and prev == frm:
            return float(t)
        prev = v
    return None


# ── 2-D Cf0 DP transition tau ─────────────────────────────────────────
def dp2d_transition(dp, frm, to, lo, hi):
    prev = None
    for n in range(1, dp.p.N + 1):
        tau = n * dp.p.dt
        if tau < lo or tau > hi:
            continue
        v = dp.threshold(n)
        if v == to and prev == frm:
            return tau
        prev = v
    return None


# ── 3-D (I2, b1) DP, vectorised, threshold read at a fixed b1 ──────────
def solve3d_threshold(N, b1_read=20):
    s1, s2 = LAM1 * T, LAM2 * T
    Imin = -int(math.ceil(s2 + 4 * math.sqrt(s2)))
    Imax = int(math.ceil(max(40, s2 + 4 * math.sqrt(s2))))
    b1m = int(math.ceil(s1 + 4 * math.sqrt(s1)))
    dt = T / N
    p1, p2 = LAM1 * dt, LAM2 * dt
    p0 = 1 - p1 - p2
    I2v = np.arange(Imin, Imax + 1)
    I2g = I2v[:, None]
    b1g = np.arange(0, b1m + 1)[None, :]
    cI = lambda x: np.clip(x, Imin, Imax) - Imin
    cB = lambda x: np.clip(x, 0, b1m)
    V = np.zeros((len(I2v), b1m + 1))
    qmax = min(Imax, b1m)
    br = min(b1_read, b1m)
    thr = np.full(N + 1, np.nan)
    for n in range(1, N + 1):
        best = np.full(V.shape, np.inf)
        bq = np.zeros(V.shape, int)
        for q in range(0, qmax + 1):
            feas = (np.ones(V.shape, bool) if q == 0
                    else ((I2g >= q) & (b1g >= q) & (I2g > 0) & (b1g > 0)))
            if not feas.any():
                continue
            I2a = np.broadcast_to(I2g - q, V.shape)
            b1a = np.broadcast_to(b1g - q, V.shape)
            flow = dt * (H * np.maximum(0, I2a) + PI1 * b1a
                         + PI2 * np.maximum(0, -I2a))
            cost = (CU * q + flow
                    + p0 * V[cI(I2a), cB(b1a)]
                    + p1 * V[cI(I2a), cB(b1a + 1)]
                    + p2 * V[cI(I2a - 1), cB(b1a)])
            cost = np.where(feas, cost, np.inf)
            upd = cost < best
            best = np.where(upd, cost, best)
            bq = np.where(upd, q, bq)
        V = best
        col = bq[:, br]
        pos = np.where((I2v >= 1) & (col >= 1))[0]
        thr[n] = I2v[pos[0]] if len(pos) else np.nan
    return thr, dt


def dp3d_transition(thr, dt, frm, to, lo, hi):
    prev = None
    for n in range(1, len(thr)):
        tau = n * dt
        if tau < lo or tau > hi:
            continue
        if thr[n] == to and prev == frm:
            return tau
        prev = thr[n]
    return None


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    base = ParamsCf0(T=T, N=2000, lam1=LAM1, lam2=LAM2,
                     h=H, cu=CU, pi1=PI1, pi2=PI2).with_auto_bounds()

    # locate a search window around each analytic transition
    windows = {}
    for frm, to in STEPS:
        a = analytic_transition(frm, to, 1.0, 5.0, base)
        windows[(frm, to)] = (a, max(0.3, a - 0.8), min(5.0, a + 0.8))

    print("=" * 74)
    print("Collaborator case:  lam2=5, cu=2, h=1, Cf=0, pi1=1.5, pi2=4, T=5")
    print("=" * 74)
    print("\nAnalytic (note) step-up locations:")
    for (frm, to), (a, _, _) in windows.items():
        print(f"   {frm}->{to} at tau = {a:.4f}")

    print("\n--- (b) NEW 2-D Cf=0 DP  [note's own model]  gap = analytic - DP ---")
    for N in (500, 1000, 2000, 4000, 8000, 16000):
        p = ParamsCf0(**{**vars(base), 'N': N})
        dp = SwitchingDPCf0(p)
        dp.solve(verbose=False)
        row = []
        for (frm, to), (a, lo, hi) in windows.items():
            t = dp2d_transition(dp, frm, to, lo, hi)
            row.append(f"{frm}->{to}: {a - t:+.4f}" if t else f"{frm}->{to}: NA")
        print(f"   N={N:6d} dt={T/N:.5f} | " + " | ".join(row))

    print("\n--- (c) EXISTING 3-D (I2,b1) DP  [operational model] ---")
    for N in (400, 800, 1600):
        thr, dt = solve3d_threshold(N)
        row = []
        for (frm, to), (a, lo, hi) in windows.items():
            t = dp3d_transition(thr, dt, frm, to, lo, hi)
            row.append(f"{frm}->{to}: {a - t:+.4f}" if t else f"{frm}->{to}: NA")
        print(f"   N={N:6d} dt={dt:.5f} | " + " | ".join(row))