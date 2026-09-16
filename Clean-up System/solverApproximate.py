"""
solverApproximate.py  —  solve once, then check / modify / evaluate offline
==========================================================================
Standalone companion to app.py. No Streamlit and no Excel needed.

    pip install numpy pandas            # required
    pip install matplotlib              # only for --plots
    pip install openpyxl                # only for --format xlsx

Two stages
----------
STAGE 1  solve   Run solver.py (3-D model) or solver_cf0_2d.py (2-D Cf=0
                 model of the note) and save the optimal policy table and
                 value function to a compressed .npz file. This is the only
                 stage that needs the solver files.

STAGE 2  approx  Load one or more .npz files and, without re-solving,
                 CHECK    the optimal table for structural irregularities,
                 MODIFY   it with a monotone operator (fill M- / remove M+),
                 EVALUATE the modified table exactly by backward induction
                          and report the gap V_mod - V*,
                 REPORT   plain-language conclusions in report.md: is the
                          evaluation valid, is there a structural bump, is
                          monotonicity established, how much cost is lost,
                          and which operator is better.

`run` does both stages in one call.

The two models
--------------
3d  solver.py, state (I2, b1). Structural property checked: the dispatch
    threshold b1bar(I2, tau) is non-increasing in I2.
      fill   : b1bar -> running minimum over smaller I2; waiting cells below
               the DP's own b1bar dispatch the best q under V*.
      remove : b1bar -> running maximum over larger I2; dispatching cells
               below it wait.
2d  solver_cf0_2d.py, state I2 only (Cf = 0 note). An arriving Retailer-1
    demand is served at cost cu or rejected at cost pi1*tau. Structural
    property checked: the serve set is an upper set in I2, i.e. serve iff
    I2 >= I2bar(tau).
      fill   : serve at every I2 >= the smallest serving level.
      remove : reject at every I2 below the largest rejecting level + 1.
    Evaluating a 2-D table requires the 2-D Bellman convention. The script
    recovers it by re-solving with a few candidate conventions and keeping
    the one that reproduces the saved optimal table; the match rate is
    printed and must be 100% for the gaps to be trusted.

Cf = 0 in the 3d model is NOT the 2d model. Do not compare the two.

Three ways to set parameters (later ones override earlier ones)
---------------------------------------------------------------
1. the CONFIG block below: edit it and press Run in PyCharm
2. a JSON file with any subset of CONFIG keys:   --config my_case.json
3. command-line flags, e.g.
    python solverApproximate.py run --model 3d --Cf 0 4 8 --pi1 4 6 8
    python solverApproximate.py solve --model both --N 400 --b1_max 130
    python solverApproximate.py approx --files "solved/*.npz" --format xlsx
Every model parameter accepts several values; all combinations are run.

From Python
-----------
    from solverApproximate import Base, solve_3d, solve_2d, approx_file
    f = solve_3d(Base(N=400), Cf=8)          # returns the .npz path
    tables = approx_file(f)                  # dict of DataFrames
    print(tables["summary"])
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd


# ======================================================================
# USER CONFIGURATION  —  edit everything here, then press Run
# ======================================================================
# Any MODEL PARAMETER may be a single number or a list. Lists are swept:
# every combination of all listed values is solved and analysed.
#   e.g.  Cf=[0, 4, 8], pi1=[4, 6, 8]  ->  3 x 3 = 9 instances.
# Bounds may be a number or None (None = auto rule of app.py).
# The command line and a JSON file (--config) can override any key.
CONFIG = dict(
    # what to do
    stage="run",              # "run" = solve + analyse, "solve", "approx"
    model="3d",               # "3d" (solver.py), "2d" (solver_cf0_2d.py), "both"

    # model parameters (number or list)
    T=5.0,
    N=400,                    # periods, 3d model
    N2d=2000,                 # periods, 2d model
    lam1=5.0,
    lam2=3.0,
    h=1.0,
    cu=1.0,
    pi1=6.0,
    pi2=6.0,
    Cf=[8.0],                 # 3d model only; the 2d model is Cf = 0
    c1=0.0,
    c2=0.0,
    v2=0.0,
    I2_min=-31,               # None = auto
    I2_max=40,                # None = auto
    b1_max=130,               # None = auto; keep it well above lam1*T

    # analysis
    files=["solved/*.npz"],   # stage "approx" only: files or glob patterns
    ops=["fill", "remove"],   # "fill" (M-), "remove" (M+)
    state=[30, 2],            # reporting state (I2, b1); 2d uses I2 only
    b1_cap=40,                # 3d: b1 range used for region maxima
    top_margin=5,             # 2d: ignore the top levels near I_max

    # output
    save_dir="solved",        # where .npz solutions go
    out="approx_out",         # where tables and report.md go
    format="csv",             # "csv" or "xlsx"
    plots="plots",            # folder for PNGs, or None
)

SWEEP_3D = ["T", "N", "lam1", "lam2", "h", "cu", "pi1", "pi2", "Cf",
            "c1", "c2", "v2", "I2_min", "I2_max", "b1_max"]
SWEEP_2D = ["T", "N2d", "lam1", "lam2", "h", "cu", "pi1", "pi2", "c2", "v2",
            "I2_min", "I2_max"]
INT_KEYS = {"N", "N2d", "I2_min", "I2_max", "b1_max", "b1_cap",
            "top_margin"}

# ======================================================================
# CONFIGURATION
# ======================================================================
@dataclass
class Base:
    """All model parameters except Cf. Bounds of None mean auto bounds."""
    T: float = 5.0
    N: int = 400
    lam1: float = 5.0
    lam2: float = 3.0
    h: float = 1.0
    cu: float = 1.0
    pi1: float = 6.0
    pi2: float = 6.0
    c1: float = 0.0
    c2: float = 0.0
    v2: float = 0.0
    I2_min: int | None = None
    I2_max: int | None = None
    b1_max: int | None = None

    def bounds(self):
        """Same auto-bound rule as app.py: expected demand plus a 4-sigma buffer."""
        s1, s2 = self.lam1 * self.T, self.lam2 * self.T
        auto = (-int(math.ceil(s2 + 4.0 * math.sqrt(s2))),
                int(math.ceil(max(40.0, s2 + 4.0 * math.sqrt(s2)))),
                int(math.ceil(s1 + 4.0 * math.sqrt(s1))))
        given = (self.I2_min, self.I2_max, self.b1_max)
        return tuple(g if g is not None else a for g, a in zip(given, auto))


def _tag(x):
    return f"{x:g}".replace(".", "p").replace("-", "m")


def _fname(prefix, params, keys):
    """Readable name from the main parameters plus a short hash of all of them."""
    import hashlib
    h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    parts = [f"{k}{_tag(params[k])}" for k in keys if k in params]
    return f"{prefix}_" + "_".join(parts) + f"_{h}.npz"


def _save(path, model, params, policy, V):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = dict(model=np.array(model), params=np.array(json.dumps(params)),
                  policy=policy)
    if V is not None:
        arrays["V"] = V
    np.savez_compressed(path, **arrays)
    return path


def load(path):
    d = np.load(path, allow_pickle=False)
    V = d["V"] if "V" in d.files else None
    return str(d["model"]), json.loads(str(d["params"])), d["policy"], V


# ======================================================================
# STAGE 1: SOLVE AND SAVE
# ======================================================================
def solve_3d(base: Base, Cf: float, save_dir="solved", verbose=True):
    from solver import Params, TransshipmentDP
    I2_min, I2_max, b1_max = base.bounds()
    p = Params(T=base.T, N=base.N, lam1=base.lam1, lam2=base.lam2,
               h=base.h, Cf=Cf, cu=base.cu, pi1=base.pi1, pi2=base.pi2,
               c1=base.c1, c2=base.c2, v2=base.v2,
               I2_max=I2_max, I2_min=I2_min, b1_max=b1_max)
    dp = TransshipmentDP(p)
    t0 = time.time()
    dp.solve(store_V=True, verbose=False)
    N = p.N
    policy = np.stack([np.asarray(dp.policy[n]) for n in range(1, N + 1)]
                      ).astype(np.int16)
    Vs = [np.asarray(v, float) for v in dp.V_all]
    if len(Vs) < N + 1:            # solver keeps V^0..V^{N-1} only
        Vs.append(np.array([[dp.get_value(N, i, b)
                             for b in range(0, b1_max + 1)]
                            for i in range(I2_min, I2_max + 1)]))
    V = np.stack(Vs[:N + 1])
    params = dict(asdict(base), Cf=float(Cf), N=int(N),
                  I2_min=int(p.I2_min), I2_max=int(p.I2_max),
                  b1_max=int(p.b1_max), dt=float(p.dt),
                  p0=float(p.p0), p1=float(p.p1), p2=float(p.p2))
    path = os.path.join(save_dir, _fname(
        "model3d", params, ["Cf", "N", "T", "lam1", "lam2", "h", "cu",
                            "pi1", "pi2"]))
    _save(path, "3d", params, policy, V)
    if verbose:
        print(f"  3d Cf={Cf:g} solved in {time.time() - t0:.1f}s -> {path}")
    return path


def solve_2d(base: Base, N2d=2000, save_dir="solved", verbose=True):
    from solver_cf0_2d import ParamsCf0, SwitchingDPCf0
    p0 = ParamsCf0(T=base.T, N=N2d, lam1=base.lam1, lam2=base.lam2,
                   h=base.h, cu=base.cu, pi1=base.pi1, pi2=base.pi2,
                   c2=base.c2, v2=base.v2).with_auto_bounds()
    dp = SwitchingDPCf0(p0)
    t0 = time.time()
    dp.solve(verbose=False)
    q = dp.p
    N = int(getattr(q, "N", N2d))
    lo = hi = None
    for a, b in (("I2_min", "I2_max"), ("I_min", "I_max"),
                 ("Imin", "Imax"), ("I2min", "I2max")):
        if hasattr(q, a) and hasattr(q, b):
            lo, hi = int(getattr(q, a)), int(getattr(q, b))
            break
    if lo is None:
        lo, hi, _ = base.bounds()
        if verbose:
            print(f"  2d: bounds not found on the solver, using [{lo}, {hi}]")
    levels = np.arange(lo, hi + 1)
    policy = np.zeros((N, levels.size), np.int8)
    for n in range(1, N + 1):
        for k, I in enumerate(levels):
            if I < 1:
                continue
            try:
                policy[n - 1, k] = int(dp.get_policy(n, int(I)) > 0)
            except Exception:
                pass
    V = None
    for name in ("V_all", "V"):
        cand = getattr(dp, name, None)
        if cand is None:
            continue
        try:
            arr = np.asarray(cand, float)
            if arr.shape == (N + 1, levels.size):
                V = arr
                break
        except Exception:
            pass
    params = dict(T=float(base.T), N=N, lam1=base.lam1, lam2=base.lam2,
                  h=base.h, cu=base.cu, pi1=base.pi1, pi2=base.pi2,
                  c2=base.c2, v2=base.v2, I_min=lo, I_max=hi, Cf=0.0)
    path = os.path.join(save_dir, _fname(
        "model2d", params, ["N", "T", "lam1", "lam2", "h", "cu", "pi1",
                            "pi2"]))
    _save(path, "2d", params, policy, V)
    if verbose:
        print(f"  2d solved in {time.time() - t0:.1f}s -> {path}"
              + ("" if V is not None else "  (no value array exposed)"))
    return path


# ======================================================================
# SHARED HELPERS
# ======================================================================
def _viol(b):
    """Increases (or finite -> +inf) between consecutive entries on the last axis."""
    a, c = b[..., :-1], b[..., 1:]
    return (c > a) & np.isfinite(a)


def _first(D):
    """Smallest 1-based column index with D True in each row, +inf if none."""
    return np.where(D.any(axis=1), D.argmax(axis=1) + 1.0, np.inf)


def _rel(g, v):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(v) > 1e-9, 100 * g / np.abs(v), np.nan)


# ======================================================================
# STAGE 2, 3-D MODEL
# ======================================================================
def _grids3(p):
    I2g = np.arange(p.I2_min, p.I2_max + 1)[:, None]
    b1g = np.arange(0, p.b1_max + 1)[None, :]
    return I2g, b1g, (I2g.shape[0], p.b1_max + 1)


def _branch3(p, V, q, I2g, b1g, sh):
    """Q-value of dispatching q (q = 0 waits) against next-stage V."""
    cI = lambda x: np.clip(x, p.I2_min, p.I2_max) - p.I2_min
    cB = lambda x: np.clip(x, 0, p.b1_max)
    I2a = np.broadcast_to(I2g - q, sh)
    b1a = np.broadcast_to(b1g - q, sh)
    g = (p.Cf if q > 0 else 0.0) + p.cu * q + p.dt * (
        p.h * np.maximum(0, I2a) + p.pi1 * b1a + p.pi2 * np.maximum(0, -I2a))
    return (g + p.p0 * V[cI(I2a), cB(b1a)]
              + p.p1 * V[cI(I2a), cB(b1a + 1)]
              + p.p2 * V[cI(I2a - 1), cB(b1a)])


def _best3(p, V, I2g, b1g, sh):
    best = np.full(sh, np.inf)
    bq = np.zeros(sh, int)
    for q in range(1, max(1, min(p.I2_max, p.b1_max)) + 1):
        feas = (I2g >= q) & (b1g >= q)
        if not feas.any():
            break
        val = np.where(feas, _branch3(p, V, q, I2g, b1g, sh), np.inf)
        better = val < best
        best = np.where(better, val, best)
        bq = np.where(better, q, bq)
    return bq, best


def _eval3(p, V, A, I2g, b1g, sh):
    out = np.empty(sh)
    for q in np.unique(A):
        m = A == q
        out[m] = _branch3(p, V, int(q), I2g, b1g, sh)[m]
    return out


def approx_3d(params, policy, V, ops, state, b1_cap, tol=1e-9,
              plots=None, label="", verbose=True):
    p = SimpleNamespace(**params)
    if V is None:
        raise ValueError("3d file has no value array")
    I2g, b1g, sh = _grids3(p)
    N, K, B = p.N, p.I2_max, p.b1_max
    rows = slice(1 - p.I2_min, p.I2_max - p.I2_min + 1)
    r0 = 1 - p.I2_min
    si = int(np.clip(state[0], p.I2_min, p.I2_max)) - p.I2_min
    sb = int(np.clip(state[1], 0, B))
    cap = int(min(b1_cap, B))
    cols = np.arange(1, B + 1)[None, :]
    taus = np.arange(0, N + 1) * p.T / N

    # ── CHECK ──────────────────────────────────────────────────────────
    t0 = time.time()
    bw = np.full((N, K), np.inf); bd = np.full((N, K), np.inf)
    ties = np.zeros(N, int); holes = np.zeros(N, int)
    for n in range(1, N + 1):
        Vp = V[n - 1]
        _, best = _best3(p, Vp, I2g, b1g, sh)
        M = (_branch3(p, Vp, 0, I2g, b1g, sh) - best)[rows, 1:]
        Dw = policy[n - 1][rows, 1:] > 0
        bw[n - 1], bd[n - 1] = _first(Dw), _first(M >= -tol)
        ties[n - 1] = int((np.abs(M) <= tol).sum())
        holes[n - 1] = int(((cols >= bw[n - 1][:, None]) & (cols <= cap)
                            & ~Dw).any(axis=1).sum())
    vw, vd = _viol(bw), _viol(bd)
    check = pd.DataFrame(dict(
        n=np.arange(1, N + 1), tau=np.round(taus[1:], 6),
        I2_viol_tie_wait=vw.sum(1), I2_viol_tie_dispatch=vd.sum(1),
        I2_viol_both_rules=(vw & vd).sum(1), tie_cells=ties,
        not_upper_set_rows=holes))
    totals = dict(I2_viol_tie_wait=int(vw.sum()),
                  I2_viol_tie_dispatch=int(vd.sum()),
                  I2_viol_both_rules=int((vw & vd).sum()),
                  tau_viol=int(_viol(bw.T).sum()),
                  tie_cells=int(ties.sum()),
                  not_upper_set_rows=int(holes.sum()))
    if verbose:
        print(f"  check {time.time() - t0:.1f}s: I2 violations "
              f"{totals['I2_viol_tie_wait']} (ties to wait), "
              f"{totals['I2_viol_tie_dispatch']} (ties to dispatch), "
              f"{totals['I2_viol_both_rules']} under both rules, "
              f"tau violations {totals['tau_viol']}")

    # ── MODIFY + EVALUATE ──────────────────────────────────────────────
    summary, gaps, cells_all = [], [], []
    for op in ops:
        t0 = time.time()
        Vd = V[0].astype(float).copy(); Vm = Vd.copy()
        g_s = np.zeros(N + 1); v_s = np.zeros(N + 1); v_s[0] = Vd[si, sb]
        mg = np.zeros(N + 1); mr = np.zeros(N + 1); nmod = np.zeros(N + 1, int)
        bnew = np.full((N, K), np.inf); min_gap = 0.0; cells = []
        lv_shift = 0; mx_shift = 0.0
        for n in range(1, N + 1):
            pol = policy[n - 1].astype(int)
            A = pol.copy()
            sub = A[rows, 1:]
            D = sub > 0
            bb = bw[n - 1]
            if op == "fill":
                env = np.minimum.accumulate(bb)
                mask = (~D) & (cols >= env[:, None]) & (cols < bb[:, None])
                if mask.any():
                    bq, _ = _best3(p, V[n - 1], I2g, b1g, sh)
                    sub[mask] = bq[rows, 1:][mask]
            else:
                env = np.maximum.accumulate(bb[::-1])[::-1]
                mask = D & (cols < env[:, None])
                sub[mask] = 0
            bnew[n - 1] = _first(sub > 0)
            moved = bnew[n - 1] != bb
            lv_shift = max(lv_shift, int(moved.sum()))
            both_f = moved & np.isfinite(bnew[n - 1]) & np.isfinite(bb)
            if both_f.any():
                mx_shift = max(mx_shift, float(np.abs(
                    bnew[n - 1][both_f] - bb[both_f]).max()))
            Vd = _eval3(p, Vd, pol, I2g, b1g, sh)
            Vm = _eval3(p, Vm, A, I2g, b1g, sh)
            G = Vm - Vd
            min_gap = min(min_gap, float(G.min()))
            g_s[n], v_s[n] = G[si, sb], Vd[si, sb]
            reg, regV = G[r0:, :cap + 1], Vd[r0:, :cap + 1]
            mg[n] = reg.max()
            mr[n] = np.nanmax(np.nan_to_num(_rel(reg, regV)))
            nmod[n] = int(mask.sum())
            if mask.any():
                ii, jj = np.nonzero(mask)
                cells.append(pd.DataFrame(dict(
                    n=n, tau=round(n * p.T / N, 6), I2=ii + 1, b1=jj + 1,
                    q_opt=pol[ii + r0, jj + 1], q_mod=A[ii + r0, jj + 1],
                    gap_here=G[ii + r0, jj + 1])))
        sanity = float(np.max(np.abs(Vd - V[N])))
        rel = _rel(g_s, v_s)
        regT = G[r0:, :cap + 1]
        wi, wb = np.unravel_index(int(np.argmax(regT)), regT.shape)
        worst_state = f"({wi + 1}, {wb})"
        worst_tau = float(taus[int(np.argmax(mg))])
        if verbose:
            flag = "" if sanity <= 1e-6 else "   <-- SANITY CHECK FAILED"
            print(f"  {op:6s} {time.time() - t0:5.1f}s  modified={nmod.sum():6d}  "
                  f"gap@({state[0]},{state[1]})={g_s[-1]:.4f} ({rel[-1]:.3f}%)  "
                  f"max gap at T={mg[-1]:.4f}  check={sanity:.1e}{flag}")
        summary.append(dict(
            file=label, model="3d", Cf=p.Cf, operator=op,
            state=f"({state[0]}, {state[1]})", V_opt=v_s[-1],
            V_mod=v_s[-1] + g_s[-1], gap=g_s[-1], rel_gap_pct=rel[-1],
            max_gap_region_at_T=mg[-1], max_rel_gap_region_at_T_pct=mr[-1],
            max_gap_region_any_tau=mg.max(), modified_cells=int(nmod.sum()),
            periods_touched=int((nmod > 0).sum()),
            worst_state_at_T=worst_state, worst_tau=worst_tau,
            max_levels_shifted=lv_shift, max_threshold_shift=mx_shift,
            viol_after=int(_viol(bnew).sum()),
            tau_viol_after=int(_viol(bnew.T).sum()), min_gap=min_gap,
            sanity_check=sanity, **totals))
        gaps.append(pd.DataFrame(dict(
            file=label, model="3d", Cf=p.Cf, operator=op,
            n=np.arange(0, N + 1), tau=np.round(taus, 6),
            modified_cells=nmod, V_opt_state=v_s, gap_state=g_s,
            rel_gap_state_pct=rel, max_gap_region=mg,
            max_rel_gap_region_pct=mr)))
        c = (pd.concat(cells, ignore_index=True) if cells else pd.DataFrame(
            columns=["n", "tau", "I2", "b1", "q_opt", "q_mod", "gap_here"]))
        c.insert(0, "operator", op); c.insert(0, "Cf", p.Cf)
        c.insert(0, "model", "3d"); c.insert(0, "file", label)
        cells_all.append(c)
        if plots:
            _plot(plots, f"{label}_{op}", taus, bw, bnew, c["I2"], c["tau"],
                  g_s, mg, f"state ({state[0]}, {state[1]})",
                  f"I₂ ≥ 1, b₁ ≤ {cap}", "b̄₁", "I₂",
                  f"3d Cf={p.Cf:g}, {op}: gap at T = {g_s[-1]:.4f} "
                  f"({rel[-1]:.3f}%)")
    check.insert(0, "Cf", p.Cf); check.insert(0, "model", "3d")
    check.insert(0, "file", label)
    return dict(summary=pd.DataFrame(summary), check=check,
                gaps=pd.concat(gaps, ignore_index=True),
                cells=pd.concat(cells_all, ignore_index=True))


# ======================================================================
# STAGE 2, 2-D MODEL (Cf = 0 note)
# ======================================================================
def _bellman2(p, conv, policy=None):
    """
    Backward pass of the 2-D switching model under a convention
    conv = (shift, tie):  tau at period n is (n - shift) * dt, and on an exact
    tie the optimiser serves (tie='serve') or rejects (tie='reject').
    With policy=None the optimal policy is computed; otherwise the given
    serve table is evaluated. Returns (V stack, serve table, tie count).
    """
    shift, tie = conv
    N, dt = p.N, p.T / p.N
    p1, p2 = p.lam1 * dt, p.lam2 * dt
    p0 = 1.0 - p1 - p2
    I = np.arange(p.I_min, p.I_max + 1)
    dn = np.clip(I - 1, p.I_min, p.I_max) - p.I_min
    V = p.c2 * np.maximum(-I, 0) - p.v2 * np.maximum(I, 0)
    Vs = np.empty((N + 1, I.size)); Vs[0] = V
    S = np.zeros((N, I.size), bool)
    flow = dt * (p.h * np.maximum(I, 0) + p.pi2 * np.maximum(-I, 0))
    ties = 0
    for n in range(1, N + 1):
        tau = (n - shift) * dt
        sv = p.cu + V[dn]
        rj = p.pi1 * tau + V
        if policy is None:
            s = (sv <= rj) if tie == "serve" else (sv < rj)
            s &= I >= 1
            ties += int(((np.abs(sv - rj) <= 1e-9) & (I >= 1)).sum())
        else:
            s = policy[n - 1].astype(bool)
        S[n - 1] = s
        V = flow + p0 * V + p1 * np.where(s, sv, rj) + p2 * V[dn]
        Vs[n] = V
    return Vs, S, ties


def approx_2d(params, policy, V, ops, state, top_margin, plots=None,
              label="", verbose=True):
    p = SimpleNamespace(**params)
    N = p.N
    I = np.arange(p.I_min, p.I_max + 1)
    stored = policy.astype(bool)
    pos = I >= 1
    cap_level = p.I_max - int(top_margin)
    region = pos & (I <= cap_level)
    taus = np.arange(0, N + 1) * p.T / N

    # recover the Bellman convention that reproduces the saved table
    best = None
    for conv in ((0, "serve"), (0, "reject"), (1, "serve"), (1, "reject"),
                 (0.5, "serve"), (0.5, "reject")):
        Vs, S, ties = _bellman2(p, conv)
        match = float((S[:, pos] == stored[:, pos]).mean())
        vchk = (float(np.max(np.abs(Vs - V))) if V is not None
                and V.shape == Vs.shape else np.nan)
        if best is None or match > best[1] or (
                match == best[1] and np.nan_to_num(vchk, nan=np.inf)
                < np.nan_to_num(best[2], nan=np.inf)):
            best = (conv, match, vchk, ties)
    conv, match, vchk, ties = best
    if verbose:
        ok = match == 1.0 and (np.isnan(vchk) or vchk <= 1e-6)
        print(f"  2d convention tau=(n-{conv[0]})dt, ties->{conv[1]}: "
              f"policy match {100 * match:.3f}%"
              + ("" if np.isnan(vchk) else f", |V - V_solver| = {vchk:.1e}")
              + ("" if ok else "   <-- CONVENTION NOT RECOVERED, gaps unreliable"))
    Vopt, _, _ = _bellman2(p, conv, stored)

    # ── CHECK ──────────────────────────────────────────────────────────
    sub = stored[:, region]
    Ibar = _first(sub)                         # 1-based within region
    lvl0 = int(I[region][0]) if region.any() else 1
    cols = np.arange(1, sub.shape[1] + 1)[None, :]
    hole_cells = (cols >= Ibar[:, None]) & ~sub
    ibar_level = np.where(np.isfinite(Ibar), Ibar + lvl0 - 1, np.inf)
    tau_viol = int(_viol(ibar_level[None, :]).sum())
    tau_dec = int((ibar_level[1:] < ibar_level[:-1]).sum())
    check = pd.DataFrame(dict(
        n=np.arange(1, N + 1), tau=np.round(taus[1:], 6),
        I2bar=ibar_level, hole_cells=hole_cells.sum(1)))
    totals = dict(hole_cells=int(hole_cells.sum()),
                  periods_with_holes=int(hole_cells.any(1).sum()),
                  I2bar_increases_in_tau=tau_viol,
                  I2bar_decreases_in_tau=tau_dec,
                  tie_cells=int(ties), policy_match_pct=100 * match,
                  convention=f"tau=(n-{conv[0]})dt, ties->{conv[1]}")
    if verbose:
        print(f"  check: hole cells {totals['hole_cells']} in "
              f"{totals['periods_with_holes']} periods, I2bar increases in tau "
              f"{tau_viol}, decreases {tau_dec}"
              + ("  (pi1 >= pi2: I2bar should not increase)"
                 if p.pi1 >= p.pi2 else ""))

    si = int(np.clip(state[0], p.I_min, p.I_max)) - p.I_min
    summary, gaps, cells_all = [], [], []
    for op in ops:
        M = stored.copy()
        R = M[:, region]
        if op == "fill":
            R = cols >= Ibar[:, None]
        else:
            rej = ~R
            top = np.where(rej.any(1),
                           rej.shape[1] - np.argmax(rej[:, ::-1], axis=1), 0)
            R = R & (cols > top[:, None])
        M[:, region] = R
        changed = M != stored
        Vm, _, _ = _bellman2(p, conv, M)
        G = Vm - Vopt
        rel = _rel(G[:, si], Vopt[:, si])
        mg = G[:, region].max(axis=1) if region.any() else np.zeros(N + 1)
        mr = (np.nanmax(np.nan_to_num(_rel(G[:, region], Vopt[:, region])),
                        axis=1) if region.any() else np.zeros(N + 1))
        nmod = np.r_[0, changed.sum(1)]
        ibar_new = np.where(R.any(1), R.argmax(1) + lvl0, np.inf)
        holes_after = int(((cols >= _first(R)[:, None]) & ~R).sum())
        worst_state = (f"I2={int(I[region][int(np.argmax(G[-1, region]))])}"
                       if region.any() else "")
        worst_tau = float(taus[int(np.argmax(mg))])
        nn, kk = np.nonzero(changed)
        c = pd.DataFrame(dict(
            file=label, model="2d", Cf=0.0, operator=op, n=nn + 1,
            tau=np.round((nn + 1) * p.T / N, 6), I2=I[kk],
            serve_opt=stored[nn, kk].astype(int),
            serve_mod=M[nn, kk].astype(int), gap_here=G[nn + 1, kk]))
        if verbose:
            print(f"  {op:6s} modified={int(changed.sum()):6d}  "
                  f"gap@I2={state[0]}: {G[-1, si]:.4f} ({rel[-1]:.3f}%)  "
                  f"max gap at T={mg[-1]:.4f}")
        summary.append(dict(
            file=label, model="2d", Cf=0.0, operator=op,
            state=f"I2={state[0]}", V_opt=Vopt[-1, si], V_mod=Vm[-1, si],
            gap=G[-1, si], rel_gap_pct=rel[-1], max_gap_region_at_T=mg[-1],
            max_rel_gap_region_at_T_pct=mr[-1],
            max_gap_region_any_tau=float(mg.max()),
            modified_cells=int(changed.sum()),
            periods_touched=int(changed.any(1).sum()),
            worst_state_at_T=worst_state, worst_tau=worst_tau,
            viol_after=holes_after,
            tau_viol_after=int(_viol(ibar_new[None, :]).sum()),
            min_gap=float(G.min()),
            sanity_check=vchk, **totals))
        gaps.append(pd.DataFrame(dict(
            file=label, model="2d", Cf=0.0, operator=op,
            n=np.arange(0, N + 1), tau=np.round(taus, 6),
            modified_cells=nmod, V_opt_state=Vopt[:, si],
            gap_state=G[:, si], rel_gap_state_pct=rel,
            max_gap_region=mg, max_rel_gap_region_pct=mr)))
        cells_all.append(c)
        if plots:
            _plot_2d(plots, f"{label}_{op}", taus, ibar_level, ibar_new,
                     G[:, si], mg, state[0], cap_level,
                     f"2d Cf=0, {op}: gap at T = {G[-1, si]:.4f} "
                     f"({rel[-1]:.3f}%)")
    check.insert(0, "model", "2d"); check.insert(0, "file", label)
    return dict(summary=pd.DataFrame(summary), check=check,
                gaps=pd.concat(gaps, ignore_index=True),
                cells=pd.concat(cells_all, ignore_index=True))


# ======================================================================
# FINAL ANALYSIS
# ======================================================================
def _fmt(x, d=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{d}f}"


def report(summary: pd.DataFrame) -> str:
    """
    Plain-language conclusions per solved file:
      1. validity of the evaluation,
      2. whether the optimal table has a structural monotonicity violation,
      3. whether the operator establishes monotonicity (in I2, and what
         happens in tau, which is not enforced),
      4. how much worse the approximation is than the optimum,
      5. how the two operators compare.
    """
    COUNTS = ["max_levels_shifted", "I2_viol_tie_wait", "I2_viol_tie_dispatch", "I2_viol_both_rules",
              "tau_viol", "tie_cells", "not_upper_set_rows", "hole_cells",
              "periods_with_holes", "I2bar_increases_in_tau",
              "I2bar_decreases_in_tau", "modified_cells", "periods_touched",
              "viol_after", "tau_viol_after"]
    summary = summary.copy()
    for c in COUNTS:
        if c in summary:
            summary[c] = summary[c].map(
                lambda x: int(x) if pd.notna(x) else 0).astype(object)
    if "policy_match_pct" not in summary:
        summary["policy_match_pct"] = np.nan
    summary["_valid"] = np.where(
        summary["model"] == "3d",
        summary["sanity_check"].fillna(np.inf) <= 1e-6,
        (summary["policy_match_pct"].fillna(0) >= 100 - 1e-9)
        & ~(summary["sanity_check"].fillna(0) > 1e-6))
    L = ["# Monotone approximation report", ""]
    for fname, grp in summary.groupby("file", sort=False):
        r0 = grp.iloc[0]
        is3 = r0["model"] == "3d"
        L.append(f"## {fname}  ({'3-D model' if is3 else '2-D Cf=0 model'}, "
                 f"Cf = {r0['Cf']:g})")
        if is3 and r0["Cf"] == 0:
            L.append("- Note: this is the 3-D model at Cf = 0, not the note's "
                     "2-D model.")

        # 1. validity
        sc = r0["sanity_check"]
        if is3:
            ok = np.isfinite(sc) and sc <= 1e-6
            L.append(f"- Validity: re-evaluated optimal table vs solver V* "
                     f"differs by {_fmt(sc, 2) if np.isfinite(sc) else 'n/a'}"
                     f" -> {'OK' if ok else 'NOT OK, gaps below are unreliable'}.")
        else:
            m = r0["policy_match_pct"]
            ok = m >= 100 - 1e-9 and (not np.isfinite(sc) or sc <= 1e-6)
            L.append(f"- Validity: Bellman convention {r0['convention']} "
                     f"reproduces {m:.3f}% of the saved table"
                     + (f", value check {_fmt(sc, 2)}" if np.isfinite(sc)
                        else "")
                     + f" -> {'OK' if ok else 'NOT OK, gaps below are unreliable'}.")

        valid = bool(r0["_valid"])
        # 2. structure of the optimal table
        if is3:
            s_both, s_w, s_d = (r0["I2_viol_both_rules"],
                                r0["I2_viol_tie_wait"],
                                r0["I2_viol_tie_dispatch"])
            if s_w == 0 and s_d == 0:
                L.append("- Optimal table: b̄₁ is already non-increasing in I₂ "
                         "at every τ. There is no bump, so no approximation "
                         "is needed and both gaps are zero.")
            elif s_both == 0:
                L.append(f"- Optimal table: {max(s_w, s_d)} I₂-violations "
                         f"appear under one tie rule only, so they are tie "
                         f"artefacts, not a structural bump.")
            else:
                L.append(f"- Optimal table: {s_both} structural I₂-violations "
                         f"(present under both tie rules; {s_w} with ties to "
                         f"wait, {s_d} with ties to dispatch). The optimal "
                         f"policy is NOT monotone in I₂.")
            L.append(f"- Other diagnostics: τ-violations {r0['tau_viol']}, "
                     f"rows with a non-upper dispatch set (b₁ ≤ cap) "
                     f"{r0['not_upper_set_rows']}, exact-tie cells "
                     f"{r0['tie_cells']}.")
        else:
            if r0["hole_cells"] == 0:
                L.append("- Optimal table: the serve set is an upper set in "
                         "I₂ at every τ, i.e. a single threshold Ī₂(τ) "
                         "describes the policy. No approximation is needed.")
            else:
                L.append(f"- Optimal table: {r0['hole_cells']} hole cells in "
                         f"{r0['periods_with_holes']} periods, so a single "
                         f"threshold does NOT describe the policy.")
            L.append(f"- Ī₂(τ) increases {r0['I2bar_increases_in_tau']} times "
                     f"and decreases {r0['I2bar_decreases_in_tau']} times as τ "
                     f"grows; exact-tie cells {r0['tie_cells']}.")

        # 3 + 4. per operator
        rows = {r["operator"]: r for _, r in grp.iterrows()}
        for op, r in rows.items():
            name = "fill M⁻" if op == "fill" else "remove M⁺"
            if r["modified_cells"] == 0:
                L.append(f"- {name}: no cell changed; identical to the "
                         f"optimum.")
                continue
            mono = ("established" if r["viol_after"] == 0
                    else f"NOT fully established ({r['viol_after']} "
                         f"violations remain)")
            prop = "b̄₁ non-increasing in I₂" if is3 else "single threshold in I₂"
            tau_note = (f"; τ-direction (not enforced) has "
                        f"{r['tau_viol_after']} violations afterwards"
                        if r["tau_viol_after"] else
                        "; no τ-direction violations afterwards")
            neg = ("" if r["min_gap"] >= -1e-8 else
                   f" WARNING: a negative gap {r['min_gap']:.2e} appeared, "
                   f"which is impossible if the saved table is optimal.")
            L.append(
                f"- {name}: changed {r['modified_cells']} cells in "
                f"{r['periods_touched']} periods. Monotonicity ({prop}) is "
                f"{mono}{tau_note}.")
            if is3 and r.get("max_levels_shifted", 0):
                spread = int(r["max_levels_shifted"])
                L.append(f"  Reach: in the worst period b̄₁ was moved at "
                         f"{spread} I₂ levels, by up to "
                         f"{r['max_threshold_shift']:.0f} units.")
                n_lv = int(r["I2_max"]) if "I2_max" in r and pd.notna(
                    r["I2_max"]) else spread
                if r["max_threshold_shift"] > 1 or spread > n_lv / 2:
                    L.append("  CAUTION: this is not a one-unit bump. The "
                             "optimal threshold departs from monotonicity by "
                             "more than one unit or over most of the I₂ "
                             "range, so this operator rewrites a large part "
                             "of the policy. Its loss measures a structural "
                             "mismatch rather than the cost of smoothing a "
                             "local ridge.")
                else:
                    L.append("  The violation is a one-unit ridge, so this "
                             "is a local smoothing of the optimal policy.")
            L.append(
                f"  Cost: from {r['state']} at τ = T the expected cost rises "
                f"from {_fmt(r['V_opt'])} to {_fmt(r['V_mod'])}, "
                f"a loss of {_fmt(r['gap'])} ({_fmt(r['rel_gap_pct'], 3)}%). "
                f"The worst state at τ = T is {r['worst_state_at_T']} with a "
                f"loss of {_fmt(r['max_gap_region_at_T'])} "
                f"({_fmt(r['max_rel_gap_region_at_T_pct'], 3)}%); over all τ "
                f"the largest loss is {_fmt(r['max_gap_region_any_tau'])} at "
                f"τ = {r['worst_tau']:.3f}.{neg}")

        # 5. comparison
        if not valid:
            L.append("- Comparison: skipped, because the evaluation is not "
                     "valid for this file.")
        elif {"fill", "remove"} <= set(rows) and (
                rows["fill"]["modified_cells"] or rows["remove"]["modified_cells"]):
            gf, gr = rows["fill"]["gap"], rows["remove"]["gap"]
            better = (None if abs(gf - gr) <= 1e-6 else
                      "fill M⁻" if gf < gr else "remove M⁺")
            if better:
                L.append(f"- Comparison: {better} is cheaper at {r0['state']} "
                         f"({_fmt(min(gf, gr))} vs {_fmt(max(gf, gr))}). The "
                         f"two operators bracket the optimal table, so the "
                         f"cheaper one is the better monotone "
                         f"approximation of the two for this instance.")
            else:
                L.append("- Comparison: both operators cost the same at the "
                         "reported state.")
        L.append("")

    # cross-file view
    act = summary[(summary["modified_cells"] > 0) & summary["_valid"]]
    if len(summary["file"].unique()) > 1 and len(act):
        L.append("## Across files")
        for op, g in act.groupby("operator"):
            w = g.loc[g["rel_gap_pct"].idxmax()]
            L.append(f"- {op}: relative loss at the reported state ranges "
                     f"from {g['rel_gap_pct'].min():.3f}% to "
                     f"{g['rel_gap_pct'].max():.3f}% "
                     f"(largest in {w['file']}).")
        none = summary.groupby("file")["modified_cells"].sum()
        none = list(none[none == 0].index)
        if none:
            L.append(f"- Already monotone, no approximation needed: "
                     f"{', '.join(none)}.")
        bad = sorted(set(summary.loc[~summary["_valid"], "file"]))
        if bad:
            L.append(f"- Excluded as not valid: {', '.join(bad)}.")
        L.append("")
    return "\n".join(L)


# ======================================================================
# STAGE 2 DRIVER
# ======================================================================
def approx_file(path, ops=("fill", "remove"), state=(30, 2), b1_cap=40,
                top_margin=5, plots=None, verbose=True):
    model, params, policy, V = load(path)
    label = os.path.splitext(os.path.basename(path))[0]
    if verbose:
        print(f"{label}  [{model}]"
              + ("   (3-D model at Cf=0, not the note's 2-D model)"
                 if model == "3d" and params["Cf"] == 0 else ""))
    if model == "3d":
        res = approx_3d(params, policy, V, ops, state, b1_cap,
                        plots=plots, label=label, verbose=verbose)
    else:
        res = approx_2d(params, policy, V, ops, state, top_margin,
                        plots=plots, label=label, verbose=verbose)
    sm = res["summary"]
    keys = ["T", "N", "lam1", "lam2", "h", "cu", "pi1", "pi2", "c1", "c2",
            "v2", "I2_min", "I2_max", "b1_max", "I_min", "I_max"]
    for k in keys:
        if k in params and k not in sm:
            sm[k] = params[k]
    return res


def approx_files(paths, out="approx_out", fmt="csv", **kw):
    parts = [approx_file(f, **kw) for f in paths]
    res = {k: pd.concat([d[k] for d in parts], ignore_index=True)
           for k in ("summary", "check", "gaps", "cells")}
    write_tables(res, out, fmt)
    txt = report(res["summary"])
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print("\n" + txt)
    print(f"report written to {os.path.join(out, 'report.md')}")
    return res


def write_tables(res, out, fmt="csv"):
    os.makedirs(out, exist_ok=True)
    if fmt == "xlsx":
        try:
            path = os.path.join(out, "approximation_results.xlsx")
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                for k, df in res.items():
                    if len(df) > 1_000_000:
                        df.to_csv(os.path.join(out, f"{k}.csv"), index=False)
                        df = df.iloc[:1_000_000]
                    df.to_excel(xw, sheet_name=k, index=False)
            print(f"written {path}")
            return
        except ImportError:
            print("openpyxl not installed, writing CSV instead")
    for k, df in res.items():
        df.to_csv(os.path.join(out, f"{k}.csv"), index=False)
    print(f"written {', '.join(k + '.csv' for k in res)} to {out}/")


# ======================================================================
# PLOTS (optional)
# ======================================================================
def _heat(ax, arr, taus, xmax, title, vmax, cmap):
    im = ax.imshow(np.ma.masked_invalid(np.where(np.isfinite(arr), arr,
                                                 np.nan)),
                   aspect="auto", origin="lower", cmap=cmap, vmin=1,
                   vmax=vmax, interpolation="nearest",
                   extent=[0.5, xmax + 0.5, taus[1], taus[-1]])
    ax.set_title(title + ", grey = +∞", fontsize=10)
    ax.set_ylabel("τ")
    return im


def _plot(folder, name, taus, bold, bnew, cx, cy, g_s, mg, slabel, rlabel,
          zlabel, xlabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#C8C8C8")
    fin = bold[np.isfinite(bold)]
    vmax = fin.max() if fin.size else 1
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    _heat(ax[0], bold, taus, bold.shape[1], f"optimal {zlabel}", vmax, cmap)
    im = _heat(ax[1], bnew, taus, bold.shape[1], f"modified {zlabel}",
               vmax, cmap)
    if len(cx):
        ax[0].scatter(cx, cy, s=3, color="crimson", label="modified")
        ax[0].legend(fontsize=8, loc="upper right")
    for a in ax[:2]:
        a.set_xlabel(xlabel)
    fig.colorbar(im, ax=ax[:2], label=zlabel)
    ax[2].plot(taus, g_s, lw=1.8, label=f"gap at {slabel}")
    ax[2].plot(taus, mg, lw=1.2, ls="--", label=f"max gap, {rlabel}")
    ax[2].set_xlabel("τ"); ax[2].set_ylabel("V_mod − V*")
    ax[2].grid(True, alpha=0.3); ax[2].legend(fontsize=8)
    ax[2].set_title(title, fontsize=10)
    fig.savefig(os.path.join(folder, f"{name}.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def _plot_2d(folder, name, taus, iold, inew, g_s, mg, I0, cap, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].step(taus[1:], iold, where="post", lw=1.8, label="optimal Ī₂")
    ax[0].step(taus[1:], inew, where="post", lw=1.2, ls="--",
               label="modified Ī₂")
    ax[0].set_xlabel("τ"); ax[0].set_ylabel("smallest serving I₂")
    ax[0].grid(True, alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(taus, g_s, lw=1.8, label=f"gap at I₂={I0}")
    ax[1].plot(taus, mg, lw=1.2, ls="--", label=f"max gap, 1 ≤ I₂ ≤ {cap}")
    ax[1].set_xlabel("τ"); ax[1].set_ylabel("V_mod − V*")
    ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set_title(title, fontsize=10)
    fig.savefig(os.path.join(folder, f"{name}.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# CONFIGURATION DRIVER
# ======================================================================
def _as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _parse_value(key, text):
    if isinstance(text, str) and text.lower() in ("none", "auto", "null"):
        return None
    return int(text) if key in INT_KEYS else float(text)


def build_config(argv):
    """CONFIG  <-  JSON file (--config)  <-  command-line flags."""
    ap = argparse.ArgumentParser(
        description="Solve once, then check, modify and evaluate monotone "
                    "approximations of the transshipment SDP. Without "
                    "arguments the CONFIG block at the top of the file is "
                    "used; every key can be overridden here.")
    ap.add_argument("stage", nargs="?", choices=["run", "solve", "approx"])
    ap.add_argument("--config", help="JSON file with any subset of CONFIG")
    ap.add_argument("--model", choices=["3d", "2d", "both"])
    for k in sorted(set(SWEEP_3D) | set(SWEEP_2D)):
        ap.add_argument(f"--{k}", nargs="+", metavar="V",
                        help="one value or several (swept); 'auto' for bounds"
                        if k in ("I2_min", "I2_max", "b1_max") else
                        "one value or several (swept)")
    ap.add_argument("--files", nargs="+")
    ap.add_argument("--ops", nargs="+", choices=["fill", "remove"])
    ap.add_argument("--state", type=int, nargs=2, metavar=("I2", "B1"))
    ap.add_argument("--b1_cap", type=int)
    ap.add_argument("--top_margin", type=int)
    ap.add_argument("--save_dir")
    ap.add_argument("--out")
    ap.add_argument("--format", choices=["csv", "xlsx"])
    ap.add_argument("--plots", help="folder, or 'none'")
    a = ap.parse_args(argv)

    cfg = dict(CONFIG)
    if a.config:
        with open(a.config, encoding="utf-8") as fh:
            extra = json.load(fh)
        unknown = set(extra) - set(CONFIG)
        if unknown:
            raise SystemExit(f"unknown keys in {a.config}: {sorted(unknown)}")
        cfg.update(extra)
    for k, v in vars(a).items():
        if k == "config" or v is None:
            continue
        if k in SWEEP_3D or k in SWEEP_2D:
            v = [_parse_value(k, x) for x in v]
            v = v[0] if len(v) == 1 else v
        if k == "plots" and str(v).lower() == "none":
            v = None
        cfg[k] = v
    return cfg


def _grid(cfg, keys):
    import itertools
    vals = [_as_list(cfg[k]) for k in keys]
    seen, out = set(), []
    for combo in itertools.product(*vals):
        d = dict(zip(keys, combo))
        sig = json.dumps(d, sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            out.append(d)
    return out


def solve_from_config(cfg):
    if (_as_list(cfg["c1"]), _as_list(cfg["c2"]), _as_list(cfg["v2"])) \
            != ([0.0], [0.0], [0.0]):
        print("note: nonzero terminal costs somewhere in the sweep; those "
              "instances are not comparable with the note's figures")
    files = []
    if cfg["model"] in ("3d", "both"):
        grid = _grid(cfg, SWEEP_3D)
        print(f"3d model: {len(grid)} instance(s)")
        for g in grid:
            Cf = g.pop("Cf")
            files.append(solve_3d(Base(**g), Cf, cfg["save_dir"]))
    if cfg["model"] in ("2d", "both"):
        grid = _grid(cfg, SWEEP_2D)
        print(f"2d model: {len(grid)} instance(s)")
        for g in grid:
            n2 = g.pop("N2d")
            files.append(solve_2d(Base(**g), n2, cfg["save_dir"]))
    return files


def approx_from_config(cfg, files):
    return approx_files(files, out=cfg["out"], fmt=cfg["format"],
                        ops=tuple(cfg["ops"]), state=tuple(cfg["state"]),
                        b1_cap=int(cfg["b1_cap"]),
                        top_margin=int(cfg["top_margin"]),
                        plots=cfg["plots"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = build_config(argv)
    if not argv:
        print("no command-line arguments: using the CONFIG block")
    print(f"working directory: {os.getcwd()}")
    shown = {k: v for k, v in cfg.items()
             if not (k == "files" and cfg["stage"] != "approx")}
    print("configuration: " + json.dumps(shown, ensure_ascii=False))

    if cfg["stage"] == "solve":
        solve_from_config(cfg)
        return
    if cfg["stage"] == "approx":
        files = sorted({f for pat in _as_list(cfg["files"])
                        for f in glob.glob(pat)})
        if not files:
            raise SystemExit(f"no .npz files matched {cfg['files']}")
    else:
        files = solve_from_config(cfg)
    os.makedirs(cfg["out"], exist_ok=True)
    with open(os.path.join(cfg["out"], "config_used.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dict(cfg, files_analysed=files), fh, indent=2,
                  ensure_ascii=False)
    approx_from_config(cfg, files)


if __name__ == "__main__":
    main()