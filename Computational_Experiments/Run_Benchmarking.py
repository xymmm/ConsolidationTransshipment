"""
Run_benchmarks.py — one runner for the whole benchmark study.

Reproduces every table reported in the computational experiments:

    python Run_benchmarks.py main      DP vs best Q vs best T, Cf=8, pi1=pi2=6
    python Run_benchmarks.py asym      asymmetric penalties at Cf=8
    python Run_benchmarks.py cfsweep   gap as a function of Cf
    python Run_benchmarks.py cf0asym   asymmetric penalties at Cf=0
    python Run_benchmarks.py all       everything

Everything runs on the SAME at-most-one-event chain that solver.py solves and
on the SAME common random numbers, so a gap is attributable to the policy
alone. The DP is wrapped as a Policy so that it is simulated through exactly
the same code path as the benchmarks; its simulated cost is printed next to
V^N as a self-check.

Place in Computational_Experiments/ next to base.py, policy_Q.py, Policy_T.py
and simulator.py. solver.py is located automatically in the sibling folder.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

# ── locate and load solver.py from the sibling folder ────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_solver(start, max_levels=3):
    d = start
    for _ in range(max_levels + 1):
        cand = os.path.join(d, "solver.py")
        if os.path.isfile(cand):
            return cand
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            entries = []
        for name in entries:
            if name.startswith(".") or name == "__pycache__":
                continue
            sub = os.path.join(d, name)
            if os.path.isdir(sub):
                cand = os.path.join(sub, "solver.py")
                if os.path.isfile(cand):
                    return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise ImportError("solver.py not found; set SOLVER_PATH.")


_SOLVER_PATH = os.environ.get("SOLVER_PATH") or _find_solver(_HERE)
if "solver" in sys.modules:
    solver = sys.modules["solver"]
else:
    _spec = importlib.util.spec_from_file_location("solver", _SOLVER_PATH)
    solver = importlib.util.module_from_spec(_spec)
    sys.modules["solver"] = solver
    _spec.loader.exec_module(solver)

Params = solver.Params
TransshipmentDP = solver.TransshipmentDP

from base import Policy                                   # noqa: E402
from policy_Q import QPolicy                              # noqa: E402
from Policy_T import TPolicy                              # noqa: E402
from simulator import make_uniforms, simulate, summarise   # noqa: E402


# ── study settings ───────────────────────────────────────────────────
N_PERIODS = 250
N_REPS    = 20000
SEED      = 20260810

INITS     = [(15, 0), (20, 0), (25, 0)]
Q_GRID    = list(range(1, 21))
M_GRID    = list(range(1, 41))          # Delta = T / m
CF_LIST   = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def instance(Cf=8.0, pi1=6.0, pi2=6.0, N=N_PERIODS):
    """Base instance of the general-Cf note, with Cf and the penalties free.

    Terminal costs are zero so that the simulated cost matches the closed
    forms of the note. b1_max and I2_min are wide enough that truncation
    does not bind: b1_max >> lam1*T and I2_min << -lam2*T.
    """
    return Params(T=5.0, N=N, lam1=5.0, lam2=3.0,
                  h=1.0, Cf=Cf, cu=1.0, pi1=pi1, pi2=pi2,
                  c1=0.0, c2=0.0, v2=0.0,
                  I2_max=35, I2_min=-30, b1_max=45)


# ── DP as a Policy, so it goes through the same simulator ────────────
class DPPolicy(Policy):
    """Wraps a solved TransshipmentDP so it can be simulated like a benchmark."""

    def __init__(self, dp):
        self.pol    = dp.policy                # (N+1, nI2, nb1)
        self.I2_min = dp.p.I2_min
        self.I2_max = dp.p.I2_max
        self.b1_max = dp.p.b1_max
        self.name   = "DP"

    def decide(self, step, n_remaining, I2, b1):
        ii = np.clip(I2, self.I2_min, self.I2_max) - self.I2_min
        jj = np.clip(b1, 0, self.b1_max)
        return self.pol[n_remaining, ii, jj].astype(np.int64)


# ── helpers ──────────────────────────────────────────────────────────
def solve_dp(p, verbose=False):
    dp = TransshipmentDP(p)
    dp.solve(store_V=False, verbose=verbose)
    return dp


def evaluate(pol, p, U, I2_0, b1_0):
    return summarise(simulate(pol, p, U, I2_0, b1_0))


def best_q(p, U, I2_0, b1_0, grid=Q_GRID):
    """Tune the lot size; return (Q*, summary)."""
    out = [(Q, evaluate(QPolicy(Q), p, U, I2_0, b1_0)) for Q in grid]
    return min(out, key=lambda z: z[1]["total"]["mean"])


def best_t(p, U, I2_0, b1_0, grid=M_GRID):
    """Tune the review interval; return (Delta*, summary)."""
    out = [(p.T / m, evaluate(TPolicy(p.T / m, p), p, U, I2_0, b1_0))
           for m in grid]
    return min(out, key=lambda z: z[1]["total"]["mean"])


def _gap(bench, dp_mean):
    return 100.0 * (bench["total"]["mean"] - dp_mean) / dp_mean


def compare(p, label):
    """DP vs best Q vs best T at every starting state. Returns list of rows."""
    print("\n" + "=" * 96)
    print(label)
    print("  " + p.summary())
    print("=" * 96)

    dp = solve_dp(p)
    U  = make_uniforms(N_REPS, p.N, SEED)
    dpp = DPPolicy(dp)

    print("  %-9s | %-22s | %-26s | %-26s" %
          ("start", "DP (sim / V^N)", "Q-policy", "T-policy"))
    print("  " + "-" * 92)

    rows = []
    for (I2_0, b1_0) in INITS:
        d  = evaluate(dpp, p, U, I2_0, b1_0)
        vn = dp.get_value(p.N, I2_0, b1_0)
        dm = d["total"]["mean"]
        Qs, q = best_q(p, U, I2_0, b1_0)
        Ds, t = best_t(p, U, I2_0, b1_0)

        print("  (%2d,%d)    | %8.2f / %8.2f    | Q*=%2d %8.2f  %+5.1f%%  "
              "| d*=%.3f %8.2f  %+5.1f%%"
              % (I2_0, b1_0, dm, vn,
                 Qs, q["total"]["mean"], _gap(q, dm),
                 Ds, t["total"]["mean"], _gap(t, dm)))

        rows.append(dict(init=(I2_0, b1_0), dp=dm, vN=vn,
                         Q_star=Qs, Q_cost=q["total"]["mean"], Q_gap=_gap(q, dm),
                         d_star=Ds, T_cost=t["total"]["mean"], T_gap=_gap(t, dm),
                         dp_ci=d["total"]["ci95"],
                         dp_disp=d["n_dispatch"]["mean"],
                         Q_disp=q["n_dispatch"]["mean"],
                         T_disp=t["n_dispatch"]["mean"]))
    return rows


# ── the four reported experiments ────────────────────────────────────
def run_main():
    return compare(instance(Cf=8.0, pi1=6.0, pi2=6.0),
                   "EQUAL PENALTIES   Cf=8, pi1=pi2=6")


def run_asym():
    a = compare(instance(Cf=8.0, pi1=3.0, pi2=9.0),
                "ASYMMETRIC   Cf=8, pi1=3 < pi2=9")
    b = compare(instance(Cf=8.0, pi1=9.0, pi2=3.0),
                "ASYMMETRIC   Cf=8, pi1=9 > pi2=3")
    return a, b


def run_cf0_asym():
    """Cf=0 with unequal penalties: does the Q-gap survive without a fixed cost?"""
    a = compare(instance(Cf=0.0, pi1=3.0, pi2=9.0),
                "Cf=0   pi1=3 < pi2=9")
    b = compare(instance(Cf=0.0, pi1=9.0, pi2=3.0),
                "Cf=0   pi1=9 > pi2=3")
    c = compare(instance(Cf=0.0, pi1=6.0, pi2=6.0),
                "Cf=0   pi1=pi2=6   (reference)")
    return a, b, c


def run_cf_sweep(I2_0=20, b1_0=0, cf_list=CF_LIST, pi1=6.0, pi2=6.0):
    print("\n" + "=" * 96)
    print("FIXED-COST SWEEP   start (%d,%d), pi1=%g, pi2=%g" % (I2_0, b1_0, pi1, pi2))
    print("=" * 96)
    print("  %4s | %10s | %4s %9s %8s | %7s %9s %8s"
          % ("Cf", "DP (V^N)", "Q*", "Q cost", "Q gap", "Delta*", "T cost", "T gap"))
    print("  " + "-" * 88)

    rows = []
    for Cf in cf_list:
        p  = instance(Cf=Cf, pi1=pi1, pi2=pi2)
        dp = solve_dp(p)
        U  = make_uniforms(N_REPS, p.N, SEED)
        dm = evaluate(DPPolicy(dp), p, U, I2_0, b1_0)["total"]["mean"]
        vn = dp.get_value(p.N, I2_0, b1_0)
        Qs, q = best_q(p, U, I2_0, b1_0)
        Ds, t = best_t(p, U, I2_0, b1_0)

        print("  %4g | %10.2f | %4d %9.2f %7.1f%% | %7.3f %9.2f %7.1f%%"
              % (Cf, vn, Qs, q["total"]["mean"], _gap(q, dm),
                 Ds, t["total"]["mean"], _gap(t, dm)))
        rows.append(dict(Cf=Cf, vN=vn, dp=dm, Q_star=Qs, Q_gap=_gap(q, dm),
                         d_star=Ds, T_gap=_gap(t, dm)))
    return rows


EXPERIMENTS = {
    "main":    run_main,
    "asym":    run_asym,
    "cfsweep": run_cf_sweep,
    "cf0asym": run_cf0_asym,
}


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("solver.py : %s" % _SOLVER_PATH)
    print("settings  : reps=%d  seed=%d  N=%d" % (N_REPS, SEED, N_PERIODS))
    if what == "all":
        run_main(); run_asym(); run_cf_sweep(); run_cf0_asym()
    elif what in EXPERIMENTS:
        EXPERIMENTS[what]()
    else:
        print("unknown experiment %r; choose from %s or 'all'"
              % (what, list(EXPERIMENTS)))


if __name__ == "__main__":
    main()