"""
Config.py — Single source of truth for the computational study.

Project layout
--------------
    ConsolidationTransshipment/
        Clean-up System/
            solver.py                 <- the SDP solver (SIBLING folder)
            app.py, drawer.py, ...
        Computational_Experiments/
            Config.py                 <- this file
            base.py
            policy_Q.py
            Policy_T.py
            Run_q_policy.py
            Run_t_policy.py
            simulator.py
            Policy_main.py

solver.py sits in a sibling folder whose name contains a space and a
hyphen, so it cannot be imported as a package.  The loader below walks up
from this file and, at every level, checks both that level and its direct
subfolders for solver.py.  The sibling folder is therefore found at the
first step up.  The module is then loaded directly from its path via
importlib, independently of sys.path, of the working directory, and of
any PyCharm setting.

Override: if the search ever fails, set the environment variable
SOLVER_PATH to the absolute path of solver.py, or edit _EXPLICIT_PATH
below.

If PyCharm underlines `solver` anywhere, right-click the "Clean-up System"
folder and choose  Mark Directory as -> Sources Root.  That is an IDE-only
setting and changes nothing at runtime.

The cost functional itself is NOT duplicated here.  It lives in
solver.TransshipmentDP.g / .terminal and is mirrored by simulator.py.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Optional hard override: put an absolute path here to skip the search.
_EXPLICIT_PATH = os.environ.get("SOLVER_PATH", "")


def _find_solver_path(start_dir, max_levels=3):
    """
    Locate solver.py.

    At each level going up from start_dir, check:
      1. <level>/solver.py
      2. <level>/*/solver.py        (direct subfolders, i.e. siblings of
                                     start_dir when level is its parent)
    Returns (path_or_None, list_of_paths_tried).
    """
    tried = []
    d = start_dir
    for _ in range(max_levels + 1):
        candidate = os.path.join(d, "solver.py")
        tried.append(candidate)
        if os.path.isfile(candidate):
            return candidate, tried

        try:
            entries = sorted(os.listdir(d))
        except OSError:
            entries = []
        for name in entries:
            if name.startswith(".") or name == "__pycache__":
                continue
            sub = os.path.join(d, name)
            if not os.path.isdir(sub):
                continue
            candidate = os.path.join(sub, "solver.py")
            tried.append(candidate)
            if os.path.isfile(candidate):
                return candidate, tried

        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None, tried


if _EXPLICIT_PATH:
    if not os.path.isfile(_EXPLICIT_PATH):
        raise ImportError("SOLVER_PATH is set but does not exist: %s" % _EXPLICIT_PATH)
    _SOLVER_PATH, _TRIED = _EXPLICIT_PATH, [_EXPLICIT_PATH]
else:
    _SOLVER_PATH, _TRIED = _find_solver_path(_HERE)

if _SOLVER_PATH is None:
    raise ImportError(
        "Could not locate solver.py.  Looked in:\n  " + "\n  ".join(_TRIED)
        + "\n\nSet the SOLVER_PATH environment variable, or edit "
          "_EXPLICIT_PATH at the top of Config.py."
    )

_SOLVER_DIR = os.path.dirname(_SOLVER_PATH)
if _SOLVER_DIR not in sys.path:
    sys.path.insert(0, _SOLVER_DIR)

# -- load solver.py directly from its path ----------------------------
if "solver" in sys.modules:
    solver = sys.modules["solver"]
else:
    _spec = importlib.util.spec_from_file_location("solver", _SOLVER_PATH)
    solver = importlib.util.module_from_spec(_spec)
    sys.modules["solver"] = solver
    _spec.loader.exec_module(solver)

Params = solver.Params
TransshipmentDP = solver.TransshipmentDP

SOLVER_PATH = _SOLVER_PATH          # exposed for logging / debugging


# -- Instance (must match the instance solved by the SDP) -------------
PARAMS = Params(
    T=2.0, N=200,
    lam1=8.0, lam2=5.0,
    h=0.1, Cf=20.0, cu=1.0,
    pi1=10.0, pi2=10.0,
    c1=10.0, c2=10.0, v2=1.0,
    I2_max=35, I2_min=-10, b1_max=45,
)

# -- Initial state ----------------------------------------------------
I2_INIT = 15
B1_INIT = 0

# -- Simulation settings ----------------------------------------------
N_REPS = 5000          # replications per parameter value
SEED   = 20260722      # CRN seed: same uniforms for every policy & parameter

# -- Parameter grids --------------------------------------------------
# Q-policy: dispatch lot sizes
Q_GRID = list(range(1, 21))

# T-policy: review intervals Delta = T/m  (m = number of intervals).
# m = 1 gives no interior review epoch, i.e. the never-dispatch baseline.
M_GRID     = list(range(1, 41))
DELTA_GRID = [PARAMS.T / m for m in M_GRID]

# -- Output locations (absolute, so scripts run from any working dir) --
RESULTS_DIR    = os.path.join(_HERE, "results")
Q_RESULTS_JSON = os.path.join(RESULTS_DIR, "q_policy.json")
T_RESULTS_JSON = os.path.join(RESULTS_DIR, "t_policy.json")


# -- self-test --------------------------------------------------------
if __name__ == "__main__":
    print("solver.py loaded from : %s" % SOLVER_PATH)
    print("instance              : %s" % PARAMS.summary())
    PARAMS.validate()
    print("validate()            : OK")
    print("results dir           : %s" % RESULTS_DIR)