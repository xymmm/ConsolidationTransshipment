"""
Policy_T.py — Time-triggered benchmark (review interval Delta).

Rule (Section \\ref{sec:t-policy} of the paper):
  * Review epochs t_k = k·Delta for k = 1, 2, …, with k·Delta < T strictly.
    The epoch at the terminal time is excluded by definition.
  * At each review epoch dispatch q = min(b1, I2) whenever both positive.
    Unconditional: no cost comparison is embedded.

Review epochs are mapped to the nearest period-start grid point of the
at-most-one-event discretisation: step s = round(k·Delta / dt), and the
dispatch executes at the start of that period (time s·dt).  Steps are
deduplicated and restricted to 1 <= s <= N-1, which enforces the strict
exclusion of the terminal time.
"""

import numpy as np

from base import Policy


class TPolicy(Policy):

    def __init__(self, delta, params):
        assert delta > 0
        self.delta = float(delta)
        self.name = "T-policy(Delta=%.4g)" % self.delta

        tol = 1e-9
        steps = set()
        k = 1
        while k * self.delta < params.T - tol:
            s = int(round(k * self.delta / params.dt))
            if 1 <= s <= params.N - 1:
                steps.add(s)
            k += 1
        self.review_steps = frozenset(steps)

    def decide(self, step, n_remaining, I2, b1):
        if step not in self.review_steps:
            return np.zeros_like(b1)
        q = np.minimum(np.maximum(b1, 0), np.maximum(I2, 0))
        return q.astype(np.int64)