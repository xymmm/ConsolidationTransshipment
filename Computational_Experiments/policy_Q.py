"""
policy_Q.py — Quantity-triggered benchmark (fixed lot Q).

Rule (Section \\ref{sec:q-policy} of the paper):
  * Monitor b1.  Trigger when b1 >= Q and I2 > 0.
  * Dispatch q = min(Q, I2)               [rule A1: partial dispatch]
  * No forced dispatch at the end of the horizon.

Under unit Poisson arrivals the trigger normally fires with b1 == Q.
The >= comparison also covers the residual case after a partial dispatch,
where b1 > 0 remains but I2 == 0, in which case the trigger can never
fire again and the policy is de facto retired — the logic below handles
this automatically because the I2 > 0 condition fails.
"""

import numpy as np

from base import Policy


class QPolicy(Policy):

    def __init__(self, Q):
        assert Q >= 1
        self.Q = int(Q)
        self.name = "Q-policy(Q=%d)" % self.Q

    def decide(self, step, n_remaining, I2, b1):
        trigger = (b1 >= self.Q) & (I2 > 0)
        q = np.where(trigger, np.minimum(self.Q, I2), 0)
        return q.astype(np.int64)