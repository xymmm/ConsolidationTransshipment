"""
base.py — Policy interface for the discrete-chain simulator.

A Policy maps the current period and state to a dispatch quantity.  The
decide() method is VECTORISED across replications: I2 and b1 are integer
numpy arrays of shape (n_reps,) and the return value is an integer numpy
array of the same shape.  The simulator clips the returned quantities to
the feasible range min(I2^+, b1^+), so a policy may return its nominal
quantity without re-implementing feasibility.

Concrete policies in this project:
    policy_Q.QPolicy    quantity-triggered benchmark
    Policy_T.TPolicy    time-triggered benchmark
"""

from abc import ABC, abstractmethod


class Policy(ABC):
    """Vectorised dispatch policy."""

    name = "policy"

    @abstractmethod
    def decide(self, step, n_remaining, I2, b1):
        """
        Dispatch decision at the START of a period.

        Parameters
        ----------
        step        : forward period index, 0 … N-1  (period start time = step·dt)
        n_remaining : periods remaining = N - step   (the DP's n index)
        I2, b1      : integer numpy arrays, shape (n_reps,)

        Returns
        -------
        q : integer numpy array, shape (n_reps,), nominal dispatch quantities.
        """
        raise NotImplementedError

    def label(self):
        """Short label for results files and plots."""
        return self.name