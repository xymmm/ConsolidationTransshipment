"""
transship_core.py — vectorised building blocks on the solver.py chain
=====================================================================

Everything here uses exactly the discretisation and cost convention of
solver.py: at-most-one-event Poisson chain with step dt = T/N, dispatch at
the start of a period, flow cost charged on the post-dispatch state, the
same clipping at the state-space bounds and the same tie rule (smallest q
among the minimisers, so ties go to waiting).  `Chain.solve()` reproduces
`solver.TransshipmentDP.solve()` bit for bit and is about two orders of
magnitude faster.

Index convention: n = number of remaining periods, tau = n*dt, time
t = T - n*dt.  The start of the horizon is n = N.

Contents
    Chain                      grids, one-step operators, exact DP
    Chain.solve_schedule       open-loop epochs, state-dependent quantity
    Chain.forward              exact distribution of the number of
                               dispatches and expected cost of any table
"""

import warnings
import numpy as np
from solver import Params


def _dot(a, b):
    """
    Inner product computed elementwise, without BLAS. On macOS, numpy linked
    to Apple Accelerate can raise spurious 'divide by zero / overflow /
    invalid value encountered in matmul' warnings for an ordinary '@' on
    finite vectors; the result is correct but the console is misleading.
    An elementwise sum never calls BLAS, so the warnings cannot appear.
    """
    return float(np.sum(np.asarray(a, float) * np.asarray(b, float)))


def warn_if_cf0(p, where):
    """
    Chain, like solver.py, is the 3-D model V(I2, b1, tau) for every Cf.
    At Cf = 0 it is NOT the 2-D switching model of the 1 August note
    (solver_cf0_2d.py), and questions about dispatch epochs or the number of
    dispatches are not the same questions in that model.
    """
    if float(p.Cf) == 0.0:
        warnings.warn(
            f"{where}: Cf = 0. This computes the 3-D model of solver.py at "
            "Cf = 0, not the 2-D switching model of the Cf = 0 note "
            "(solver_cf0_2d.py). Results are not comparable with that note.",
            stacklevel=3)


class Chain:
    def __init__(self, p: Params):
        p.validate()
        self.p = p
        self.I2v = np.arange(p.I2_min, p.I2_max + 1)
        self.b1v = np.arange(0, p.b1_max + 1)
        self.shape = (len(self.I2v), len(self.b1v))
        self.qmax = max(0, min(p.I2_max, p.b1_max))
        I2g = self.I2v[:, None]
        b1g = self.b1v[None, :]
        sh = self.shape
        cI = lambda x: np.clip(x, p.I2_min, p.I2_max) - p.I2_min
        cB = lambda x: np.clip(x, 0, p.b1_max)
        self.ii0, self.jj0, self.jj1, self.ii2 = [], [], [], []
        self.g, self.feas = [], []
        for q in range(self.qmax + 1):
            I2a = np.broadcast_to(I2g - q, sh)
            b1a = np.broadcast_to(b1g - q, sh)
            g = p.cu * q + p.dt * (p.h * np.maximum(0, I2a) + p.pi1 * b1a
                                   + p.pi2 * np.maximum(0, -I2a))
            if q > 0:
                g = g + p.Cf
                feas = (I2g >= q) & (b1g >= q)
            else:
                feas = np.ones(sh, bool)
            self.g.append(np.asarray(g, float))
            self.feas.append(np.broadcast_to(feas, sh))
            self.ii0.append(cI(I2a)); self.jj0.append(cB(b1a))
            self.jj1.append(cB(b1a + 1)); self.ii2.append(cI(I2a - 1))
        # largest q that is feasible anywhere on the grid
        self.qtop = self.qmax

    # ── one-step operators ───────────────────────────────────────────
    def Q(self, q, V):
        """Q-value of action q against next-stage values V (solver.py order)."""
        return self.g[q] + (self.p.p0 * V[self.ii0[q], self.jj0[q]]
                            + self.p.p1 * V[self.ii0[q], self.jj1[q]]
                            + self.p.p2 * V[self.ii2[q], self.jj0[q]])

    def best_dispatch(self, V):
        """min_{q>=1} Q(q) and the smallest minimiser; +inf / 0 if infeasible."""
        best = np.full(self.shape, np.inf)
        bq = np.zeros(self.shape, np.int16)
        for q in range(1, self.qtop + 1):
            f = self.feas[q]
            if not f.any():
                break
            val = np.where(f, self.Q(q, V), np.inf)
            better = val < best
            best = np.where(better, val, best)
            bq = np.where(better, q, bq)
        return best, bq

    def terminal(self):
        p = self.p
        I2 = self.I2v[:, None]; b1 = self.b1v[None, :]
        return (p.c1 * b1 + p.c2 * np.maximum(0, -I2)
                - p.v2 * np.maximum(0, I2)) * np.ones(self.shape)

    # ── exact DP (replica of solver.py) ──────────────────────────────
    def solve(self):
        N = self.p.N
        V = self.terminal()
        V_all = np.empty((N + 1,) + self.shape)
        pol = np.zeros((N + 1,) + self.shape, np.int16)
        V_all[0] = V
        for n in range(1, N + 1):
            w = self.Q(0, V)
            d, bq = self.best_dispatch(V)
            disp = d < w
            V = np.where(disp, d, w)
            pol[n] = np.where(disp, bq, 0)
            V_all[n] = V
        return V_all, pol

    def solve_wait_only(self):
        """Never dispatch: V^(0)."""
        N = self.p.N
        V = self.terminal()
        V_all = np.empty((N + 1,) + self.shape)
        V_all[0] = V
        for n in range(1, N + 1):
            V = self.Q(0, V)
            V_all[n] = V
        return V_all

    # ── open loop epochs ─────────────────────────────────────────────
    def n_of_time(self, t):
        """Period index (remaining periods) of a dispatch epoch at time t."""
        p = self.p
        return int(np.clip(round((p.T - t) / p.dt), 1, p.N))

    def solve_schedule(self, epochs_n, rule="optimal"):
        """
        Dispatch is allowed only at the start of the periods in epochs_n
        (remaining-period indices).  rule = 'optimal': at an epoch choose
        q in {0,...,min(I2,b1)} minimising cost given the remaining epochs
        (state-dependent quantity, may skip).  rule = 'clear': dispatch
        q = min(I2, b1) whenever positive (the T-policy quantity rule).
        Returns (V_all, pol).
        """
        N = self.p.N
        E = set(int(e) for e in epochs_n)
        V = self.terminal()
        V_all = np.empty((N + 1,) + self.shape)
        pol = np.zeros((N + 1,) + self.shape, np.int16)
        V_all[0] = V
        for n in range(1, N + 1):
            w = self.Q(0, V)
            if n in E:
                if rule == "optimal":
                    d, bq = self.best_dispatch(V)
                    disp = d < w
                    Vn = np.where(disp, d, w)
                    pol[n] = np.where(disp, bq, 0)
                else:
                    qc = np.clip(np.minimum(self.I2v[:, None], self.b1v[None, :]),
                                 0, None).astype(np.int16)
                    Vn = w.copy()
                    for q in range(1, self.qtop + 1):
                        m = qc == q
                        if m.any():
                            Vn[m] = self.Q(q, V)[m]
                    pol[n] = qc
                V = Vn
            else:
                V = w
            V_all[n] = V
        return V_all, pol

    # ── exact forward distribution under a policy table ──────────────
    def forward(self, pol, I2_0, b1_0, cmax=6):
        """
        Exact law of the number of dispatches (capped at cmax) and the exact
        expected cost from (I2_0, b1_0) at n = N under policy table pol[n].
        Also returns the probability of a dispatch at each period.
        """
        p = self.p
        N = p.N
        nI, nB = self.shape
        size = nI * nB
        mass = np.zeros((cmax + 1, size))
        i0 = int(np.clip(I2_0, p.I2_min, p.I2_max) - p.I2_min)
        mass[0, i0 * nB + int(np.clip(b1_0, 0, p.b1_max))] = 1.0
        cost = 0.0
        disp_prob = np.zeros(N + 1)
        ii_flat = {}
        for n in range(N, 0, -1):
            A = np.asarray(pol[n]).ravel()
            new = np.zeros_like(mass)
            for q in np.unique(A):
                q = int(q)
                cells = np.nonzero(A == q)[0]
                gq = self.g[q].ravel()[cells]
                d0 = self.ii0[q].ravel()[cells] * nB + self.jj0[q].ravel()[cells]
                d1 = self.ii0[q].ravel()[cells] * nB + self.jj1[q].ravel()[cells]
                d2 = self.ii2[q].ravel()[cells] * nB + self.jj0[q].ravel()[cells]
                for c in range(cmax + 1):
                    m = mass[c, cells]
                    if not m.any():
                        continue
                    cost += _dot(m, gq)
                    c2 = min(c + 1, cmax) if q > 0 else c
                    if q > 0:
                        disp_prob[n] += m.sum()
                    new[c2] += (np.bincount(d0, m * p.p0, size)
                                + np.bincount(d1, m * p.p1, size)
                                + np.bincount(d2, m * p.p2, size))
            mass = new
        Vterm = self.terminal().ravel()
        cost += _dot(mass.sum(axis=0), Vterm)
        if not np.isfinite(cost) or not np.all(np.isfinite(mass)):
            raise FloatingPointError("forward(): non-finite cost or mass")
        return dict(count_pmf=mass.sum(axis=1), cost=cost,
                    disp_prob=disp_prob, final_mass=mass)

    def wait_distributions(self, I2_0, b1_0):
        """
        Distribution of the state at the start of every period under
        waiting only, and the expected flow cost incurred before each period.
        P[n] is the law at the start of period n; R[n] is the expected cost
        of periods N..n+1.
        """
        p = self.p
        N = p.N
        nI, nB = self.shape
        size = nI * nB
        x = np.zeros(size)
        i0 = int(np.clip(I2_0, p.I2_min, p.I2_max) - p.I2_min)
        x[i0 * nB + int(np.clip(b1_0, 0, p.b1_max))] = 1.0
        g0 = self.g[0].ravel()
        d0 = (self.ii0[0] * nB + self.jj0[0]).ravel()
        d1 = (self.ii0[0] * nB + self.jj1[0]).ravel()
        d2 = (self.ii2[0] * nB + self.jj0[0]).ravel()
        P = np.empty((N + 1, size)); R = np.zeros(N + 1)
        acc = 0.0
        for n in range(N, 0, -1):
            P[n] = x; R[n] = acc
            acc += _dot(x, g0)
            x = (np.bincount(d0, x * p.p0, size) + np.bincount(d1, x * p.p1, size)
                 + np.bincount(d2, x * p.p2, size))
        P[0] = x; R[0] = acc
        return P, R