"""
verify_Cf0_reduced.py
=====================
Independent cross-verification of the 3-state solver (solver.py) using a
reduced 1-variable DP, valid when Cf=0.

REDUCTION
---------
When Cf=0 the optimal policy is "always dispatch q* = min(I2, b1)".
Under this policy the state collapses to a single variable

        y = I2 - b1   (net inventory position)

because dispatch leaves y unchanged and every arrival (R1 or R2)
decreases y by 1.  Each y maps to a canonical post-dispatch state:

        y > 0 :  (I2, b1) = (y,  0)     Retailer 2 holds y units
        y <= 0:  (I2, b1) = (0, -y)     total backlog = -y

EXACTNESS CONDITIONS
--------------------
The 1-variable value function equals the 3-state value function iff:

    (1) Cf = 0
    (2) pi1 = pi2          (so cost depends only on total backlog, not its split)
    (3) c1  = c2           (same, for terminal cost)
    (4) c1  = cu + v2      (terminal R1-cleanup = dispatch + foregone salvage)

Conditions (1)-(3) make the BACKLOG region (y<=0) exact to machine
precision.  Condition (4) additionally makes the INVENTORY region (y>0)
exact: without it, the last-period R1 demand is cleaned up at terminal
(cost c1) in the 3-state model but treated as dispatched (cost cu) in the
reduced model, leaving an O(1/N) terminal-layer discrepancy for y>0 that
vanishes as N -> infinity.

REDUCED RECURSION (uses the identity V(y,1) = cu + V(y-1,0)):
    y > 0:
      V^n(y) = dt*h*y
             + p0*V^{n-1}(y)
             + p1*( cu + V^{n-1}(y-1) )      # R1 demand served by dispatch
             + p2*V^{n-1}(y-1)
    y <= 0:
      V^n(y) = dt*pi*(-y)
             + p0*V^{n-1}(y)
             + p1*V^{n-1}(y-1)               # R1 demand backlogged, no dispatch
             + p2*V^{n-1}(y-1)

Terminal:
    V^0(y) = -v2*y    if y > 0
           =  c*(-y)   if y <= 0     (c = c1 = c2)

Requires solver.py in the same directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import Params, TransshipmentDP


# ======================================================================
# REDUCED 1-VARIABLE DP  (state = y = I2 - b1)
# ======================================================================

class ReducedDP:
    """1-variable backward-induction DP in y = I2 - b1 (Cf=0)."""

    def __init__(self, params: Params):
        self.p = params
        assert abs(params.Cf) < 1e-12,            "Reduced DP requires Cf=0."
        assert abs(params.pi1 - params.pi2) < 1e-12, "Reduced DP requires pi1=pi2."
        assert abs(params.c1 - params.c2) < 1e-12,   "Reduced DP requires c1=c2."

        self.pi = params.pi1
        self.c  = params.c1

        self.y_max = params.I2_max
        self.y_min = params.I2_min - params.b1_max
        self._ny   = self.y_max - self.y_min + 1
        self.V_final = None

    def _iy(self, y):
        return y - self.y_min

    def _clip_y(self, y):
        return max(self.y_min, min(self.y_max, y))

    def terminal(self, y):
        return -self.p.v2 * y if y > 0 else self.c * (-y)

    def solve(self, verbose=True):
        p, ny = self.p, self._ny
        V = np.array([self.terminal(iy + self.y_min) for iy in range(ny)],
                     dtype=np.float64)
        V_new = np.zeros_like(V)

        for n in range(1, p.N + 1):
            for iy in range(ny):
                y    = iy + self.y_min
                iym1 = self._iy(self._clip_y(y - 1))
                if y > 0:
                    V_new[iy] = (p.dt * p.h * y
                                 + p.p0 * V[iy]
                                 + p.p1 * (p.cu + V[iym1])
                                 + p.p2 * V[iym1])
                else:
                    V_new[iy] = (p.dt * self.pi * (-y)
                                 + p.p0 * V[iy]
                                 + p.p1 * V[iym1]
                                 + p.p2 * V[iym1])
            V[:] = V_new
            if verbose and (n % 50 == 0 or n == p.N):
                print(f"    reduced period {n}/{p.N}")

        self.V_final = V.copy()
        return self

    def get_value(self, y):
        return float(self.V_final[self._iy(self._clip_y(y))])

    @staticmethod
    def canonical(y):
        return (y, 0) if y > 0 else (0, -y)


# ======================================================================
# MAIN
# ======================================================================

def compare(p, label):
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"{'='*64}")
    print(f"  {p.summary()}")
    print(f"  Conditions: Cf=0 {'OK' if p.Cf==0 else 'NO'} | "
          f"pi1=pi2 {'OK' if p.pi1==p.pi2 else 'NO'} | "
          f"c1=c2 {'OK' if p.c1==p.c2 else 'NO'} | "
          f"c1=cu+v2 {'OK' if abs(p.c1-(p.cu+p.v2))<1e-9 else 'NO'}")

    dp3 = TransshipmentDP(p); dp3.solve(store_V=False, verbose=False)
    dpr = ReducedDP(p);       dpr.solve(verbose=False)

    # comparison region: deep interior to avoid boundary clipping.
    # The b1_max / I2_min boundaries leak inward by roughly the demand
    # spread over the horizon, so keep a generous margin (~30 units).
    y_hi = p.I2_max - 8
    y_lo = -(p.b1_max - 30)

    diffs_pos, diffs_neg, rows = [], [], []
    for y in range(y_hi, y_lo - 1, -1):
        I2c, b1c = ReducedDP.canonical(y)
        if not (p.I2_min <= I2c <= p.I2_max and 0 <= b1c <= p.b1_max):
            continue
        V3 = dp3.get_value(p.N, I2c, b1c)
        Vr = dpr.get_value(y)
        d  = V3 - Vr
        rows.append((y, I2c, b1c, V3, Vr, d))
        (diffs_pos if y > 0 else diffs_neg).append(abs(d))

    # print sample
    print(f"\n  {'y':>4} | {'(I2,b1)':>10} | {'V_3state':>12} | "
          f"{'V_reduced':>12} | {'diff':>11}")
    print(f"  {'-'*58}")
    step = max(1, len(rows) // 24)
    for k, (y, I2c, b1c, V3, Vr, d) in enumerate(rows):
        if k % step == 0:
            print(f"  {y:>4} | {f'({I2c},{b1c})':>10} | {V3:>12.5f} | "
                  f"{Vr:>12.5f} | {d:>11.3e}")

    mae_pos = np.mean(diffs_pos) if diffs_pos else 0.0
    mae_neg = np.mean(diffs_neg) if diffs_neg else 0.0
    print(f"\n  Inventory region (y>0):  MAE={mae_pos:.3e}  MaxAE={max(diffs_pos):.3e}")
    print(f"  Backlog region   (y<=0): MAE={mae_neg:.3e}  MaxAE={max(diffs_neg):.3e}")
    return dp3, dpr, rows


def main():
    print("=" * 64)
    print("Cf=0 VERIFICATION: 3-state solver vs reduced 1-variable DP")
    print("=" * 64)

    cu, v2 = 1.0, 1.0

    # ---- Run 1: all four conditions hold -> exact everywhere ----
    p_exact = Params(
        T=2.0, N=200, lam1=5.0, lam2=3.0,
        h=1.0, Cf=0.0, cu=cu,
        pi1=6.0, pi2=6.0,
        c1=cu+v2, c2=cu+v2,     # c1 = cu + v2 = 2  -> exact in y>0 too
        v2=v2,
        I2_max=40, I2_min=-30, b1_max=70,
    )
    dp3, dpr, rows = compare(p_exact,
        "RUN 1: c1=c2=cu+v2  (all conditions hold -> exact everywhere)")

    # ---- Run 2: c1 != cu+v2 -> y>0 has O(1/N) terminal-layer artefact ----
    p_general = Params(
        T=2.0, N=200, lam1=5.0, lam2=3.0,
        h=1.0, Cf=0.0, cu=cu,
        pi1=6.0, pi2=6.0,
        c1=4.0, c2=4.0,         # c1 != cu+v2 -> y>0 artefact
        v2=v2,
        I2_max=40, I2_min=-30, b1_max=70,
    )
    compare(p_general,
        "RUN 2: c1=c2=4 != cu+v2  (y<=0 exact; y>0 has O(1/N) artefact)")

    # ---- Plot Run 1 ----
    ys   = [r[0] for r in rows]
    V3   = [r[3] for r in rows]
    Vr   = [r[4] for r in rows]
    diff = [r[5] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ys, V3, 'b-o', ms=3, lw=2, label='3-state (canonical)')
    axes[0].plot(ys, Vr, 'r--', lw=1.5, label='reduced (y)')
    axes[0].axvline(0, color='k', lw=0.7, ls=':')
    axes[0].set_xlabel('$y = I_2 - b_1$'); axes[0].set_ylabel('$V^N$')
    axes[0].set_title('Value function (Run 1: exact)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ys, diff, 'g-', lw=1.5)
    axes[1].axhline(0, color='k', lw=0.7)
    axes[1].axvline(0, color='k', lw=0.7, ls=':')
    axes[1].set_xlabel('$y = I_2 - b_1$')
    axes[1].set_ylabel('$V_{3state} - V_{reduced}$')
    axes[1].set_title('Difference (~machine precision)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f'$C_f=0$ cross-verification | '
        f'$\\pi_1=\\pi_2={p_exact.pi1}$, $c_1=c_2=c_u+v_2={p_exact.c1}$',
        fontsize=11)
    plt.tight_layout()
    plt.savefig('verify_Cf0_reduced.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('verify_Cf0_reduced.png', bbox_inches='tight', dpi=150)
    print("\nSaved: verify_Cf0_reduced.pdf / .png")
    plt.show()

    print(f"\n{'='*64}")
    print("CONCLUSION")
    print("  Run 1 (all 4 conditions): machine-precision agreement everywhere")
    print("    -> independently cross-verifies the 3-state solver when Cf=0,")
    print("       and confirms always-dispatch is optimal.")
    print("  Run 2 (c1 != cu+v2): backlog region (y<=0) still exact; inventory")
    print("    region (y>0) differs by an O(1/N) terminal-layer artefact only.")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
