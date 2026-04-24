"""
policy_visualizer.py
====================
Interactive policy explorer for the two-location transshipment model.
Run as:  python policy_visualizer.py

After the DP solves, four plot types are available:
  1. q*(I2, b1)     -- heatmap of optimal dispatch quantity at chosen tau
  2. q*(b1)         -- line plot for fixed I2 values, across b1
  3. q*(I2)         -- line plot for fixed b1 values, across I2
  4. threshold(tau) -- b1*(I2,tau) and I2bar(tau) as tau varies

All plots use the same solved DP, so you only pay the solve cost once.
Parameters can be changed and the DP re-solved at any time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import math
from solver import Params, TransshipmentDP


# ======================================================================
# DEFAULT PARAMETERS  -- change these before running
# ======================================================================
DEFAULT_PARAMS = dict(
    # Horizon
    T    = 2.0,
    N    = 200,
    # Demand
    lam1 = 5.0,
    lam2 = 3.0,
    # Costs
    h    = 1.0,
    Cf   = 10.0,
    cu   = 1.0,
    pi1  = 8.0,
    pi2  = 8.0,
    # Terminal
    c1   = 8.0,
    c2   = 8.0,
    v2   = 1.0,
    # State space
    I2_max = 25,
    I2_min = -5,
    b1_max = 40,
)

# ======================================================================
# ANALYTICAL THRESHOLDS  (paper formulas, for overlay on plots)
# ======================================================================

def _sqrt(x):
    return math.sqrt(max(0.0, x))

def analytical_b1star(I2, tau, lam2, cu, h, pi1, pi2, Cf):
    """b1*(I2, tau) -- paper Case 1 threshold (no error in Case 1 formula)."""
    a2   = lam2 * cu / (h + pi2)
    phi2 = 2 * lam2 * Cf / (h + pi2)
    gam  = lam2 * (pi1 - pi2) / (h + pi2)
    A    = 2 * I2 + 1 - 2 * a2 + 2 * gam * tau
    D    = A**2 - 4 * phi2
    if D < 0:
        return None
    return 0.5 * (A - _sqrt(D))

def analytical_I2bar_correct(tau, lam2, cu, h, pi1, pi2, Cf):
    """Ī₂(tau) -- corrected formula."""
    a2   = lam2 * cu / (h + pi2)
    phi2 = 2 * lam2 * Cf / (h + pi2)
    gam  = lam2 * (pi1 - pi2) / (h + pi2)
    xi   = 1 - 2 * a2 + 2 * gam * tau
    return 0.5 * (_sqrt(xi**2 + 4 * phi2) - xi)

def analytical_I2bar_paper(tau, lam2, cu, h, pi1, pi2, Cf):
    """Ī₂(tau) -- paper formula (errors kept)."""
    a2      = lam2 * cu / (h + pi2)
    gam     = lam2 * (pi1 - pi2) / (h + pi2)
    eta     = 1 + 2 * a2 - 2 * gam * tau
    phi_err = 2 * lam2 * Cf / (h + pi1)
    return 0.5 * (_sqrt(eta**2 + 4 * phi_err) - eta)


# ======================================================================
# SOLVER WRAPPER
# ======================================================================

def solve(params: dict, verbose=True) -> TransshipmentDP:
    p = Params(
        T      = params['T'],
        N      = params['N'],
        lam1   = params['lam1'],
        lam2   = params['lam2'],
        h      = params['h'],
        Cf     = params['Cf'],
        cu     = params['cu'],
        pi1    = params['pi1'],
        pi2    = params['pi2'],
        c1     = params['c1'],
        c2     = params['c2'],
        v2     = params['v2'],
        I2_max = params['I2_max'],
        I2_min = params['I2_min'],
        b1_max = params['b1_max'],
    )
    dp = TransshipmentDP(p)
    dp.solve(store_V=False, verbose=verbose)
    return dp

def n_for_tau(tau, dp):
    """Convert remaining time tau to period index n."""
    return max(0, min(dp.p.N, round(tau / dp.p.dt)))


# ======================================================================
# DIMENSIONLESS GROUP SUMMARY
# ======================================================================

def print_groups(params):
    lam2, cu, h = params['lam2'], params['cu'], params['h']
    pi1, pi2    = params['pi1'], params['pi2']
    Cf, T       = params['Cf'], params['T']
    pi          = pi2  # symmetric reference
    a   = lam2 * cu / (h + pi)
    a2  = lam2 * cu / (h + pi2)
    phi = 2 * lam2 * Cf / (h + pi)
    phi2= 2 * lam2 * Cf / (h + pi2)
    gam = lam2 * (pi1 - pi2) / (h + pi2)
    print(f"\n  alpha  = lam2*cu/(h+pi2) = {a2:.4f}  "
          f"{'[A1/B1a: alpha<=0.5 OK]' if a2<=0.5 else '[A2: alpha>0.5]'}")
    print(f"  Phi2   = 2*lam2*Cf/(h+pi2) = {phi2:.4f}")
    print(f"  gamma  = lam2*(pi1-pi2)/(h+pi2) = {gam:.4f}  "
          f"{'[pi1>pi2 -> B1a/b/c]' if gam>0 else '[pi1=pi2 -> A1/A2]' if gam==0 else '[pi1<pi2 -> B2a/b/c]'}")
    print(f"  gamma*T = {gam*T:.4f}  "
          f"  boundary: alpha2-0.5 = {a2-0.5:.4f}")
    if gam != 0:
        tau_star = (2*a2-1)/(2*gam)
        print(f"  tau*   = {tau_star:.4f}  "
              f"{'(tau*<0: B1a/B2a region)' if tau_star<0 else '(tau* in [0,T]: B1b/B2b region)' if 0<tau_star<T else '(tau*>T: B1c/B2c region)'}")
    beta = round(a - 0.5)
    print(f"  beta   = round(alpha-0.5) = {beta}  (A2 hold-back quantity)")


# ======================================================================
# PLOT 1: q*(I2, b1) HEATMAP at fixed tau
# ======================================================================

def plot_heatmap(dp, tau, ax=None, title_extra=""):
    n   = n_for_tau(tau, dp)
    p   = dp.p
    I2s = np.arange(1, p.I2_max + 1)
    b1s = np.arange(0, min(p.I2_max + 5, p.b1_max) + 1)

    Q = np.zeros((len(b1s), len(I2s)), dtype=int)
    for j, I2 in enumerate(I2s):
        for i, b1 in enumerate(b1s):
            Q[i, j] = dp.get_policy(n, I2, b1)

    q_max = Q.max()
    cmap  = get_cmap('Blues', q_max + 1) if q_max > 0 else get_cmap('Blues', 2)
    bounds = np.arange(-0.5, q_max + 1.5)
    norm   = BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(I2s, b1s, Q, cmap=cmap, norm=norm, shading='nearest')
    plt.colorbar(im, ax=ax, label='q*', ticks=np.arange(0, q_max + 1))

    # overlay analytical thresholds
    params = dict(lam2=p.lam2, cu=p.cu, h=p.h, pi1=p.pi1, pi2=p.pi2, Cf=p.Cf)
    b1_thresh = []
    for I2 in I2s:
        v = analytical_b1star(I2, tau, **params)
        b1_thresh.append(v if v is not None else np.nan)
    ax.plot(I2s, b1_thresh, 'r--', lw=1.5, label='b₁*(I₂,τ) analytical')

    I2bar_c = analytical_I2bar_correct(tau, **params)
    I2bar_p = analytical_I2bar_paper(tau, **params)
    ax.axvline(I2bar_c, color='green', lw=1.5, ls='--', label=f'Ī₂ corrected={I2bar_c:.2f}')
    ax.axvline(I2bar_p, color='orange', lw=1.5, ls=':', label=f'Ī₂ paper={I2bar_p:.2f}')
    ax.plot([1, p.I2_max], [1, p.I2_max], 'w-', lw=0.8, alpha=0.4, label='b₁=I₂ (case boundary)')

    ax.set_xlabel('I₂ (retailer 2 inventory)')
    ax.set_ylabel('b₁ (retailer 1 backlog)')
    ax.set_title(f'q*(I₂, b₁)  at τ={tau:.2f}  {title_extra}')
    ax.legend(fontsize=8, loc='upper left')
    return ax


# ======================================================================
# PLOT 2: q*(b1) for fixed I2 values
# ======================================================================

def plot_q_vs_b1(dp, tau, I2_list=None, ax=None, title_extra=""):
    n = n_for_tau(tau, dp)
    p = dp.p
    if I2_list is None:
        I2_list = [3, 6, 10, 15, 20]
    I2_list = [I2 for I2 in I2_list if I2 <= p.I2_max]
    b1s = np.arange(0, min(40, p.b1_max) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(I2_list)))
    for I2, col in zip(I2_list, colors):
        qs = [dp.get_policy(n, I2, b1) for b1 in b1s]
        ax.step(b1s, qs, where='post', color=col, lw=1.8, label=f'I₂={I2}')
        # mark b1* threshold
        params = dict(lam2=p.lam2, cu=p.cu, h=p.h, pi1=p.pi1, pi2=p.pi2, Cf=p.Cf)
        v = analytical_b1star(I2, tau, **params)
        if v is not None and v <= b1s[-1]:
            ax.axvline(v, color=col, lw=0.8, ls=':', alpha=0.7)

    ax.set_xlabel('b₁ (retailer 1 backlog)')
    ax.set_ylabel('q* (dispatch quantity)')
    ax.set_title(f'q*(b₁) for fixed I₂  at τ={tau:.2f}  {title_extra}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, None)
    return ax


# ======================================================================
# PLOT 3: q*(I2) for fixed b1 values
# ======================================================================

def plot_q_vs_I2(dp, tau, b1_list=None, ax=None, title_extra=""):
    n = n_for_tau(tau, dp)
    p = dp.p
    if b1_list is None:
        b1_list = [1, 3, 5, 10, 20]
    b1_list = [b1 for b1 in b1_list if b1 <= p.b1_max]
    I2s = np.arange(1, p.I2_max + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(b1_list)))
    params = dict(lam2=p.lam2, cu=p.cu, h=p.h, pi1=p.pi1, pi2=p.pi2, Cf=p.Cf)

    for b1, col in zip(b1_list, colors):
        qs = [dp.get_policy(n, I2, b1) for I2 in I2s]
        ax.step(I2s, qs, where='post', color=col, lw=1.8, label=f'b₁={b1}')

    # I2bar overlays
    I2bar_c = analytical_I2bar_correct(tau, **params)
    I2bar_p = analytical_I2bar_paper(tau, **params)
    ax.axvline(I2bar_c, color='green', lw=1.5, ls='--', label=f'Ī₂ corrected={I2bar_c:.2f}')
    ax.axvline(I2bar_p, color='orange', lw=1.5, ls=':', label=f'Ī₂ paper={I2bar_p:.2f}')

    ax.set_xlabel('I₂ (retailer 2 inventory)')
    ax.set_ylabel('q* (dispatch quantity)')
    ax.set_title(f'q*(I₂) for fixed b₁  at τ={tau:.2f}  {title_extra}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, None)
    return ax


# ======================================================================
# PLOT 4: threshold vs tau
# ======================================================================

def plot_thresholds_vs_tau(dp, I2_list_case1=None, b1_large=45, ax=None, title_extra=""):
    p = dp.p
    taus = np.linspace(0.05, p.T, 40)
    params = dict(lam2=p.lam2, cu=p.cu, h=p.h, pi1=p.pi1, pi2=p.pi2, Cf=p.Cf)

    if I2_list_case1 is None:
        I2_list_case1 = [5, 10, 15]
    I2_list_case1 = [I2 for I2 in I2_list_case1 if I2 <= p.I2_max]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    # -- b1*(I2, tau): DP vs analytical --
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(I2_list_case1)))
    for I2, col in zip(I2_list_case1, colors):
        # DP threshold
        dp_thresh = []
        for tau in taus:
            n  = n_for_tau(tau, dp)
            th = None
            for b1 in range(1, min(I2, p.b1_max) + 1):
                if dp.get_policy(n, I2, b1) > 0:
                    th = b1
                    break
            dp_thresh.append(th if th is not None else np.nan)
        ax.plot(taus, dp_thresh, color=col, lw=2, label=f'b₁*(I₂={I2}) DP')
        # analytical
        an_thresh = [analytical_b1star(I2, tau, **params) for tau in taus]
        an_thresh = [v if v is not None else np.nan for v in an_thresh]
        ax.plot(taus, an_thresh, color=col, lw=1.2, ls='--', alpha=0.7,
                label=f'b₁*(I₂={I2}) analytical')

    # -- I2bar(tau): DP vs corrected vs paper --
    dp_I2bar = []
    for tau in taus:
        n  = n_for_tau(tau, dp)
        th = None
        for I2 in range(1, p.I2_max + 1):
            if dp.get_policy(n, I2, min(b1_large, p.b1_max)) > 0:
                th = I2
                break
        dp_I2bar.append(th if th is not None else np.nan)
    ax.plot(taus, dp_I2bar, 'k-', lw=2.5, label='Ī₂(τ) DP')

    I2bar_corr  = [analytical_I2bar_correct(tau, **params) for tau in taus]
    I2bar_paper = [analytical_I2bar_paper(tau, **params)   for tau in taus]
    ax.plot(taus, I2bar_corr,  'g--', lw=1.5, label='Ī₂(τ) corrected')
    ax.plot(taus, I2bar_paper, 'r:',  lw=1.5, label='Ī₂(τ) paper (errors)')

    ax.set_xlabel('τ (remaining time)')
    ax.set_ylabel('threshold value')
    ax.set_title(f'Thresholds vs τ  {title_extra}')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, p.T)
    return ax


# ======================================================================
# COMBINED DASHBOARD (all 4 plots in one figure)
# ======================================================================

def dashboard(dp, tau=None, params=None,
              I2_fixed=[3, 6, 10, 15],
              b1_fixed=[1, 3, 5, 10, 20],
              I2_thresh=[5, 10, 15]):
    if tau is None:
        tau = dp.p.T

    tag = ""
    if params:
        tag = (f"  [lam2={params['lam2']} cu={params['cu']} "
               f"pi1={params['pi1']} pi2={params['pi2']} "
               f"Cf={params['Cf']} h={params['h']}]")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Optimal dispatch policy{tag}\nτ={tau:.2f}  (for heatmap and line plots)",
                 fontsize=12, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_heatmap(dp, tau, ax=ax1)
    plot_q_vs_b1(dp, tau, I2_list=I2_fixed, ax=ax2)
    plot_q_vs_I2(dp, tau, b1_list=b1_fixed, ax=ax3)
    plot_thresholds_vs_tau(dp, I2_list_case1=I2_thresh, ax=ax4)

    plt.savefig('policy_dashboard.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('policy_dashboard.png', bbox_inches='tight', dpi=150)
    print("\n  Saved: policy_dashboard.pdf and policy_dashboard.png")
    plt.show()


# ======================================================================
# INTERACTIVE MENU
# ======================================================================

def prompt_float(msg, default):
    s = input(f"  {msg} [{default}]: ").strip()
    return float(s) if s else default

def prompt_int(msg, default):
    s = input(f"  {msg} [{default}]: ").strip()
    return int(s) if s else default

def prompt_list(msg, default):
    s = input(f"  {msg} [{' '.join(map(str,default))}]: ").strip()
    if not s:
        return default
    return [int(x) for x in s.split()]

def edit_params(params):
    print("\n  --- Edit parameters (press Enter to keep current value) ---")
    params['T']      = prompt_float("T (horizon)",            params['T'])
    params['N']      = prompt_int  ("N (periods)",            params['N'])
    params['lam1']   = prompt_float("lam1",                   params['lam1'])
    params['lam2']   = prompt_float("lam2",                   params['lam2'])
    params['h']      = prompt_float("h (holding)",            params['h'])
    params['Cf']     = prompt_float("Cf (fixed ship cost)",   params['Cf'])
    params['cu']     = prompt_float("cu (unit ship cost)",    params['cu'])
    params['pi1']    = prompt_float("pi1 (R1 penalty)",       params['pi1'])
    params['pi2']    = prompt_float("pi2 (R2 penalty)",       params['pi2'])
    params['c1']     = prompt_float("c1 (terminal R1)",       params['c1'])
    params['c2']     = prompt_float("c2 (terminal R2)",       params['c2'])
    params['v2']     = prompt_float("v2 (salvage)",           params['v2'])
    params['I2_max'] = prompt_int  ("I2_max",                 params['I2_max'])
    params['I2_min'] = prompt_int  ("I2_min",                 params['I2_min'])
    params['b1_max'] = prompt_int  ("b1_max",                 params['b1_max'])
    return params


def main():
    params = DEFAULT_PARAMS.copy()
    dp     = None

    print("\n" + "="*60)
    print("  OPTIMAL DISPATCH POLICY VISUALIZER")
    print("="*60)

    while True:
        print(f"""
  [1] Edit parameters and re-solve
  [2] Print dimensionless groups
  [3] Dashboard (all 4 plots)
  [4] Heatmap only  (q* vs I2, b1 at given tau)
  [5] q*(b1) lines  (fixed I2 values)
  [6] q*(I2) lines  (fixed b1 values)
  [7] Thresholds vs tau
  [0] Quit
""")
        choice = input("  Choice: ").strip()

        if choice == '0':
            break

        elif choice == '1':
            params = edit_params(params)
            print("\n  Solving DP...")
            dp = solve(params, verbose=True)
            print_groups(params)

        elif dp is None:
            print("\n  Solve first (option 1).")
            continue

        elif choice == '2':
            print_groups(params)

        elif choice == '3':
            tau = prompt_float("tau for heatmap/line plots", params['T'])
            I2_fixed  = prompt_list("I2 values for q*(b1) plot", [3, 6, 10, 15])
            b1_fixed  = prompt_list("b1 values for q*(I2) plot", [1, 3, 5, 10, 20])
            I2_thresh = prompt_list("I2 values for threshold plot", [5, 10, 15])
            dashboard(dp, tau=tau, params=params,
                      I2_fixed=I2_fixed, b1_fixed=b1_fixed,
                      I2_thresh=I2_thresh)

        elif choice == '4':
            tau = prompt_float("tau", params['T'])
            fig, ax = plt.subplots(figsize=(9, 6))
            plot_heatmap(dp, tau, ax=ax)
            plt.tight_layout()
            plt.show()

        elif choice == '5':
            tau     = prompt_float("tau", params['T'])
            I2_list = prompt_list("I2 values", [3, 6, 10, 15])
            fig, ax = plt.subplots(figsize=(9, 5))
            plot_q_vs_b1(dp, tau, I2_list=I2_list, ax=ax)
            plt.tight_layout()
            plt.show()

        elif choice == '6':
            tau     = prompt_float("tau", params['T'])
            b1_list = prompt_list("b1 values", [1, 3, 5, 10, 20])
            fig, ax = plt.subplots(figsize=(9, 5))
            plot_q_vs_I2(dp, tau, b1_list=b1_list, ax=ax)
            plt.tight_layout()
            plt.show()

        elif choice == '7':
            I2_list = prompt_list("I2 values for b1*(I2,tau)", [5, 10, 15])
            b1_large = prompt_int("b1_large for I2bar query", 45)
            fig, ax  = plt.subplots(figsize=(10, 5))
            plot_thresholds_vs_tau(dp, I2_list_case1=I2_list,
                                   b1_large=b1_large, ax=ax)
            plt.tight_layout()
            plt.show()

        else:
            print("  Unknown option.")


if __name__ == "__main__":
    main()