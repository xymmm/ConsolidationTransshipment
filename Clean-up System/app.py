"""
app.py  —  Interactive Transshipment Policy Explorer
=====================================================
Run with:
    streamlit run app.py

Requires solver.py in the same directory.
Install dependencies:
    pip install streamlit matplotlib numpy plotly scipy
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson
from solver import Params, TransshipmentDP
from solver_cf0_2d import ParamsCf0, SwitchingDPCf0


@st.cache_data(show_spinner=False)
def solve_cf0_2d(T_, N_, lam1_, lam2_, h_, cu_, pi1_, pi2_, c2_, v2_, taus_key):
    """
    Solve the note's 2-D Cf=0 switching model and return its threshold curve.
    The state space is 1-D, so even a large N is cheap. Cached on the parameter
    tuple so that reruns do not re-solve.
    """
    p0 = ParamsCf0(T=T_, N=N_, lam1=lam1_, lam2=lam2_, h=h_, cu=cu_,
                   pi1=pi1_, pi2=pi2_, c2=c2_, v2=v2_).with_auto_bounds()
    dp0 = SwitchingDPCf0(p0)
    dp0.solve(verbose=False)
    return dp0.threshold_curve(np.array(taus_key))

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Transshipment Policy Explorer",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Transshipment Policy Explorer")
st.caption("Optimal dispatch policy via backward-induction DP · analytical overlays · Cf=0 and general cases")

# ======================================================================
# SIDEBAR: PARAMETERS
# ======================================================================
st.sidebar.header("Model Parameters")

with st.sidebar.expander("⏱  Horizon & Discretisation", expanded=True):
    T = st.slider("T (horizon length)", 0.5, 10.0, 2.0, 0.5)
    N = st.select_slider("N (periods)", options=[50, 100, 200, 400, 800], value=200)

with st.sidebar.expander("📈  Demand rates", expanded=True):
    lam1 = st.slider("λ₁ (Retailer 1 rate)", 0.5, 15.0, 5.0, 0.5)
    lam2 = st.slider("λ₂ (Retailer 2 rate)", 0.5, 10.0, 3.0, 0.5)

# All cost-related parameters now allow a minimum of 0 and default to 0.
with st.sidebar.expander("💰  Cost parameters", expanded=True):
    h   = st.slider("h  (holding cost)",     0.0, 5.0,  0.0, 0.1)
    Cf  = st.slider("Cf (fixed ship cost)",  0.0, 30.0, 0.0, 0.5)
    cu  = st.slider("cu (unit ship cost)",   0.0, 15.0, 0.0, 0.1)
    pi1 = st.slider("π₁ (R1 penalty)",       0.0, 20.0, 0.0, 0.5)
    pi2 = st.slider("π₂ (R2 penalty)",       0.0, 20.0, 0.0, 0.5)

with st.sidebar.expander("🏁  Terminal costs", expanded=False):
    c1 = st.slider("c₁ (clear R1 backlog)", 0.0, 20.0, 0.0, 0.5)
    c2 = st.slider("c₂ (clear R2 backlog)", 0.0, 20.0, 0.0, 0.5)
    v2 = st.slider("v₂ (salvage R2 inv)",   0.0, 10.0, 0.0, 0.5)
    st.caption("The exact Cf=0 staircase assumes zero terminal cost "
               "(c₁=c₂=v₂=0). Set them to 0 to match the analytical figure.")

# ── adaptive state-space bounds ─────────────────────────────────────────
# Truncating the state space too tightly caps the Retailer-2 backorder cost
# and biases the DP toward early dispatch. The safe rule scales the bounds
# with the expected demand mass over the horizon plus a ~4-sigma buffer.
with st.sidebar.expander("📐  State space", expanded=False):
    s1 = lam1 * T          # expected Retailer-1 demand over the horizon
    s2 = lam2 * T          # expected Retailer-2 demand over the horizon
    rec_I2_min = -int(math.ceil(s2 + 4.0 * math.sqrt(s2)))
    rec_I2_max =  int(math.ceil(max(40.0, s2 + 4.0 * math.sqrt(s2))))
    rec_b1_max =  int(math.ceil(s1 + 4.0 * math.sqrt(s1)))

    auto_bounds = st.checkbox("Auto bounds (recommended)", value=True)
    if auto_bounds:
        I2_max, I2_min, b1_max = rec_I2_max, rec_I2_min, rec_b1_max
        st.caption(f"Auto: I₂∈[{I2_min}, {I2_max}], b1_max={b1_max}  "
                   f"(from λ₁T={s1:.0f}, λ₂T={s2:.0f})")
    else:
        # Widened ranges so the recommended values are always reachable.
        I2_max = st.slider("I2_max",  10, 200, rec_I2_max, 5)
        I2_min = st.slider("I2_min", -200,  0, rec_I2_min, 1)
        b1_max = st.slider("b1_max",  10, 200, rec_b1_max, 5)

# ── Cf = 0 comparison overlays ──────────────────────────────────────────
with st.sidebar.expander("🔍  Cf = 0 model comparison", expanded=False):
    if Cf == 0:
        show_note_staircase = st.checkbox("Show note staircase (eq. 20-22)", value=True)
        show_cf0_2d = st.checkbox("Overlay 2-D Cf=0 DP (note's model)", value=True)
        n_cf0 = st.select_slider("N for the 2-D DP",
                                 options=[500, 1000, 2000, 4000, 8000],
                                 value=2000)
        st.caption(
            "Applies to the 'Ī₂ threshold (Case 2)' plot only. These curves are "
            "controlled here, not by any other toggle.\n\n"
            "The note solves a TWO-dimensional model with value function "
            "V(I₂, τ). Retailer-1 demand is either satisfied on arrival at cost "
            "cᵤ or rejected and charged π₁τ, and the resulting backlog is never "
            "cleared. There is no b₁ state variable.\n\n"
            "solver.py is a different, three-dimensional model V(I₂, b₁, τ) in "
            "which the Retailer-1 backlog is tracked and can be cleared by a "
            "later dispatch. The two models therefore have different optimal "
            "thresholds.\n\n"
            "Use the overlay to compare the note staircase against a DP of the "
            "note's OWN model. The fixed b₁ slider does not affect these curves, "
            "since the note's model has no b₁."
        )
    else:
        show_note_staircase = False
        show_cf0_2d = False
        n_cf0 = 2000
        st.caption("Set Cf = 0 to enable the comparison with the note's 2-D model.")

# ── threshold display option ────────────────────────────────────────────
with st.sidebar.expander("📊  Threshold display", expanded=False):
    retained = st.checkbox("Show retained level (Ī₂ − 1)", value=False)
    st.caption(
        "Applies to the 'Ī₂ threshold (Case 2)' plot only.\n\n"
        "OFF (default): the curve is the PARTICIPATION THRESHOLD Ī₂, i.e. the "
        "smallest Retailer-2 inventory I₂ at which a dispatch happens.\n\n"
        "ON: the curve is the RETAINED LEVEL Ī₂ − 1, i.e. the inventory kept "
        "after dispatch (equivalently, the largest I₂ that does NOT dispatch). "
        "Turn this on to line up with a figure that plots retained inventory "
        "instead of the participation threshold."
    )

# ── solve button ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
solve_btn = st.sidebar.button("▶  Solve DP", type="primary", use_container_width=True)
st.sidebar.caption("Press after changing parameters. DP may take a few seconds.")

# ======================================================================
# DIMENSIONLESS GROUPS (always shown, no solve needed)
# ======================================================================
hp2 = max(h + pi2, 1e-9)   # guard against division by zero when h=pi2=0
alpha2 = lam2 * cu / hp2
phi2   = 2 * lam2 * Cf / hp2 if Cf > 0 else 0.0
gamma  = lam2 * (pi1 - pi2) / hp2
beta   = round(alpha2 - 0.5)
gamT   = gamma * T

def classify(a2, gT):
    if abs(gamma) < 1e-9:
        return "A1" if a2 <= 0.5 else "A2"
    elif gamma > 0:
        if a2 <= 0.5:
            return "B1a"
        elif gT > a2 - 0.5:
            return "B1b"
        else:
            return "B1c"
    else:
        if a2 <= 0.5:
            return "B2a" if gT >= a2 - 0.5 else "B2b"
        else:
            return "B2c"

region = classify(alpha2, gamT)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("α₂ = λ₂cᵤ/(h+π₂)",  f"{alpha2:.3f}")
col_info2.metric("γ = λ₂(π₁−π₂)/(h+π₂)", f"{gamma:.3f}")
col_info3.metric("Φ₂ = 2λ₂Cf/(h+π₂)",  f"{phi2:.3f}")
col_info4.metric("Region",  region,
                 delta="β="+str(beta) if region in ("A2","B1b","B1c","B2b","B2c") else None)

st.markdown("---")

# ======================================================================
# ANALYTICAL HELPERS
# ======================================================================

def tau_grid(T_, n_uniform=300, n_dense=400, band=0.6):
    """
    Dense τ grid for the plots. A uniform grid over (0, T] is combined with a
    fine cluster just above τ* = cu/(h+π₁), where the threshold plunges from
    +∞ through a few one-unit microsteps. A coarse uniform grid samples too
    few points there and aliases the microsteps away.
    """
    tau_star = cu / max(h + pi1, 1e-9)
    base  = np.linspace(0.05, T_, n_uniform)
    lo    = max(0.05, tau_star - 0.05)
    hi    = min(T_, tau_star + band)
    dense = np.linspace(lo, hi, n_dense) if hi > lo else np.array([])
    return np.unique(np.concatenate([base, dense]))


def an_I2bar_Cf0_exact(tau, nmax=200):
    """
    Note staircase, eq. (20)-(22):
        Ibar(tau) = min{ n>=1 : M(n,tau) >= g(tau) },
        M(n,tau) = E[min(K,n)] = sum_{j=1}^n P(K>=j),   K ~ Poisson(lam2*tau),
        g(tau)   = lam2 * (cu + (pi2-pi1)*tau) / (h+pi2).
    Returns np.nan when the threshold is +infinity (no finite dispatch level).
    """
    if tau <= 0:
        return np.nan
    g = lam2 * (cu + (pi2 - pi1) * tau) / hp2
    mu = lam2 * tau
    # Poisson(mu) tail built iteratively (no scipy):
    #   pmf(0) = e^{-mu},  pmf(j) = pmf(j-1) * mu / j
    #   P(K >= n) = 1 - P(K <= n-1)
    M = 0.0
    cdf_below = 0.0          # P(K <= n-2), i.e. mass strictly below n-1
    pmf = math.exp(-mu)      # pmf(n-1), starts at pmf(0)
    for n in range(1, nmax + 1):
        p_ge_n = 1.0 - (cdf_below + pmf)   # P(K >= n)
        if p_ge_n < 0.0:
            p_ge_n = 0.0
        M += p_ge_n
        if M >= g:
            return float(n)
        cdf_below += pmf     # now P(K <= n-1)
        pmf *= mu / n        # advance to pmf(n)
    return np.nan

# ======================================================================
# SESSION STATE
# ======================================================================
if "dp" not in st.session_state:
    st.session_state.dp = None
if "dp_params" not in st.session_state:
    st.session_state.dp_params = None

if solve_btn:
    # Warn if the chosen lower bound is likely to bind and distort the policy.
    if s2 + 3.0 * math.sqrt(s2) > -I2_min:
        st.warning(
            f"I2_min={I2_min} may be too tight for λ₂T={s2:.0f}. "
            f"Recommended I2_min ≤ {rec_I2_min}. "
            f"A binding lower bound caps the Retailer-2 backorder cost and can "
            f"shift the DP thresholds. Enable 'Auto bounds' for a safe setting."
        )
    try:
        with st.spinner("Solving DP..."):
            p = Params(
                T=T, N=N, lam1=lam1, lam2=lam2,
                h=h, Cf=Cf, cu=cu, pi1=pi1, pi2=pi2,
                c1=c1, c2=c2, v2=v2,
                I2_max=I2_max, I2_min=I2_min, b1_max=b1_max,
            )
            dp = TransshipmentDP(p)
            dp.solve(store_V=True, verbose=False)  # store_V=True needed for V^n queries
            st.session_state.dp = dp
            st.session_state.dp_params = dict(
                T=T, N=N, lam1=lam1, lam2=lam2,
                h=h, Cf=Cf, cu=cu, pi1=pi1, pi2=pi2,
                c1=c1, c2=c2, v2=v2,
                I2_max=I2_max, I2_min=I2_min, b1_max=b1_max,
            )
        st.success("DP solved!")
    except AssertionError as e:
        # e.g. v2 > c2 violates the model assumption, or Delta t too large.
        st.error(f"Invalid parameters: {e}")

dp = st.session_state.dp

def n_for_tau(tau, dp):
    dt = dp.p.T / dp.p.N
    return min(dp.p.N, max(1, round(tau / dt)))

# ======================================================================
# TABS:  2D PLOTS  /  3D PLOTS
# ======================================================================
tab_2d, tab_3d = st.tabs(["📈 2D Plots", "🧊 3D Plots"])

# ======================================================================
# TAB 1: 2D PLOTS
# ======================================================================
with tab_2d:
    st.subheader("Plot settings")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        x_choice = st.selectbox(
            "X axis",
            ["τ (remaining time)", "I₂ (inventory)", "b₁ (backlog)"],
            key="x2d",
        )
    with pc2:
        y_choice = st.selectbox(
            "Y axis",
            ["q* (optimal dispatch quantity)",
             "b₁* threshold (Case 1)",
             "Ī₂ threshold (Case 2)",
             "V^n (value function)"],
            key="y2d",
        )
    with pc3:
        st.caption("")

    # fixed-dimension sliders
    _T_f    = float(dp.p.T)    if dp is not None else float(T)
    _I2_max = dp.p.I2_max      if dp is not None else I2_max
    _I2_min = dp.p.I2_min      if dp is not None else I2_min
    _b1_max = dp.p.b1_max      if dp is not None else b1_max

    pc4, pc5 = st.columns(2)
    with pc4:
        if x_choice != "τ (remaining time)":
            tau_fixed = st.slider("Fixed τ", 0.05, _T_f, _T_f, 0.05, key="tau2d")
        else:
            tau_fixed = _T_f
        if x_choice != "I₂ (inventory)":
            I2_fixed = st.slider("Fixed I₂", 1, _I2_max, min(10, _I2_max), key="i22d")
        else:
            I2_fixed = min(10, _I2_max)
    with pc5:
        if x_choice != "b₁ (backlog)":
            b1_fixed = st.slider("Fixed b₁", 1, _b1_max, min(5, _b1_max), key="b12d")
        else:
            b1_fixed = min(5, _b1_max)
        n_lines = st.slider("Number of curves", 1, 8, 3, key="nl2d")

    # plot
    if dp is None:
        st.info("👈  Set parameters and press **Solve DP** to generate plots.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        colours = cm.tab10(np.linspace(0, 0.9, n_lines))
        p = dp.p

        if x_choice == "τ (remaining time)":
            xs = tau_grid(p.T); xlabel = "τ (remaining time)"
        elif x_choice == "I₂ (inventory)":
            xs = np.arange(1, p.I2_max + 1);  xlabel = "I₂ (Retailer 2 inventory)"
        else:
            xs = np.arange(1, p.b1_max + 1);  xlabel = "b₁ (Retailer 1 backlog)"

        is_I2bar = (y_choice == "Ī₂ threshold (Case 2)")

        # Retained level = participation threshold − 1 (inventory kept after
        # dispatch). Applied to the DP curve and every analytical overlay so
        # they stay comparable.
        offset = 1.0 if retained else 0.0

        if is_I2bar:
            ys_dp = []
            for x in xs:
                tau_q = float(x) if x_choice == "τ (remaining time)" else tau_fixed
                I2_q  = int(x)   if x_choice == "I₂ (inventory)"     else I2_fixed
                b1_q  = max(0, min(p.b1_max, b1_fixed))
                n     = n_for_tau(tau_q, dp)
                th = None
                for I2t in range(1, p.I2_max + 1):
                    if dp.get_policy(n, I2t, b1_q) > 0:
                        th = I2t; break
                ys_dp.append((th - offset) if th is not None else np.nan)

            ax.plot(xs, ys_dp, color='steelblue', lw=2,
                    label="3-D DP (I₂, b₁)")
            if Cf == 0 and x_choice == "τ (remaining time)":
                # DP of the note's OWN 2-D model, for a like-for-like comparison
                # against the note's analytical staircase.
                if show_cf0_2d:
                    try:
                        cf0_vals = solve_cf0_2d(
                            float(p.T), int(n_cf0), float(lam1), float(lam2),
                            float(h), float(cu), float(pi1), float(pi2),
                            float(c2), float(v2), tuple(float(x) for x in xs),
                        ) - offset
                        ax.plot(xs, cf0_vals, color='seagreen', lw=1.8, ls='--',
                                label="2-D Cf=0 DP (note's model)")
                    except Exception as e:
                        st.warning(f"2-D Cf=0 DP failed: {e}")
                # Note eq. (20)-(22). Controlled by its own toggle in the
                # "Cf = 0 model comparison" panel, controlled only by this panel.
                if show_note_staircase:
                    exact_vals = [an_I2bar_Cf0_exact(float(x)) - offset for x in xs]
                    ax.plot(xs, exact_vals, color='crimson', lw=1.8,
                            ls='-', alpha=0.85, label="Note staircase (eq. 20-22)")

        else:
            if x_choice == "τ (remaining time)":
                vary_vals = np.linspace(1, p.I2_max, n_lines).astype(int)
                vary_label = "I₂"
            elif x_choice == "I₂ (inventory)":
                vary_vals = np.round(np.linspace(0.1*p.T, p.T, n_lines), 2)
                vary_label = "τ"
            else:
                vary_vals = np.linspace(1, p.I2_max, n_lines).astype(int)
                vary_label = "I₂"

            for vv, col in zip(vary_vals, colours):
                ys_dp = []
                for x in xs:
                    if x_choice == "τ (remaining time)":
                        tau_q = float(x); I2_q = int(vv);   b1_q = b1_fixed
                    elif x_choice == "I₂ (inventory)":
                        tau_q = float(vv); I2_q = int(x);   b1_q = b1_fixed
                    else:
                        tau_q = tau_fixed; I2_q = I2_fixed; b1_q = int(x)

                    n    = n_for_tau(tau_q, dp)
                    I2_q = max(p.I2_min, min(p.I2_max, I2_q))
                    b1_q = max(0, min(p.b1_max, b1_q))

                    if y_choice == "q* (optimal dispatch quantity)":
                        ys_dp.append(dp.get_policy(n, I2_q, b1_q))
                    elif y_choice == "b₁* threshold (Case 1)":
                        th = None
                        for b1t in range(1, min(I2_q, p.b1_max)+1):
                            if dp.get_policy(n, I2_q, b1t) > 0:
                                th = b1t; break
                        ys_dp.append(th if th is not None else np.nan)
                    else:
                        try:
                            ys_dp.append(dp.get_value(n, I2_q, b1_q))
                        except Exception:
                            ys_dp.append(np.nan)

                lbl = f"{vary_label}={vv}"
                ax.plot(xs, ys_dp, color=col, lw=2, label=f"DP  {lbl}")

        # Relabel when the Case-2 curve shows the retained level.
        y_label_txt = y_choice
        if is_I2bar and retained:
            y_label_txt = "Retained I₂  (= Ī₂ − 1)"

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(y_label_txt, fontsize=11)

        if x_choice == "τ (remaining time)":
            ax.invert_xaxis()
            ax.set_xlabel("τ  ←  end of horizon", fontsize=11)

        title_params = (f"λ₂={lam2}, cu={cu}, h={h}, Cf={Cf}, "
                        f"π₁={pi1}, π₂={pi2}, T={T}")
        ax.set_title(f"{y_label_txt}  vs  {xlabel}\n{title_params}", fontsize=10)
        ax.legend(fontsize=8, loc='best', framealpha=0.85)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Parameters used in current solve"):
            st.json(st.session_state.dp_params)


# ======================================================================
# TAB 2: 3D PLOTS  (Plotly surface)
# ======================================================================
with tab_3d:
    st.subheader("3D surface settings")
    st.caption("Drag to rotate · scroll to zoom · hover to read values")

    if dp is None:
        st.info("👈  Set parameters and press **Solve DP** to generate plots.")
    else:
        p = dp.p
        _T_f3   = float(p.T)
        _I2_max3 = p.I2_max
        _b1_max3 = p.b1_max

        c1_, c2_, c3_ = st.columns(3)
        with c1_:
            z_choice = st.selectbox(
                "Z axis (surface height)",
                ["q* (optimal dispatch quantity)",
                 "b̄₁ dispatch trigger",
                 "V^n (value function)"],
                key="z3d",
            )
        is_b1bar = z_choice.startswith("b̄₁")
        with c2_:
            if is_b1bar:
                # b1bar is a function of (I2, tau) only, so the plane is fixed.
                xy_choice = "I₂ × τ   (fixed b₁)"
                st.selectbox("X-Y plane", [xy_choice], key="xy3d_b", disabled=True)
            else:
                xy_choice = st.selectbox(
                    "X-Y plane",
                    ["I₂ × b₁  (fixed τ)",
                     "I₂ × τ   (fixed b₁)",
                     "b₁ × τ   (fixed I₂)"],
                    key="xy3d",
                )
        with c3_:
            colorscale = st.selectbox(
                "Colour scheme",
                ["Viridis", "Plasma", "RdBu", "Blues", "Cividis"],
                key="cs3d",
            )

        if is_b1bar:
            st.caption(
                "b̄₁(I₂, τ) is the smallest Retailer-1 backlog at which the DP "
                "dispatches, i.e. the dispatch trigger of the general-Cf note. "
                "It depends on I₂ and τ only, so the X-Y plane is fixed and the "
                "b₁ slider does not apply. States where the DP never dispatches "
                "(b̄₁ = +∞) are shown as gaps in the surface."
            )
            b1bar_cap = st.slider("Cap for display (states above are left blank)",
                                  5, int(_b1_max3), min(30, int(_b1_max3)),
                                  key="b1barcap")

        # fixed slider for the third dimension
        if is_b1bar:
            tau_fixed3, I2_fixed3, b1_fixed3 = None, None, None
        elif "fixed τ" in xy_choice:
            tau_fixed3 = st.slider("Fixed τ", 0.05, _T_f3, _T_f3, 0.05, key="tau3d")
            I2_fixed3, b1_fixed3 = None, None
        elif "fixed b₁" in xy_choice:
            b1_fixed3 = st.slider("Fixed b₁", 0, _b1_max3, min(5, _b1_max3), key="b13d")
            tau_fixed3, I2_fixed3 = None, None
        else:  # fixed I₂
            I2_fixed3 = st.slider("Fixed I₂", 1, _I2_max3, min(10, _I2_max3), key="i23d")
            tau_fixed3, b1_fixed3 = None, None

        # build grid
        if "I₂ × b₁" in xy_choice:
            xs = np.arange(0, _I2_max3 + 1)
            ys = np.arange(0, _b1_max3 + 1)
            x_label, y_label = "I₂", "b₁"
        elif "I₂ × τ" in xy_choice:
            xs = np.arange(0, _I2_max3 + 1)
            ys = np.linspace(0.05, _T_f3, 30)
            x_label, y_label = "I₂", "τ"
        else:
            xs = np.arange(0, _b1_max3 + 1)
            ys = np.linspace(0.05, _T_f3, 30)
            x_label, y_label = "b₁", "τ"

        # compute Z
        Z = np.zeros((len(ys), len(xs)))
        for i, yv in enumerate(ys):
            for j, xv in enumerate(xs):
                if is_b1bar:
                    # scan b1 upward for the first dispatch at this (I2, tau)
                    I2_q = max(p.I2_min, min(p.I2_max, int(xv)))
                    n = n_for_tau(float(yv), dp)
                    trig = np.nan
                    if I2_q >= 1:
                        for b1t in range(1, min(b1bar_cap, p.b1_max) + 1):
                            if dp.get_policy(n, I2_q, b1t) > 0:
                                trig = b1t
                                break
                    Z[i, j] = trig
                    continue

                if "I₂ × b₁" in xy_choice:
                    I2_q, b1_q, tau_q = int(xv), int(yv), tau_fixed3
                elif "I₂ × τ" in xy_choice:
                    I2_q, b1_q, tau_q = int(xv), b1_fixed3, float(yv)
                else:
                    I2_q, b1_q, tau_q = I2_fixed3, int(xv), float(yv)

                n    = n_for_tau(tau_q, dp)
                I2_q = max(p.I2_min, min(p.I2_max, I2_q))
                b1_q = max(0, min(p.b1_max, b1_q))

                if z_choice.startswith("q*"):
                    Z[i, j] = dp.get_policy(n, I2_q, b1_q)
                else:
                    try:
                        Z[i, j] = dp.get_value(n, I2_q, b1_q)
                    except Exception:
                        Z[i, j] = np.nan

        # plot
        fig = go.Figure(data=[go.Surface(
            x=xs, y=ys, z=Z,
            colorscale=colorscale,
            colorbar=dict(title=z_choice.split()[0]),
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>"
                          f"{z_choice.split()[0]}: %{{z:.3f}}<extra></extra>",
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_choice.split()[0],
                aspectmode='cube',
            ),
            height=650,
            margin=dict(l=0, r=0, t=30, b=0),
            title=dict(
                text=f"{z_choice}  over  {xy_choice}<br>"
                     f"<sub>λ₂={lam2}, cu={cu}, h={h}, Cf={Cf}, "
                     f"π₁={pi1}, π₂={pi2}, T={T}</sub>",
                x=0.5,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("How to interact"):
            st.markdown("""
            - **Rotate**: click and drag
            - **Zoom**: scroll wheel
            - **Pan**: right-click and drag (or ctrl + drag)
            - **Reset view**: double-click
            - **Hover**: read exact values at any point
            - **Toolbar** (top right of plot): camera presets, download as PNG
            """)