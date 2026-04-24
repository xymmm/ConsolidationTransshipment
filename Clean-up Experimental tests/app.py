"""
app.py  —  Interactive Transshipment Policy Explorer
=====================================================
Run with:
    streamlit run app.py

Requires solver.py in the same directory.
Install dependencies:
    pip install streamlit matplotlib numpy
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
from solver import Params, TransshipmentDP

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

with st.sidebar.expander("💰  Cost parameters", expanded=True):
    h   = st.slider("h  (holding cost)",     0.0, 5.0,  1.0, 0.1)
    Cf  = st.slider("Cf (fixed ship cost)",  0.0, 30.0, 8.0, 0.5)
    cu  = st.slider("cu (unit ship cost)",   0.0, 15.0, 1.0, 0.1)
    pi1 = st.slider("π₁ (R1 penalty)",       0.0, 20.0, 6.0, 0.5)
    pi2 = st.slider("π₂ (R2 penalty)",       0.0, 20.0, 6.0, 0.5)

with st.sidebar.expander("🏁  Terminal costs", expanded=False):
    c1 = st.slider("c₁ (clear R1 backlog)", 0.0, 20.0, 6.0, 0.5)
    c2 = st.slider("c₂ (clear R2 backlog)", 0.0, 20.0, 6.0, 0.5)
    v2 = st.slider("v₂ (salvage R2 inv)",   0.0, 10.0, 1.0, 0.5)

with st.sidebar.expander("📐  State space", expanded=False):
    I2_max = st.slider("I2_max", 10, 100, 40,  5)
    I2_min = st.slider("I2_min", -20, 0, -10,  1)
    b1_max = st.slider("b1_max", 10, 100, 40,  5)

# ── solve button ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
solve_btn = st.sidebar.button("▶  Solve DP", type="primary", use_container_width=True)
st.sidebar.caption("Press after changing parameters. DP may take a few seconds.")

# ======================================================================
# DIMENSIONLESS GROUPS (always shown, no solve needed)
# ======================================================================
alpha2 = lam2 * cu / (h + pi2)
phi2   = 2 * lam2 * Cf / (h + pi2) if Cf > 0 else 0.0
gamma  = lam2 * (pi1 - pi2) / (h + pi2)
beta   = round(alpha2 - 0.5)
gamT   = gamma * T

# classify region
def classify(a2, gT):
    if abs(gamma) < 1e-9:   # pi1 == pi2
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

def _sqrt(x):
    return math.sqrt(max(0.0, x))

def an_b1star(I2, tau):
    a2 = alpha2; ph = phi2
    g  = gamma
    A  = 2*I2 + 1 - 2*a2 + 2*g*tau
    D  = A**2 - 4*ph
    if D < 0 or ph == 0:
        return None
    return 0.5*(A - _sqrt(D))

def an_I2bar(tau):
    xi = 1 - 2*alpha2 + 2*gamma*tau
    if phi2 == 0:
        return lam2*cu/(h+pi2) - gamma*tau  # Cf=0 formula
    return 0.5*(_sqrt(xi**2 + 4*phi2) - xi)

def an_I2bar_Cf0(tau):
    return lam2*cu/(h+pi2) - gamma*tau

# ======================================================================
# SESSION STATE: store solved DP
# ======================================================================
if "dp" not in st.session_state:
    st.session_state.dp = None
if "dp_params" not in st.session_state:
    st.session_state.dp_params = None

if solve_btn:
    with st.spinner("Solving DP..."):
        p = Params(
            T=T, N=N, lam1=lam1, lam2=lam2,
            h=h, Cf=Cf, cu=cu, pi1=pi1, pi2=pi2,
            c1=c1, c2=c2, v2=v2,
            I2_max=I2_max, I2_min=I2_min, b1_max=b1_max,
        )
        dp = TransshipmentDP(p)
        dp.solve(store_V=False, verbose=False)
        st.session_state.dp = dp
        st.session_state.dp_params = dict(
            T=T, N=N, lam1=lam1, lam2=lam2,
            h=h, Cf=Cf, cu=cu, pi1=pi1, pi2=pi2,
            c1=c1, c2=c2, v2=v2,
            I2_max=I2_max, I2_min=I2_min, b1_max=b1_max,
        )
    st.success("DP solved!")

dp = st.session_state.dp

# ======================================================================
# PLOT CONTROLS
# ======================================================================
st.subheader("Plot settings")

pc1, pc2, pc3 = st.columns(3)
with pc1:
    x_choice = st.selectbox(
        "X axis",
        ["τ (remaining time)", "I₂ (inventory)", "b₁ (backlog)"],
    )
with pc2:
    y_choice = st.selectbox(
        "Y axis",
        ["q* (optimal dispatch quantity)",
         "b₁* threshold (Case 1)",
         "Ī₂ threshold (Case 2)",
         "V^n (value function)"],
    )
with pc3:
    show_analytical = st.checkbox("Overlay analytical formula", value=True)

# fixed-dimension sliders — use solved dp bounds if available, else sidebar
_T      = int(dp.p.T)      if dp is not None else int(T)
_T_f    = float(dp.p.T)    if dp is not None else float(T)
_I2_max = dp.p.I2_max      if dp is not None else I2_max
_I2_min = dp.p.I2_min      if dp is not None else I2_min
_b1_max = dp.p.b1_max      if dp is not None else b1_max

pc4, pc5 = st.columns(2)
with pc4:
    if x_choice != "τ (remaining time)":
        tau_fixed = st.slider("Fixed τ", 0.05, _T_f, _T_f, 0.05)
    else:
        tau_fixed = _T_f
    if x_choice != "I₂ (inventory)":
        I2_fixed = st.slider("Fixed I₂", 1, _I2_max, min(10, _I2_max))
    else:
        I2_fixed = min(10, _I2_max)
with pc5:
    if x_choice != "b₁ (backlog)":
        b1_fixed = st.slider("Fixed b₁", 1, _b1_max, min(5, _b1_max))
    else:
        b1_fixed = min(5, _b1_max)
    n_lines = st.slider("Number of curves (vary the fixed dimension)", 1, 8, 3)

# ======================================================================
# PLOT
# ======================================================================

def n_for_tau(tau, dp):
    dt = dp.p.T / dp.p.N
    return min(dp.p.N, max(1, round(tau / dt)))

if dp is None:
    st.info("👈  Set parameters and press **Solve DP** to generate plots.")
else:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colours = cm.tab10(np.linspace(0, 0.9, n_lines))
    p = dp.p

    # ── build x-axis values ──────────────────────────────────────────
    if x_choice == "τ (remaining time)":
        xs = np.linspace(0.05, p.T, 100)
        xlabel = "τ (remaining time)"
    elif x_choice == "I₂ (inventory)":
        xs = np.arange(1, p.I2_max + 1)
        xlabel = "I₂ (Retailer 2 inventory)"
    else:
        xs = np.arange(1, p.b1_max + 1)
        xlabel = "b₁ (Retailer 1 backlog)"

    # ── for Ī₂ threshold: single curve (no vary dimension needed) ──────
    # ── for other y choices: multiple curves varying one dimension ────
    is_I2bar = (y_choice == "Ī₂ threshold (Case 2)")

    if is_I2bar:
        # single DP line + single analytical line
        ys_dp, ys_an = [], []
        for x in xs:
            tau_q = float(x) if x_choice == "τ (remaining time)" else tau_fixed
            I2_q  = int(x)   if x_choice == "I₂ (inventory)"     else I2_fixed
            b1_q  = b1_fixed
            n     = n_for_tau(tau_q, dp)
            b1_q  = max(0, min(p.b1_max, b1_q))
            th = None
            for I2t in range(1, p.I2_max + 1):
                if dp.get_policy(n, I2t, b1_q) > 0:
                    th = I2t; break
            ys_dp.append(th if th is not None else np.nan)
            ys_an.append(an_I2bar(tau_q))

        ax.plot(xs, ys_dp, color='steelblue', lw=2, label="DP")
        if show_analytical:
            an_clean = [v if v is not None else np.nan for v in ys_an]
            ax.plot(xs, an_clean, color='steelblue', lw=1.2,
                    ls='--', alpha=0.7, label="Analytical (corrected)")
        if Cf == 0 and show_analytical and x_choice == "τ (remaining time)":
            cf0_vals = [an_I2bar_Cf0(float(x)) for x in xs]
            ax.plot(xs, cf0_vals, 'k:', lw=1.5, alpha=0.6,
                    label="Cf=0 formula: α₂−γτ")

    else:
        # multiple curves — vary one dimension
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
            ys_dp, ys_an = [], []
            for x in xs:
                if x_choice == "τ (remaining time)":
                    tau_q = float(x); I2_q = int(vv);    b1_q = b1_fixed
                elif x_choice == "I₂ (inventory)":
                    tau_q = float(vv); I2_q = int(x);    b1_q = b1_fixed
                else:
                    tau_q = tau_fixed; I2_q = I2_fixed;  b1_q = int(x)

                n    = n_for_tau(tau_q, dp)
                I2_q = max(p.I2_min, min(p.I2_max, I2_q))
                b1_q = max(0, min(p.b1_max, b1_q))

                if y_choice == "q* (optimal dispatch quantity)":
                    ys_dp.append(dp.get_policy(n, I2_q, b1_q))
                    ys_an.append(None)

                elif y_choice == "b₁* threshold (Case 1)":
                    th = None
                    for b1t in range(1, min(I2_q, p.b1_max)+1):
                        if dp.get_policy(n, I2_q, b1t) > 0:
                            th = b1t; break
                    ys_dp.append(th if th is not None else np.nan)
                    ys_an.append(an_b1star(I2_q, tau_q))

                else:  # value function
                    try:
                        ys_dp.append(dp.get_value(n, I2_q, b1_q))
                    except Exception:
                        ys_dp.append(np.nan)
                    ys_an.append(None)

            lbl = f"{vary_label}={vv}"
            ax.plot(xs, ys_dp, color=col, lw=2, label=f"DP  {lbl}")
            if show_analytical and any(v is not None for v in ys_an):
                an_clean = [v if v is not None else np.nan for v in ys_an]
                ax.plot(xs, an_clean, color=col, lw=1.2,
                        ls='--', alpha=0.7, label=f"Analytical  {lbl}")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(y_choice, fontsize=11)

    # invert x for tau plots
    if x_choice == "τ (remaining time)":
        ax.invert_xaxis()
        ax.set_xlabel("τ  ←  end of horizon", fontsize=11)

    title_params = (f"λ₂={lam2}, cu={cu}, h={h}, Cf={Cf}, "
                    f"π₁={pi1}, π₂={pi2}, T={T}")
    ax.set_title(f"{y_choice}  vs  {xlabel}\n{title_params}", fontsize=10)
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # ── params used for current solve ─────────────────────────────────
    with st.expander("Parameters used in current solve"):
        st.json(st.session_state.dp_params)