"""
app.py  —  Interactive Transshipment Policy Explorer
=====================================================
Run with:
    streamlit run app.py

Requires solver.py and solver_cf0_2d.py in the same directory.
Install dependencies:
    pip install streamlit matplotlib numpy plotly scipy

CHANGE LOG (b̄₁ 3-D surface)
---------------------------
The b̄₁ branch of the 3-D tab has been rewritten. The previous version
produced a misleading surface for six reasons, all fixed here.

1. The b₁ scan stopped at a display cap, so any state whose true threshold
   exceeded the cap was silently written as NaN. "Threshold above the display
   cap" and "never dispatch" became indistinguishable. The scan now covers the
   full b₁ range and the two cases are reported separately.
2. b̄₁ = +∞ was written as NaN and nothing else. A NaN hole in a Plotly surface
   looks identical to a low region once the camera is tilted, which makes the
   surface appear to RISE with I₂. The +∞ region is now marked with grey dots
   on the z = 0 floor, as in Figure 2 of the note.
3. connectgaps is now pinned to False so the +∞ region is never interpolated
   across.
4. The I₂ grid now starts at 1. b̄₁ is undefined at I₂ = 0.
5. The τ grid is now dense and clustered just above τ* = cᵤ/(h+π₁), where the
   threshold plunges from +∞ through several one-unit microsteps. A coarse
   uniform grid aliases that staircase into a single flat slab.
6. The analytic threshold of Eq. (36) can be overlaid for direct comparison,
   and diagnostic counts are printed under the figure.
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


# ── general-Cf note: analytic dispatch threshold, Eq. (36) ──────────────
def _Emin_array(mu, mmax):
    """E[min(K,m)] for m = 0..mmax with K ~ Poisson(mu). Iterative, no scipy."""
    Em = np.zeros(mmax + 1)
    pmf = math.exp(-mu)      # P(K = m-1), starting at m = 1
    cdf_below = 0.0          # P(K <= m-2)
    M = 0.0
    for m in range(1, mmax + 1):
        M += max(1.0 - (cdf_below + pmf), 0.0)   # add P(K >= m)
        Em[m] = M
        cdf_below += pmf
        pmf *= mu / m
    return Em


def b1bar_analytic_row(I2_max_, tau, lam2_, h_, pi1_, pi2_, cu_, Cf_):
    """
    Analytic dispatch threshold of the general-Cf note, Eq. (36), for every
    I₂ = 1..I2_max_ at a single τ.

        delta(m, tau) = (h+pi2)/lam2 * E[min(K,m)] - cu + (pi1-pi2)*tau,
                        K ~ Poisson(lam2*tau)
        Sigma_{I2}(b, tau) = sum_{i=0}^{b-1} delta(I2-i, tau)
        b1bar(I2, tau) = min{ b >= 1 : Sigma_{I2}(b, tau) > Cf }, else +infinity

    delta is non-decreasing in m, so the profitable levels are the top ones and
    the accumulation stops at the first non-positive margin.

    Returns an array of length I2_max_, indexed by I₂ - 1. NaN means +infinity.
    """
    out = np.full(int(I2_max_), np.nan)
    if tau <= 0 or I2_max_ < 1:
        return out
    mu = lam2_ * tau
    Em = _Emin_array(mu, int(I2_max_))
    d = (h_ + pi2_) / max(lam2_, 1e-12) * Em - cu_ + (pi1_ - pi2_) * tau
    pos = np.where(d[1:] > 0.0)[0]
    if pos.size == 0:
        return out                       # every margin is non-positive
    m_c = int(pos[0]) + 1                # critical level
    P = np.concatenate(([0.0], np.cumsum(d[1:])))   # P[m] = sum_{j<=m} d[j]
    for I2 in range(m_c, int(I2_max_) + 1):
        # b = I2 - k, so b = 1..(I2 - m_c + 1) corresponds to k = I2-1..m_c-1
        ks = np.arange(I2 - 1, m_c - 2, -1)
        vals = P[I2] - P[ks]
        hit = vals > Cf_
        if hit.any():
            out[I2 - 1] = float(int(np.argmax(hit)) + 1)
    return out


def b1bar_dp_row(dp_, n):
    """
    b̄₁ from the DP for every I₂ = 1..I2_max at period index n, computed as the
    smallest b₁ >= 1 at which the optimal action dispatches. The scan covers the
    FULL b₁ range of the solver. NaN means the DP never dispatches at any b₁.
    """
    p_ = dp_.p
    pol = dp_.policy[n]                              # (nI2, nb1)
    lo = 1 - p_.I2_min
    hi = p_.I2_max - p_.I2_min + 1
    sub = pol[lo:hi, 1:] > 0                         # I₂ = 1..I2_max, b₁ >= 1
    any_ = sub.any(axis=1)
    first = sub.argmax(axis=1) + 1
    return np.where(any_, first.astype(float), np.nan)

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
# b̄₁ SURFACE RENDERER  (general-Cf note, Figure 2 style)
# ======================================================================
def render_b1bar_surface(dp_, colorscale_):
    p_ = dp_.p
    tau_star = cu / max(h + pi1, 1e-9)

    st.caption(
        "b̄₁(I₂, τ) is the smallest Retailer-1 backlog at which a dispatch "
        "occurs. It depends on I₂ and τ only, so the X-Y plane is fixed and the "
        "b₁ slider does not apply. States where dispatch is never worthwhile "
        "have b̄₁ = +∞. They are drawn as grey dots on the z = 0 floor, exactly "
        "as in Figure 2 of the note, and are NOT part of the surface."
    )

    if not (c1 == 0.0 and c2 == 0.0 and v2 == 0.0):
        st.warning(
            "Figure 2 of the note assumes V(I₂, b₁, 0) = 0. Set c₁ = c₂ = v₂ = 0 "
            "in the sidebar, otherwise the DP threshold is not comparable with "
            "the analytic threshold of Eq. (36)."
        )

    st.info(
        f"τ* = cᵤ/(h+π₁) = {tau_star:.4f}. For τ ≤ τ* the note proves "
        f"b̄₁ = +∞ at every I₂. The whole staircase lives just above τ*, so the "
        f"τ grid is densified there."
    )

    cA, cB, cC = st.columns(3)
    with cA:
        show_analytic = st.checkbox("Overlay analytic b̄₁ (Eq. 36)",
                                   value=True, key="b1bar_an")
    with cB:
        z_cap = st.slider(
            "z-axis cap (display only, the b₁ scan is never truncated)",
            5, max(10, int(p_.b1_max)), min(30, int(p_.b1_max)),
            key="b1bar_zcap")
    with cC:
        n_tau_pts = st.select_slider("τ resolution",
                                     options=[40, 80, 120, 200],
                                     value=120, key="b1bar_taures")

    # ── grids ──────────────────────────────────────────────────────────
    # I₂ starts at 1. b̄₁ is undefined at I₂ = 0.
    xs = np.arange(1, p_.I2_max + 1)
    base = np.linspace(0.02, float(p_.T), int(n_tau_pts))
    lo = max(0.02, tau_star - 0.05)
    hi = min(float(p_.T), tau_star + 0.8)
    dense = np.linspace(lo, hi, int(n_tau_pts)) if hi > lo else np.array([])
    ys = np.unique(np.concatenate([base, dense]))

    # ── compute ────────────────────────────────────────────────────────
    with st.spinner("Building the b̄₁ surface..."):
        Z_dp = np.full((len(ys), len(xs)), np.nan)
        Z_an = np.full((len(ys), len(xs)), np.nan)
        for i, tv in enumerate(ys):
            n = n_for_tau(float(tv), dp_)
            Z_dp[i, :] = b1bar_dp_row(dp_, n)
            if show_analytic:
                Z_an[i, :] = b1bar_analytic_row(
                    p_.I2_max, float(tv), lam2, h, pi1, pi2, cu, Cf)

    # +infinity and "finite but above the display cap" are different things
    inf_mask = np.isnan(Z_dp)
    over_cap = (~inf_mask) & (Z_dp > z_cap)
    Z_dp_plot = np.where(over_cap, np.nan, Z_dp)

    XX, YY = np.meshgrid(xs, ys)

    data = [go.Surface(
        x=xs, y=ys, z=Z_dp_plot,
        colorscale=colorscale_,
        connectgaps=False,                 # never interpolate across +infinity
        cmin=1, cmax=float(z_cap),
        colorbar=dict(title="b̄₁ (DP)", len=0.6),
        name="DP",
        hovertemplate="I₂: %{x}<br>τ: %{y:.3f}<br>"
                      "b̄₁ (DP): %{z:.0f}<extra></extra>",
    )]

    if show_analytic:
        Z_an_plot = np.where((~np.isnan(Z_an)) & (Z_an > z_cap), np.nan, Z_an)
        data.append(go.Surface(
            x=xs, y=ys, z=Z_an_plot,
            colorscale="Greys", showscale=False, opacity=0.45,
            connectgaps=False,
            name="analytic (Eq. 36)",
            hovertemplate="I₂: %{x}<br>τ: %{y:.3f}<br>"
                          "b̄₁ (analytic): %{z:.0f}<extra></extra>",
        ))

    if inf_mask.any():
        data.append(go.Scatter3d(
            x=XX[inf_mask], y=YY[inf_mask],
            z=np.zeros(int(inf_mask.sum())),
            mode="markers",
            marker=dict(size=1.6, color="lightgrey"),
            name="b̄₁ = +∞",
            hovertemplate="I₂: %{x}<br>τ: %{y:.3f}<br>b̄₁ = +∞<extra></extra>",
        ))

    if over_cap.any():
        data.append(go.Scatter3d(
            x=XX[over_cap], y=YY[over_cap],
            z=np.full(int(over_cap.sum()), float(z_cap)),
            mode="markers",
            marker=dict(size=1.8, color="crimson"),
            name=f"b̄₁ > {z_cap} (finite, above cap)",
            hovertemplate="I₂: %{x}<br>τ: %{y:.3f}<br>"
                          "b̄₁ above the display cap<extra></extra>",
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis_title="I₂",
            yaxis_title="τ",
            zaxis=dict(title="b̄₁", range=[0, float(z_cap) * 1.05]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.7, y=-1.7, z=1.0)),
        ),
        height=700,
        showlegend=True,
        margin=dict(l=0, r=0, t=60, b=0),
        title=dict(
            text="Dispatch threshold b̄₁(I₂, τ)<br>"
                 f"<sub>λ₁={lam1}, λ₂={lam2}, h={h}, π₁={pi1}, π₂={pi2}, "
                 f"Cf={Cf}, cᵤ={cu}, T={p_.T}, N={p_.N} · "
                 f"grey floor dots = +∞</sub>",
            x=0.5,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── diagnostics printed under the figure ───────────────────────────
    fin = ~np.isnan(Z_dp)
    bad_I2 = 0
    for i in range(Z_dp.shape[0]):
        row = Z_dp[i]
        for j in range(len(xs) - 1):
            a, b = row[j], row[j + 1]
            if not np.isnan(a) and not np.isnan(b) and b > a:
                bad_I2 += 1
            elif np.isnan(b) and not np.isnan(a):
                bad_I2 += 1      # finite then +infinity as I₂ grows also violates
    bad_tau = 0
    for j in range(Z_dp.shape[1]):
        col = Z_dp[:, j]
        for i in range(len(ys) - 1):
            a, b = col[i], col[i + 1]
            if not np.isnan(a) and not np.isnan(b) and b > a:
                bad_tau += 1
            elif np.isnan(b) and not np.isnan(a):
                bad_tau += 1

    msgs = [f"finite cells: {int(fin.sum())} / {Z_dp.size}",
            f"+∞ cells: {int(inf_mask.sum())}",
            f"above display cap: {int(over_cap.sum())}",
            f"monotonicity violations in I₂: {bad_I2}",
            f"monotonicity violations in τ: {bad_tau}"]
    if show_analytic:
        both = fin & (~np.isnan(Z_an))
        if both.any():
            gap = Z_dp[both] - Z_an[both]
            msgs.append(f"DP − analytic on common finite cells: "
                        f"min {gap.min():.0f}, median {np.median(gap):.0f}, "
                        f"max {gap.max():.0f}")
        only_one = int((fin ^ (~np.isnan(Z_an))).sum())
        msgs.append(f"cells where exactly one of the two is +∞: {only_one}")
    st.caption(" · ".join(msgs))

    with st.expander("How to read this figure"):
        st.markdown(
            "- b̄₁ must be **non-increasing in I₂** and **non-increasing in τ**, "
            "so the surface descends from the small-τ edge down to the floor "
            "value 1.\n"
            "- If the surface ever appears to **rise with I₂**, that is the +∞ "
            "region being drawn at floor height instead of masked. Check the "
            "grey dots: they mark where b̄₁ = +∞, and the surface must have a "
            "hole there.\n"
            "- Grey floor dots and red cap dots are different. Grey means "
            "dispatch is never worthwhile. Red means the threshold is finite "
            "but taller than the z-axis cap. Raise the cap to see those cells.\n"
            "- Near the trigger boundary the cost advantage of dispatching is "
            "O(Δt), so b̄₁ from the DP can wobble by one unit. Raise N before "
            "reading anything into a one-unit step.\n"
            "- To reproduce Figure 2 of the note, set c₁ = c₂ = v₂ = 0 and use "
            "the Section 6.2 parameters λ₁=5, λ₂=3, h=1, π₁=π₂=6, Cf=8, cᵤ=1, "
            "T=5, with N=800."
        )


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

        # ── b̄₁ has its own renderer, see render_b1bar_surface above ────
        if is_b1bar:
            render_b1bar_surface(dp, colorscale)

        else:
            # fixed slider for the third dimension
            if "fixed τ" in xy_choice:
                tau_fixed3 = st.slider("Fixed τ", 0.05, _T_f3, _T_f3, 0.05,
                                       key="tau3d")
                I2_fixed3, b1_fixed3 = None, None
            elif "fixed b₁" in xy_choice:
                b1_fixed3 = st.slider("Fixed b₁", 0, _b1_max3, min(5, _b1_max3),
                                      key="b13d")
                tau_fixed3, I2_fixed3 = None, None
            else:  # fixed I₂
                I2_fixed3 = st.slider("Fixed I₂", 1, _I2_max3, min(10, _I2_max3),
                                      key="i23d")
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
                connectgaps=False,
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