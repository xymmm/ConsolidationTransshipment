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
import pandas as pd
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


@st.cache_resource(show_spinner=False)
def solve_cf0_2d_policy(T_, N_, lam1_, lam2_, h_, cu_, pi1_, pi2_, c2_, v2_):
    """
    The note's 2-D Cf=0 switching model, returned as a solved object so the
    Simulation tab can query its policy. At Cf = 0 this, not solver.py, is
    the right model: there is no b1 state, a rejected Retailer-1 demand is
    charged pi1*tau once and is never served afterwards, and the flow cost
    carries no pi1*b1 term.
    """
    p0 = ParamsCf0(T=T_, N=N_, lam1=lam1_, lam2=lam2_, h=h_, cu=cu_,
                   pi1=pi1_, pi2=pi2_, c2=c2_, v2=v2_).with_auto_bounds()
    dp0 = SwitchingDPCf0(p0)
    dp0.solve(verbose=False)
    return dp0

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


def dispatch_margin(dp_, n):
    """
    wait_cost - min_q dispatch_cost at period index n, for every (I2, b1).

    A strictly positive value means dispatching is strictly better. A value of
    zero means the DP is exactly indifferent, and then the reported threshold
    depends entirely on the tie-breaking convention. solver.py compares with a
    strict '<' and evaluates q = 0 first, so ties are awarded to waiting.

    Returns the margin array shaped like the state grid, or None when the DP
    was solved without store_V=True.
    """
    p_ = dp_.p
    if dp_.V_all is None or n < 1 or n > p_.N:
        return None
    V = dp_.V_all[n - 1]
    dt = p_.dt
    I2v = np.arange(p_.I2_min, p_.I2_max + 1)
    I2g = I2v[:, None]
    b1g = np.arange(0, p_.b1_max + 1)[None, :]
    sh = (len(I2v), p_.b1_max + 1)
    cI = lambda x: np.clip(x, p_.I2_min, p_.I2_max) - p_.I2_min
    cB = lambda x: np.clip(x, 0, p_.b1_max)

    def branch(q):
        I2a = np.broadcast_to(I2g - q, sh)
        b1a = np.broadcast_to(b1g - q, sh)
        g = (p_.Cf if q > 0 else 0.0) + p_.cu * q + dt * (
            p_.h * np.maximum(0, I2a) + p_.pi1 * b1a
            + p_.pi2 * np.maximum(0, -I2a))
        return (g + p_.p0 * V[cI(I2a), cB(b1a)]
                  + p_.p1 * V[cI(I2a), cB(b1a + 1)]
                  + p_.p2 * V[cI(I2a - 1), cB(b1a)])

    wait = branch(0)
    best = np.full(sh, np.inf)
    qmax = max(1, min(p_.I2_max, p_.b1_max))
    for q in range(1, qmax + 1):
        feas = (I2g >= q) & (b1g >= q)
        if not feas.any():
            continue
        best = np.minimum(best, np.where(feas, branch(q), np.inf))
    return wait - best


def b1bar_dp_row_tie(dp_, n, tol, prefer_dispatch):
    """
    b̄₁ for every I₂ = 1..I2_max, computed from the exact dispatch margin so
    that the tie-breaking convention is explicit.

    Returns (b1bar, tie_at_b1bar). tie_at_b1bar flags the I₂ values whose
    threshold is decided by a tie rather than by a strict cost advantage.
    """
    p_ = dp_.p
    M = dispatch_margin(dp_, n)
    if M is None:
        return b1bar_dp_row(dp_, n), np.zeros(p_.I2_max, bool)
    lo = 1 - p_.I2_min
    hi = p_.I2_max - p_.I2_min + 1
    sub = M[lo:hi, 1:]                               # I₂ = 1..I2_max, b₁ >= 1
    disp = sub >= -tol if prefer_dispatch else sub > tol
    any_ = disp.any(axis=1)
    first = disp.argmax(axis=1)
    margin_at = np.take_along_axis(sub, first[:, None], axis=1).ravel()
    out = np.where(any_, (first + 1).astype(float), np.nan)
    tie = any_ & (np.abs(margin_at) <= tol)
    return out, tie

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

    cD, cE = st.columns(2)
    with cD:
        inf_mode = st.radio(
            "How to draw b̄₁ = +∞",
            ["Spikes at the cap", "Grey dots on the floor", "Both"],
            index=0, key="b1bar_infmode", horizontal=False)
        st.caption(
            "Spikes render +∞ at the top of the z axis, which keeps the "
            "surface visually monotone. Floor dots reproduce Figure 2 of the "
            "note. Neither is part of the fitted surface."
        )
    with cE:
        tie_rule = st.radio(
            "Tie-breaking when dispatch and wait cost the same",
            ["Prefer dispatch", "Prefer wait (solver.py default)"],
            index=0, key="b1bar_tie", horizontal=False)
        st.caption(
            "At the trigger boundary the cost advantage of dispatching is "
            "O(Δt), and at some states it is exactly zero. The reported b̄₁ "
            "then depends only on the convention. solver.py compares with a "
            "strict '<' and evaluates q = 0 first, so it awards ties to "
            "waiting. States decided by a tie are marked in orange."
        )
    prefer_dispatch = tie_rule.startswith("Prefer dispatch")
    tie_tol = 1e-9

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
        TIE = np.zeros((len(ys), len(xs)), bool)
        for i, tv in enumerate(ys):
            n = n_for_tau(float(tv), dp_)
            row, tie = b1bar_dp_row_tie(dp_, n, tie_tol, prefer_dispatch)
            Z_dp[i, :] = row
            TIE[i, :] = tie
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
        n_inf = int(inf_mask.sum())
        top = float(z_cap)
        if inf_mode in ("Spikes at the cap", "Both"):
            # one vertical segment per +infinity cell, drawn as a single trace
            sx, sy, sz = [], [], []
            for xv, yv in zip(XX[inf_mask], YY[inf_mask]):
                sx += [xv, xv, None]
                sy += [yv, yv, None]
                sz += [0.0, top, None]
            data.append(go.Scatter3d(
                x=sx, y=sy, z=sz, mode="lines",
                line=dict(color="rgba(120,120,120,0.55)", width=2),
                name="b̄₁ = +∞ (spike to cap)",
                hoverinfo="skip",
            ))
            data.append(go.Scatter3d(
                x=XX[inf_mask], y=YY[inf_mask], z=np.full(n_inf, top),
                mode="markers",
                marker=dict(size=2.4, color="dimgrey", symbol="diamond"),
                name="b̄₁ = +∞ (cap)",
                hovertemplate="I₂: %{x}<br>τ: %{y:.3f}<br>"
                              "b̄₁ = +∞<extra></extra>",
            ))
        if inf_mode in ("Grey dots on the floor", "Both"):
            data.append(go.Scatter3d(
                x=XX[inf_mask], y=YY[inf_mask], z=np.zeros(n_inf),
                mode="markers",
                marker=dict(size=2.6, color="grey"),
                name="b̄₁ = +∞ (floor, Figure 2 style)",
                hovertemplate="I₂: %{x}<br>τ: %{y:.3f}<br>"
                              "b̄₁ = +∞<extra></extra>",
            ))

    tie_show = TIE & (~inf_mask) & (~over_cap)
    if tie_show.any():
        data.append(go.Scatter3d(
            x=XX[tie_show], y=YY[tie_show], z=Z_dp[tie_show],
            mode="markers",
            marker=dict(size=2.6, color="darkorange"),
            name="threshold decided by an exact tie",
            hovertemplate="I₂: %{x}<br>τ: %{y:.3f}<br>"
                          "b̄₁ = %{z:.0f}, exact tie<extra></extra>",
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
            f"cells decided by an exact tie: {int(TIE.sum())}",
            f"monotonicity violations in I₂: {bad_I2}",
            f"monotonicity violations in τ: {bad_tau}"]
    if bad_I2 > 0 and int(TIE.sum()) > 0:
        msgs.append("try the other tie-breaking rule before treating a "
                    "one-unit ridge as structural")
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
            "- The analytic threshold of Eq. (36) is **non-increasing in I₂ "
            "and in τ**, by Theorems 3 and 4. Its surface descends from the "
            "small-τ edge down to the floor value 1.\n"
            "- The **DP threshold need not be monotone**. In the Section 6.2 "
            "instance it is not: at τ ≥ 1 it reads ∞, ∞, 3, 3, 3, 4, 4, 3, 3, "
            "3 for I₂ = 1..10, so a one-unit ridge appears at I₂ = 6 and 7. "
            "That ridge is stable at N = 200 to 6400 and under four different "
            "state-space bounds, and the cost margin behind it is about −0.39, "
            "which is far too large to be numerical. Theorems 3 and 4 apply to "
            "the analytic bound, not to the optimal threshold.\n"
            "- Before treating any ridge as structural, check the two "
            "diagnostics under the figure. A ridge that sits on orange tie "
            "markers, or that moves when the tie-breaking rule is switched, is "
            "a convention artefact. A ridge that survives both is real.\n"
            "- Three marker types appear. Grey means b̄₁ = +∞, that is, "
            "dispatch is never worthwhile. Red means the threshold is finite "
            "but taller than the z-axis cap, so raise the cap. Orange means "
            "the threshold at that cell was decided by an exact tie.\n"
            "- To reproduce Figure 2 of the note, set c₁ = c₂ = v₂ = 0 and use "
            "the Section 6.2 parameters λ₁=5, λ₂=3, h=1, π₁=π₂=6, Cf=8, cᵤ=1, "
            "T=5, with N=800."
        )


# ======================================================================
# TABS:  2D PLOTS  /  3D PLOTS
# ======================================================================
tab_2d, tab_3d, tab_q, tab_pol, tab_sim = st.tabs(
    ["📈 2D Plots", "🧊 3D Plots", "🔍 Inspector", "🗺 Policy Table", "🎬 Simulation"])

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
                        for b1t in range(1, p.b1_max + 1):  # FULL scan; capping at I2 silently hides thresholds > I2
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

# ======================================================================
# TAB 3: INSPECTOR — exact-τ threshold curve & action-value comparison
# ======================================================================
with tab_q:
    if dp is None:
        st.info("👈  Set parameters and press **Solve DP** to use the inspector.")
    else:
        p = dp.p

        st.subheader("b̄₁ vs I₂ at exact τ values")
        st.caption(
            "Type any τ values (comma-separated). Each τ is mapped to the "
            "nearest DP period n and the effective τ = n·Δt is reported, so "
            "you always know exactly which slice you are looking at. The DP "
            "scan covers the FULL b₁ range. Dashed lines are the analytic "
            "threshold of Eq. (36) at the same effective τ."
        )
        tau_text = st.text_input("τ values", value="1.0, 2.0, 5.0",
                                 key="insp_taus")
        show_an_i = st.checkbox("Overlay analytic Eq. (36)", value=True,
                                key="insp_an")
        try:
            tau_list = [float(s) for s in tau_text.replace("，", ",").split(",")
                        if s.strip()]
        except ValueError:
            tau_list = []
            st.error("Could not parse the τ list.")
        tau_list = [tv for tv in tau_list if 0 < tv <= p.T]

        if tau_list:
            figq, axq = plt.subplots(figsize=(10, 5))
            cols = cm.tab10(np.linspace(0, 0.9, max(len(tau_list), 2)))
            xsI = np.arange(1, p.I2_max + 1)
            for tv, col in zip(tau_list, cols):
                n = n_for_tau(float(tv), dp)
                te = n * p.T / p.N
                ys = b1bar_dp_row(dp, n)
                axq.step(xsI, ys, where="mid", color=col, lw=2,
                         label=f"DP  τ={tv:g} (eff {te:.4g})")
                if show_an_i:
                    ya = b1bar_analytic_row(p.I2_max, te, lam2, h, pi1, pi2,
                                            cu, Cf)
                    axq.step(xsI, ya, where="mid", color=col, lw=1.4,
                             ls="--", alpha=0.7,
                             label=f"analytic τ={te:.4g}")
            axq.set_xlabel("I₂"); axq.set_ylabel("b̄₁(I₂, τ)")
            axq.set_title("Dispatch threshold vs I₂ at exact τ\n"
                          "(missing points = b̄₁ = +∞, dispatch never pays)",
                          fontsize=10)
            axq.legend(fontsize=8); axq.grid(True, alpha=0.3)
            st.pyplot(figq); plt.close(figq)

        st.markdown("---")
        st.subheader("Action values: dispatch vs wait at one state")
        st.caption(
            "Q(q) = immediate cost of action q + expected cost-to-go under "
            "OPTIMAL play afterwards. Q(0) is waiting. The margin "
            "Q(wait) − minₖ Q(q) is positive exactly where dispatching now "
            "is strictly cheaper. Both branches share the same optimal "
            "continuation, so this is the like-for-like comparison of the "
            "two actions."
        )
        cq1, cq2, cq3 = st.columns(3)
        with cq1:
            tau_q = st.number_input("τ", min_value=float(p.T / p.N),
                                    max_value=float(p.T), value=float(p.T),
                                    step=0.05, format="%.4f", key="insp_tauq")
        with cq2:
            I2_q = st.number_input("I₂", min_value=1, max_value=int(p.I2_max),
                                   value=min(6, p.I2_max), key="insp_i2")
        with cq3:
            b1_q = st.number_input("b₁", min_value=1, max_value=int(p.b1_max),
                                   value=min(3, p.b1_max), key="insp_b1")

        if dp.V_all is None:
            st.warning("Re-solve with store_V=True to use action values.")
        else:
            n = n_for_tau(float(tau_q), dp)
            te = n * p.T / p.N
            av = dp.action_values(n, int(I2_q), int(b1_q))
            qs = [q for q, _ in av]; vs = [v for _, v in av]
            best = int(np.argmin(vs))
            figb, axb = plt.subplots(figsize=(8, 3.6))
            bars = axb.bar([str(q) if q else "wait" for q in qs], vs,
                           color=["#B03A2E" if i == best else "#5D6D7E"
                                  for i in range(len(qs))])
            lo, hi = min(vs), max(vs)
            axb.set_ylim(lo - 0.15 * (hi - lo + 1e-9),
                         hi + 0.10 * (hi - lo + 1e-9))
            for b, v in zip(bars, vs):
                axb.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=8)
            axb.set_xlabel("action q"); axb.set_ylabel("Q(q)")
            axb.set_title(f"Action values at (I₂={int(I2_q)}, b₁={int(b1_q)}, "
                          f"τ_eff={te:.4g})   red = optimal", fontsize=10)
            st.pyplot(figb); plt.close(figb)
            m = dp.wait_margin(n, int(I2_q), int(b1_q))
            if np.isnan(m):
                st.info("No dispatch action is feasible at this state.")
            elif m > 0:
                st.success(f"Q(wait) − best dispatch = **+{m:.4f}** → "
                           f"dispatching now is strictly cheaper.")
            else:
                st.info(f"Q(wait) − best dispatch = **{m:.4f}** → "
                        f"waiting is strictly cheaper.")

            st.markdown("**Dispatch-vs-wait margin.** Q(wait) − best "
                        "dispatch, at this b₁ and τ. Positive means "
                        "dispatching now is cheaper. **A dip below zero "
                        "between positive neighbours is the crease: there "
                        "waiting is strictly cheaper, so b̄₁ steps up.** "
                        "Sweep along I₂ or along τ.")
            sweep = st.radio("sweep along", ["I₂ (fixed τ)", "τ (fixed I₂)"],
                             horizontal=True, key="insp_sweep")

            if sweep.startswith("I₂"):
                xsm = np.arange(1, p.I2_max + 1)
                ms = np.array([dp.wait_margin(n, int(x), int(b1_q))
                               for x in xsm], dtype=float)
                xlabel = "I₂"
                ttl = (f"margin vs I₂ at b₁={int(b1_q)}, τ_eff={te:.4g}")
            else:
                ns = np.unique(np.linspace(1, p.N, 160).astype(int))
                xsm = ns * p.T / p.N
                ms = np.array([dp.wait_margin(int(nn), int(I2_q), int(b1_q))
                               for nn in ns], dtype=float)
                xlabel = "τ"
                ttl = (f"margin vs τ at I₂={int(I2_q)}, b₁={int(b1_q)}")

            figm, axm = plt.subplots(figsize=(10, 3.9))
            axm.axhline(0, color="0.35", lw=1.0)
            axm.plot(xsm, ms, marker="o", ms=3.2, lw=1.6, color="#1F618D",
                     zorder=3)
            # highlight every contiguous run where waiting wins
            neg = np.isfinite(ms) & (ms < 0)
            runs, i0 = [], None
            for k, v in enumerate(neg):
                if v and i0 is None:
                    i0 = k
                elif not v and i0 is not None:
                    runs.append((i0, k - 1)); i0 = None
            if i0 is not None:
                runs.append((i0, len(neg) - 1))
            for (s0, s1) in runs:
                lo = xsm[s0] - (xsm[1] - xsm[0]) * 0.5 if s0 > 0 else xsm[0]
                hi = xsm[s1] + (xsm[1] - xsm[0]) * 0.5 \
                    if s1 < len(xsm) - 1 else xsm[-1]
                axm.axvspan(lo, hi, color="#F5B041", alpha=0.30, zorder=1)
                kmin = s0 + int(np.argmin(ms[s0:s1 + 1]))
                axm.annotate(f"wait wins by {abs(ms[kmin]):.3f}",
                             xy=(xsm[kmin], ms[kmin]),
                             xytext=(0, -22), textcoords="offset points",
                             ha="center", fontsize=8, color="#7E5109",
                             arrowprops=dict(arrowstyle="->", lw=0.9,
                                             color="#7E5109"))
            axm.fill_between(xsm, 0, ms, where=np.isfinite(ms) & (ms >= 0),
                             color="#1F618D", alpha=0.10, zorder=0)
            axm.set_xlabel(xlabel)
            axm.set_ylabel("Q(wait) − best dispatch")
            axm.set_title(ttl + "   (>0: dispatch now is cheaper; "
                          "orange band: waiting is cheaper)", fontsize=10)
            axm.grid(True, alpha=0.3)
            st.pyplot(figm); plt.close(figm)
            if runs:
                spans = ", ".join(
                    f"{xlabel}∈[{xsm[s0]:.4g}, {xsm[s1]:.4g}]"
                    for s0, s1 in runs)
                st.warning(f"Waiting is strictly cheaper on {spans}. "
                           f"These are the crease cells: b̄₁ steps up here "
                           f"even though the neighbours dispatch.")
            else:
                st.caption("No waiting pocket on this sweep.")


# ======================================================================
# SHARED HELPERS for Policy Table & Simulation
# ======================================================================
def _an_q_exact(I2, b1, tau):
    """Analytic q* of Eq. (33)-(36) at exact continuous tau. 0 = wait."""
    if I2 < 1 or b1 < 1 or tau <= 0:
        return 0
    mu = lam2 * tau
    E = 0.0
    pmf = math.exp(-mu)
    cdf = pmf
    Em = [0.0]
    for k in range(1, I2 + 1):
        E += 1.0 - cdf
        Em.append(E)
        pmf *= mu / k
        cdf += pmf
    d = lambda m: (h + pi2) / lam2 * Em[m] - cu + (pi1 - pi2) * tau
    Np = sum(1 for m in range(1, I2 + 1) if d(m) > 0)
    if Np == 0:
        return 0
    qc = min(b1, Np)
    S = sum(d(I2 - i) for i in range(qc))
    return qc if S > Cf else 0




def _q_shade(vmax):
    """Cell background scaling with q: white for wait, light->deep blue for q."""
    def f(v):
        try:
            q = int(v)
        except (TypeError, ValueError):
            return ""                      # '·' or 'a/b' cells
        if q <= 0 or vmax <= 0:
            return ""
        x = min(q / vmax, 1.0)
        # interpolate white -> steel blue
        r = int(255 - x * (255 - 70))
        g = int(255 - x * (255 - 130))
        b = int(255 - x * (255 - 180))
        fg = "#FFFFFF" if x > 0.55 else "#1B2631"
        return f"background-color:rgb({r},{g},{b});color:{fg}"
    return f

# ======================================================================
# TAB 4: POLICY TABLE — the roadmap: full q*(I2, b1) at a chosen tau
# ======================================================================
with tab_pol:
    if dp is None:
        st.info("👈  Set parameters and press **Solve DP** first.")
    else:
        p = dp.p
        st.subheader("Optimal policy table q*(I₂, b₁) — the roadmap")
        st.caption(
            "The complete rule book of the DP solution for this parameter "
            "set. Rows are I₂, columns are b₁. A number is the optimal "
            "dispatch quantity; '·' means wait. Every decision taken in the "
            "Simulation tab is a lookup into this table."
        )
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            tau_p = st.number_input("τ", min_value=float(p.T / p.N),
                                    max_value=float(p.T), value=float(p.T),
                                    step=0.05, format="%.4f", key="pol_tau")
        with cp2:
            b1_show = st.number_input("show b₁ up to", min_value=5,
                                      max_value=int(p.b1_max),
                                      value=min(20, p.b1_max), key="pol_b1c")
        with cp3:
            view_mode = st.selectbox(
                "cell content", ["q*", "q* with analytic diff",
                                 "wait margin"], key="pol_view")

        n = n_for_tau(float(tau_p), dp)
        te = n * p.T / p.N
        st.caption(f"effective τ = n·Δt = {te:.4f}  (n = {n})")

        I2rows = list(range(1, p.I2_max + 1))
        b1cols = list(range(1, int(b1_show) + 1))
        st.markdown("**rows = I₂ (Retailer-2 inventory) · "
                    "columns = b₁ (Retailer-1 backlog)**")

        def _label(df):
            df.index.name = "I₂ \\ b₁"
            df.columns.name = "b₁"
            return df

        if view_mode == "wait margin":
            if dp.V_all is None:
                st.warning("Re-solve with store_V=True for margins.")
                data = None
            else:
                data = [[round(dp.wait_margin(n, I2, b1), 3)
                         for b1 in b1cols] for I2 in I2rows]
                df = _label(pd.DataFrame(data, index=I2rows,
                                         columns=b1cols))
                st.caption("Q(wait) − best dispatch. Positive = dispatch "
                           "region. A negative pocket between positive "
                           "neighbours is the crease.")
                st.dataframe(
                    df.style.map(
                        lambda v: "background-color:#FDEBD0"
                        if isinstance(v, float) and v < 0 else "")
                    .format("{:.3f}"),
                    height=560)
        else:
            qs = [[dp.get_policy(n, I2, b1) for b1 in b1cols]
                  for I2 in I2rows]
            if view_mode == "q*":
                df = _label(pd.DataFrame(
                    [["·" if q == 0 else str(q) for q in row] for row in qs],
                    index=I2rows, columns=b1cols))
                vmax = max((max(r) for r in qs), default=0)
                st.dataframe(df.style.map(_q_shade(vmax)), height=560)
                st.caption("cell shade scales with q*: white = wait, "
                           "deeper blue = larger dispatch")
            else:
                cells, marks = [], []
                for I2, row in zip(I2rows, qs):
                    crow, mrow = [], []
                    for b1, qd in zip(b1cols, row):
                        qa = _an_q_exact(I2, b1, te)
                        if qa == qd:
                            crow.append("·" if qd == 0 else str(qd))
                            mrow.append(False)
                        else:
                            crow.append(f"{qd if qd else '·'}/{qa if qa else '·'}")
                            mrow.append(True)
                    cells.append(crow); marks.append(mrow)
                df = _label(pd.DataFrame(cells, index=I2rows,
                                         columns=b1cols))
                mk = pd.DataFrame(marks, index=I2rows, columns=b1cols)
                st.caption("cell = DP/analytic where they differ (orange); "
                           "a single number where they agree, shaded by q*. "
                           "'·' = wait.")
                vmax = max((max(r) for r in qs), default=0)
                shade = _q_shade(vmax)
                st.dataframe(
                    df.style.apply(
                        lambda col: ["background-color:#F5B041"
                                     if m else shade(v)
                                     for m, v in zip(mk[col.name], col)],
                        axis=0),
                    height=560)
                ndis = int(mk.values.sum())
                st.caption(f"disagreements in this slice: {ndis} "
                           f"of {mk.size} cells")

        # ── exports ────────────────────────────────────────────────
        ce1, ce2 = st.columns(2)
        with ce1:
            qs_full = [[dp.get_policy(n, I2, b1)
                        for b1 in range(1, p.b1_max + 1)]
                       for I2 in I2rows]
            slice_df = pd.DataFrame(qs_full, index=I2rows,
                                    columns=range(1, p.b1_max + 1))
            slice_df.index.name = "I2"
            st.download_button(
                f"Download this τ slice (CSV)",
                slice_df.to_csv().encode(),
                file_name=f"policy_tau{te:.4f}.csv", mime="text/csv")
        with ce2:
            if st.button("Prepare full policy (all τ) as CSV", key="pol_full"):
                recs = []
                dtp = p.T / p.N
                for nn in range(1, p.N + 1):
                    pol_n = dp.policy[nn]
                    for I2 in I2rows:
                        row = pol_n[I2 - p.I2_min, 1:p.b1_max + 1]
                        nz = np.nonzero(row)[0]
                        for j in nz:
                            recs.append((round(nn * dtp, 6), I2,
                                         int(j) + 1, int(row[j])))
                full_df = pd.DataFrame(
                    recs, columns=["tau", "I2", "b1", "q_star"])
                st.caption(f"{len(full_df):,} dispatch cells "
                           "(wait cells omitted — every state not listed "
                           "is a wait).")
                st.download_button(
                    "Download full policy (CSV)",
                    full_df.to_csv(index=False).encode(),
                    file_name="policy_full.csv", mime="text/csv")


# ======================================================================
# TAB 5: SIMULATION — one journey on the roadmap
# ======================================================================
with tab_sim:
    if dp is None:
        st.info("👈  Set parameters and press **Solve DP** first.")
    else:
        p = dp.p
        IS_CF0 = (float(Cf) == 0.0)
        st.subheader("Sample paths under the optimal policy")
        if IS_CF0:
            st.caption(
                "**Cf = 0: the 2-D switching model of the note is simulated "
                "here, solved by solver_cf0_2d.py.** That model has no b\u2081 "
                "state: a Retailer-1 demand is either served on arrival at "
                "cost c\u1d64, or rejected and charged \u03c0\u2081\u03c4 once, "
                "never to be served. Dispatch is therefore always one unit at "
                "a time and the flow cost carries no \u03c0\u2081b\u2081 term. "
                "The overlay is the analytic staircase of Eq. (20)-(22)."
            )
        else:
            st.caption(
                "Exogenous Poisson arrivals push the system along; at every "
                "event the DP looks up the Policy Table and acts. Costs are "
                "integrated exactly in continuous time. Tick the overlay to "
                "run the analytic policy of Eq. (34)+(36) on the SAME "
                "arrivals."
            )
        cs1, cs2, cs3, cs4 = st.columns(4)
        with cs1:
            seed0 = st.number_input("seed", min_value=0, value=42,
                                    key="sim_seed")
        with cs2:
            npaths = st.number_input("paths", min_value=1, max_value=20,
                                     value=5, key="sim_np")
        with cs3:
            I0 = st.number_input("start I₂", min_value=1,
                                 max_value=int(p.I2_max),
                                 value=min(30, p.I2_max), key="sim_i0")
        with cs4:
            if IS_CF0:
                b0 = 0
                st.caption("start b₁ — not a state when Cf = 0")
            else:
                b0 = st.number_input("start b₁", min_value=0,
                                     max_value=int(p.b1_max),
                                     value=min(2, p.b1_max), key="sim_b0")
        show_an = st.checkbox("Overlay analytic policy on the same arrivals",
                              value=False, key="sim_an")

        dtp = p.T / p.N

        def _dp_q(I2, b1, tau):
            if I2 < 1 or b1 < 1:
                return 0
            n = int(np.clip(round(tau / dtp), 1, p.N))
            q = dp.get_policy(n, I2, b1)
            return min(q, min(I2, b1)) if q > 0 else 0

        if IS_CF0:
            dp2d = solve_cf0_2d_policy(p.T, max(int(p.N), 2000), lam1, lam2,
                                       h, cu, pi1, pi2, 0.0, 0.0)

            def _stair_thr(tau):
                """Analytic threshold of Eq. (20)-(22) at exact tau."""
                if tau <= 0:
                    return np.inf
                lvl = lam2 * (cu + (pi2 - pi1) * tau) / (h + pi2)
                if lvl <= 0:
                    return 1.0
                mu = lam2 * tau
                if lvl > mu:
                    return np.inf
                tot, pmf, cdf = 0.0, math.exp(-mu), math.exp(-mu)
                for m in range(1, 400):
                    tot += 1.0 - cdf
                    if tot >= lvl:
                        return float(m)
                    pmf *= mu / m
                    cdf += pmf
                return np.inf

        def run_path_cf0(seed):
            """
            Simulate the 2-D switching model. State is I2 alone. A Retailer-1
            arrival is served (cost cu, I2 falls by one) or rejected (cost
            pi1*tau once, I2 unchanged, never served later). Flow cost is
            h*I2+ + pi2*I2- only. The b1 line shown is the cumulative count of
            rejected Retailer-1 demands: a display quantity, not a state.
            """
            trng = np.random.default_rng(int(seed))
            n1 = trng.poisson(lam1 * p.T)
            n2 = trng.poisson(lam2 * p.T)
            ev = sorted([(t, "R1") for t in np.sort(trng.random(n1)) * p.T]
                        + [(t, "R2") for t in np.sort(trng.random(n2)) * p.T])
            pols = ["DP"] + (["AN"] if show_an else [])
            S = {k: dict(I2=int(I0), b1=0, flow=0.0, disp=0.0, nCf=0)
                 for k in pols}
            traj = {k: [(0.0, S[k]["I2"], 0)] for k in pols}
            ships = {k: [] for k in pols}
            rows = []

            def serve(k, tau):
                """Return 1 if this policy serves the arriving R1 demand."""
                s = S[k]
                if s["I2"] < 1:
                    return 0
                if k == "DP":
                    n = int(np.clip(round(tau / (p.T / dp2d.p.N)),
                                    1, dp2d.p.N))
                    return int(dp2d.get_policy(n, int(s["I2"])))
                return int(s["I2"] >= _stair_thr(tau))

            t = 0.0
            for te_, kind in ev:
                for k in pols:
                    s = S[k]
                    s["flow"] += (te_ - t) * (h * max(s["I2"], 0)
                                              + pi2 * max(-s["I2"], 0))
                t = te_
                tau = p.T - t
                pre = {k: (S[k]["I2"], S[k]["b1"]) for k in pols}
                act = {}
                for k in pols:
                    s = S[k]
                    if kind == "R2":
                        s["I2"] -= 1
                        act[k] = "R2 demand"
                    else:
                        q = serve(k, tau)
                        if q:
                            s["disp"] += cu
                            s["nCf"] += 1
                            s["I2"] -= 1
                            ships[k].append((t, 1))
                            act[k] = "serve q=1"
                        else:
                            s["flow"] += pi1 * tau      # one-off rejection
                            s["b1"] += 1                # display only
                            act[k] = f"reject (+{pi1 * tau:.2f})"
                    traj[k].append((t, s["I2"], s["b1"]))
                rows.append(dict(
                    t=round(t, 4), tau=round(tau, 4), ev=kind,
                    **{f"{k}_state": f"(I₂={pre[k][0]})" for k in pols},
                    **{f"{k}_act": act[k] for k in pols}))
            for k in pols:
                s = S[k]
                s["flow"] += (p.T - t) * (h * max(s["I2"], 0)
                                          + pi2 * max(-s["I2"], 0))
            return traj, ships, rows, S

        def run_path(seed):
            trng = np.random.default_rng(int(seed))
            n1 = trng.poisson(lam1 * p.T)
            n2 = trng.poisson(lam2 * p.T)
            ev = sorted([(t, "R1") for t in np.sort(trng.random(n1)) * p.T]
                        + [(t, "R2") for t in np.sort(trng.random(n2)) * p.T])
            pols = {"DP": _dp_q}
            if show_an:
                pols["AN"] = _an_q_exact
            S = {k: dict(I2=int(I0), b1=int(b0), flow=0.0, disp=0.0, nCf=0)
                 for k in pols}
            traj = {k: [(0.0, S[k]["I2"], S[k]["b1"])] for k in pols}
            ships = {k: [] for k in pols}
            rows = []

            def act(k, tau, trigger_ok):
                s = S[k]
                q = pols[k](s["I2"], s["b1"], tau) if trigger_ok else 0
                if q > 0:
                    s["disp"] += Cf + cu * q
                    s["nCf"] += 1
                    s["I2"] -= q
                    s["b1"] -= q
                return q

            q0 = {k: act(k, p.T, True) for k in pols}
            for k in pols:
                traj[k].append((0.0, S[k]["I2"], S[k]["b1"]))
                if q0[k]:
                    ships[k].append((0.0, q0[k]))
            rows.append(dict(t=0.0, tau=p.T, ev="--",
                             **{f"{k}_state": f"({S[k]['I2']+q0[k]},"
                                              f"{S[k]['b1']+q0[k]})"
                                for k in pols},
                             **{f"{k}_act": (f"q={q0[k]}" if q0[k] else "wait")
                                for k in pols}))
            t = 0.0
            for te_, kind in ev:
                for k in pols:
                    s = S[k]
                    s["flow"] += (te_ - t) * (h * max(s["I2"], 0)
                                              + pi1 * s["b1"]
                                              + pi2 * max(-s["I2"], 0))
                t = te_
                for k in pols:
                    s = S[k]
                    if kind == "R1":
                        s["b1"] += 1
                    else:
                        s["I2"] -= 1
                    traj[k].append((t, s["I2"], s["b1"]))
                tau = p.T - t
                pre = {k: (S[k]["I2"], S[k]["b1"]) for k in pols}
                qk = {}
                for k in pols:
                    # AN: only R1 arrivals can trigger (Thms 3-4, exact);
                    # DP: any arrival can trigger (threshold not monotone)
                    ok = (kind == "R1") if k == "AN" else True
                    qk[k] = act(k, tau, ok)
                    traj[k].append((t, S[k]["I2"], S[k]["b1"]))
                    if qk[k]:
                        ships[k].append((t, qk[k]))
                rows.append(dict(
                    t=round(t, 4), tau=round(tau, 4), ev=kind,
                    **{f"{k}_state": f"({pre[k][0]},{pre[k][1]})"
                       for k in pols},
                    **{f"{k}_act": (f"q={qk[k]}" if qk[k] else "wait")
                       for k in pols}))
            for k in pols:
                s = S[k]
                s["flow"] += (p.T - t) * (h * max(s["I2"], 0)
                                          + pi1 * s["b1"]
                                          + pi2 * max(-s["I2"], 0))
            return traj, ships, rows, S

        _runner = run_path_cf0 if IS_CF0 else run_path
        results = [_runner(int(seed0) + i) for i in range(int(npaths))]

        # ── summary table ─────────────────────────────────────────
        summ = []
        for i, (_, ships, _, S) in enumerate(results):
            row = dict(path=i + 1, seed=int(seed0) + i,
                       DP_cost=round(S["DP"]["flow"] + S["DP"]["disp"], 2),
                       DP_dispatches=S["DP"]["nCf"])
            if not IS_CF0:
                row["DP_fixed"] = round(S["DP"]["nCf"] * Cf, 1)
            if show_an:
                row.update(AN_cost=round(S["AN"]["flow"] + S["AN"]["disp"], 2),
                           AN_dispatches=S["AN"]["nCf"],
                           gap=round(S["AN"]["flow"] + S["AN"]["disp"]
                                     - S["DP"]["flow"] - S["DP"]["disp"], 2))
            summ.append(row)
        st.dataframe(pd.DataFrame(summ), hide_index=True)

        pick = st.selectbox(
            "show path", list(range(1, int(npaths) + 1)),
            format_func=lambda i: f"path {i} (seed {int(seed0) + i - 1})",
            key="sim_pick")
        traj, ships, rows, S = results[pick - 1]

        # ── main view: state trajectories ─────────────────────────
        figt, (axs, axc) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True,
            gridspec_kw=dict(height_ratios=[3, 2]))
        styles = {"DP": dict(lw=2.0, alpha=1.0),
                  "AN": dict(lw=1.4, alpha=0.65, ls="--")}
        colI, colB = "#1F618D", "#B03A2E"
        for k in traj:
            ts = [x[0] for x in traj[k]]
            axs.plot(ts, [x[1] for x in traj[k]], color=colI,
                     label=f"I₂ ({k})", **styles[k])
            axs.plot(ts, [x[2] for x in traj[k]], color=colB,
                     label=f"b₁ ({k})", **styles[k])
        for (ts_, q) in ships["DP"]:
            axs.axvline(ts_, color="0.55", lw=0.8)
            axs.annotate(f"q={q}", (ts_, axs.get_ylim()[1] * 0.97),
                         fontsize=7, ha="center", va="top", color="0.3")
        if show_an:
            for (ts_, q) in ships["AN"]:
                axs.plot([ts_], [0], marker="v", color="#B03A2E",
                         ms=5, alpha=0.6, clip_on=False)
            # divergence bands: events where exactly one of the two shipped
            dpT = {round(ts_, 6) for ts_, _ in ships["DP"]}
            anT = {round(ts_, 6) for ts_, _ in ships["AN"]}
            for ts_ in dpT ^ anT:
                axs.axvspan(ts_ - 0.012, ts_ + 0.012, color="#F5B041",
                            alpha=0.25, lw=0)
        axs.axhline(0, color="0.7", lw=0.6)
        axs.set_ylabel("units")
        axs.set_title(
            f"path {pick}: vertical lines = DP dispatches"
            + ("; ▾ = analytic dispatches; orange bands = divergence"
               if show_an else ""), fontsize=10)
        axs.legend(fontsize=7, ncol=2 if show_an else 1)
        axs.grid(True, alpha=0.25)

        # ── cost accumulation (exact piecewise integration) ───────
        for k in traj:
            pts_t, pts_c = [0.0], [0.0]
            flow_acc, disp_acc, tprev = 0.0, 0.0, 0.0
            idx = 0
            shp = ships[k]
            seq = traj[k]
            for j in range(1, len(seq)):
                tt, i2, bb = seq[j]
                i2p, bbp = seq[j - 1][1], seq[j - 1][2]
                flow_acc += (tt - tprev) * (h * max(i2p, 0) + pi1 * bbp
                                            + pi2 * max(-i2p, 0))
                # dispatch jump: same t, I2 decreased with b1 decreased
                if tt == seq[j - 1][0] and i2 < i2p and bb < bbp:
                    disp_acc += Cf + cu * (i2p - i2)
                tprev = tt
                pts_t.append(tt); pts_c.append(flow_acc + disp_acc)
            tail = (p.T - tprev) * (h * max(seq[-1][1], 0) + pi1 * seq[-1][2]
                                    + pi2 * max(-seq[-1][1], 0))
            pts_t.append(p.T); pts_c.append(flow_acc + disp_acc + tail)
            axc.plot(pts_t, pts_c,
                     color="#1F618D" if k == "DP" else "#B03A2E",
                     label=f"{k}: total "
                           f"{S[k]['flow'] + S[k]['disp']:.1f} "
                           f"(dispatch {S[k]['disp']:.0f} in "
                           f"{S[k]['nCf']}×)", **{
                               kk: vv for kk, vv in styles[k].items()
                               if kk != "alpha"})
        axc.set_xlabel("t"); axc.set_ylabel("cumulative cost")
        axc.legend(fontsize=8); axc.grid(True, alpha=0.25)
        figt.tight_layout()
        st.pyplot(figt); plt.close(figt)

        # ── full event table ──────────────────────────────────────
        with st.expander("Event table — every decision, including waits"):
            st.caption(
                "One row per event. state = (I₂, b₁) after the arrival, "
                "before the decision; the action column is the Policy-Table "
                "lookup at that state and τ. The analytic policy is checked "
                "only after R1 arrivals (exact by Theorems 3-4); the DP is "
                "checked after every arrival."
            )
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=420)