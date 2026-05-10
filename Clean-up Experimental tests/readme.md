# Transshipment Policy Explorer

An interactive web app for exploring the optimal dispatch policy in a two-location finite-horizon transshipment model. The app solves a backward-induction dynamic programme and lets you adjust model parameters in real time to see how the optimal policy changes.

**Access the app directly in your browser:**
> https://consolidationtransshipment-bceq7yuvreyj6jhnsckcvb.streamlit.app/

---

## What this app does

The system consists of two retailers over a planning horizon $[0, T]$. At $t = 0$, Retailer 1 stocks out. From this point, all demand at Retailer 1 is backlogged and sent to Retailer 2 as transshipment requests. Retailer 2 must decide **when to dispatch** and **how many units to send**, trading off:
- the fixed and unit transshipment cost ($C_f$, $c_u$),
- the backlog penalty at Retailer 1 ($\pi_1$),
- the holding cost and shortage penalty at Retailer 2 ($h$, $\pi_2$),
- and the terminal replenishment cost at the end of the horizon.

The app solves this problem exactly via backward induction and displays the resulting optimal policy and value function. Analytical threshold formulas from the accompanying manuscript are overlaid for comparison. [?]

---

## Understanding $\tau$ (the x-axis)

$\tau = T - t$ is the **remaining time until the end of the planning horizon**. The x-axis is reversed so that time flows left to right:

| Position | $\tau$ | Meaning |
|----------|-----|---------|
| Far left | $T$ | Just started — full horizon remaining |
| Middle | $T/2$ | Halfway through |
| Far right | $0$ | End of horizon — terminal costs apply |

Reading the plots from left to right shows how the optimal policy evolves as the deadline approaches.

---

## How to use the app

### Step 1 — Set parameters

Use the sliders in the **left panel**:

| Parameter | Description |
|-----------|-------------|
| $T$ | Length of planning horizon |
| $N$ | Number of discrete time periods (higher = more accurate, slower) |
| $\lambda_1$, $\lambda_2$ | Demand arrival rates at Retailer 1 and Retailer 2 |
| $h$ | Holding cost rate at Retailer 2 |
| $C_f$ | Fixed cost per dispatch |
| $c_u$ | Unit transshipment cost |
| $\pi_1$ | Backlog penalty rate at Retailer 1 |
| $\pi_2$ | Shortage penalty rate at Retailer 2 |
| $c_1$, $c_2$ | Terminal purchasing costs |
| $v_2$ | Salvage value of remaining inventory at Retailer 2 |
| $I_{2,\max}$, $I_{2,\min}$, $b_{1,\max}$ | State space bounds |

### Step 2 — Solve the DP

Press **▶ Solve DP**. The solver runs backward induction over all states and time periods. This may take a few seconds depending on $N$ and the state space size.

### Step 3 — Choose what to plot

**X axis:**
- $\tau$ (remaining time)
- $I_2$ (Retailer 2 inventory)
- $b_1$ (Retailer 1 backlog)

**Y axis:**
- $q^*$ — optimal dispatch quantity
- $b_1^*$ threshold — minimum backlog to trigger dispatch (when $b_1 \leq I_2$)
- $\bar{I}_2$ threshold — minimum inventory to trigger dispatch (when $b_1 \geq I_2 + 1$)
- $V^n$ — expected total cost from the current state (value function)

**Overlay analytical formula** — tick to show the manuscript's closed-form threshold as a dashed line.

### Step 4 — Fixed dimensions and curves

When the x-axis uses one dimension, the other two must be fixed using the sliders below the dropdowns. **Number of curves** plots multiple lines by varying the fixed dimension；for example, with $X = \tau$ and 3 curves, the plot shows three lines for three different $I_2$ values.

> For the I2bar threshold plot, only one curve is shown regardless of this setting, since $\bar{I}_2$ is already found by scanning over $I_2$.

---

## Interpreting each plot

### $\bar{I}_2$ threshold vs $\tau$ (most informative)

**Setup:** X = $\tau$, Y = $\bar{I}_2$ threshold, Fixed $b_1$ = 45.

**What to look for:**

- **Flat line ($\pi_1 = \pi_2$):** threshold is tau-independent. Confirmed by the formula $I_2^* = \lambda_2 c_u / (h + \pi)$.

- **Decreasing left to right ($\pi_1 > \pi_2$):** with less time remaining, the threshold rises — the system becomes more conservative near the end of the horizon. With more time remaining (far left), Retailer 2 is willing to dispatch at a lower inventory level because there is more time to benefit from clearing the backlog.

- **Upturn near the far right (small tau):** this is the finite-horizon transient effect. Very close to the end of the horizon, the remaining time is too short to recover the transshipment cost, so the DP raises the threshold and eventually stops dispatching. This is economically rational, not a bug.

- **Dashed line (analytical formula):** matches the DP well at large tau; deviates near $\tau = 0$ due to the finite-horizon transient. In the $\pi_1 > \pi_2$ case, the manuscript's original formula (Error 1) diverges at moderate tau because the coefficient changes sign.

**Parameter effects:**

| Change | Effect on threshold | Reason |
|--------|-------------------|--------|
| Larger $c_u$ | Higher threshold | More inventory needed to justify unit cost |
| Larger $C_f$ | Higher threshold | More inventory needed to amortise fixed cost |
| Larger $h$ | Lower threshold | Holding inventory is expensive; dispatch more readily |
| Larger $\pi_1$ (when $\pi_1 > \pi_2$) | Steeper decline left to right | Heavier penalty at R1 makes earlier dispatch more attractive |
| Larger $\pi_2$ | Lower threshold | $\alpha_2 = \lambda_2 c_u/(h+\pi_2)$ decreases |

---

### $q^*$ vs $\tau$

**Setup:** X = $\tau$, Y = $q^*$, Fixed $I_2$ and Fixed $b_1$.

**What to look for:**

- **Flat horizontal line:** in most regions q* does not depend on tau. For A1/B1a: $q^* = \min(I_2, b_1)$. For A2: $q^* = I_2 - \beta$.

- **Step down to 0 near the right:** as tau approaches 0, the DP may stop dispatching (the I2bar threshold rises above the current I2). The jump from $\min(I_2, b_1)$ to $0$ at a particular tau is economically rational — so close to the end of the horizon, the time remaining is too short to recover the transshipment cost through reduced penalties. This is **not** a bug.

**Parameter effects:**
- Larger $c_u$ or $C_f$: the system stops dispatching earlier (step to 0 occurs further from the end).
- Larger $h$: the system dispatches later into the horizon (step to 0 closer to the end).

---

### $q^*$ vs $b_1$

**Setup:** X = $b_1$, Y = $q^*$, Fixed $I_2$ and Fixed $\tau$.

**What to look for:**

- **Step function:** $q^*$ increases linearly with $b_1$ (slope 1) up to $\min(I_2, b_1) = I_2$, then flattens at $q^* = I_2$. The kink is at $b_1 = I_2$ for A1/B1a, or at $b_1 = I_2 - \beta$ for A2.

- **Multiple curves (varying I2):** the kink shifts right and the plateau rises as I2 increases, forming a staircase pattern across the curves.

---

### $q^*$ vs $I_2$

**Setup:** X = $I_2$, Y = $q^*$, Fixed $b_1$ and Fixed $\tau$.

**What to look for:**

- **Step function (mirror of above):** $q^*$ increases linearly with $I_2$ up to $b_1$, then flattens at $q^* = b_1$.

---

### $b_1^*$ threshold vs $\tau$

**Setup:** X = $\tau$, Y = $b_1^*$ threshold, Fixed $I_2$.

**What to look for:**

- **Persistent gap between DP and analytical formula:** the DP threshold is consistently 2–3 units above the analytical prediction, regardless of $\tau$, $I_2$, or parameter values. This is a structural limitation of the particular solution Vw used in the manuscript: Vw satisfies the ODE and the spatial boundary at $I_2 = 0$, but does not satisfy the terminal boundary condition at $\tau = 0$. The discrepancy is $O(I_2^2)$ and does not vanish within a finite horizon. The formula is asymptotically exact only as $T \to \infty$.

- **Do not expect the two lines to converge** for any parameter choice. The gap is not caused by any transcription error.

**Parameter effects:**
- Larger $C_f$: both DP and analytical threshold rise, gap remains similar.
- Larger $I_2$ (Number of curves): analytical threshold approaches 0 while DP stays around 3 — the gap widens at large $I_2$, illustrating the structural limitation most clearly.

---

### Value function $V^n$ vs $\tau$

**Setup:** X = $\tau$, Y = $V^n$, Fixed $I_2$ and Fixed $b_1$.

**What to look for:**

- **Monotone increasing from right to left:** more remaining time means higher expected total cost (more opportunity for demand to arrive and costs to accumulate).
- **Slope** is approximately the instantaneous cost rate $h I_2^+ + \pi_1 b_1 + \pi_2 I_2^-$ at the current state.
- **Multiple curves (varying b1):** lines are approximately parallel; spacing reflects the cost of one additional unit of backlog (approximately $\pi_1 \tau + c_1$).

---

### Value function $V^n$ vs $I_2$

**Setup:** X = $I_2$, Y = $V^n$, Fixed $b_1$ and Fixed $\tau$.

**What to look for:**

- **Monotone decreasing:** more inventory means lower expected cost.
- **Kink at $I_2 = 0$:** for $I_2 > 0$, slope is approximately $-v_2$ (salvage value); for $I_2 < 0$, slope steepens (shortage penalty $\pi_2$ and terminal cost $c_2$ apply).
- **Multiple curves (varying b1):** curves are approximately parallel, shifted upward by the backlog penalty. Spacing reflects the marginal cost of backlog.

---

## Dimensionless groups (top of page)

These update automatically without solving:

| Display | Formula | Meaning |
|---------|---------|---------|
| $\alpha_2$ | $\lambda_2 c_u / (h + \pi_2)$ | Determines dispatch threshold scale |
| $\gamma$ | $\lambda_2 (\pi_1 - \pi_2) / (h + \pi_2)$ | Penalty asymmetry between retailers |
| $\Phi_2$ | $2 \lambda_2 C_f / (h + \pi_2)$ | Scaled fixed cost |
| Region | — | Which of the 8 policy regions the parameters fall into |

- $\gamma = 0$ ($\pi_1 = \pi_2$): regions A1/A2 — all thresholds are $\tau$-independent.
- $\gamma > 0$ ($\pi_1 > \pi_2$): regions B1a–B1c — thresholds decrease as $\tau$ increases.
- $\gamma < 0$ ($\pi_1 < \pi_2$): regions B2a–B2c — thresholds increase as $\tau$ increases.

---

## Troubleshooting

**Solver is slow:** reduce $N$ (try 100) or reduce $I_{2,\max}$ and $b_{1,\max}$.

**Plot does not update after changing a slider:** press ▶ Solve DP again. The solver must be re-run whenever parameters change.

**No dispatch observed at some tau values:** the optimal threshold exceeds I2_max at that tau — the system finds it suboptimal to dispatch at any inventory level in the state space. Most common near the end of the horizon (small $\tau$) and for large $\alpha_2$.