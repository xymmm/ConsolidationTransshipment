# Transshipment Policy Explorer — Setup Guide

## What this app does

This is an interactive dashboard for exploring the optimal dispatch policy 
in a two-location transshipment model. You can adjust model parameters using 
sliders and instantly see how the optimal policy changes. The app runs 
entirely on your own computer — no internet connection required after setup.

---

## Files you need

Make sure you have both files in the **same folder**:

```
your_folder/
├── solver.py
└── app.py
```

---

## Step 1: Install Python

You need Python 3.8 or later.  
Check if you already have it by opening a terminal and typing:

```bash
python --version
```

If not installed, download from: https://www.python.org/downloads/

---

## Step 2: Install required packages

Open a terminal, navigate to the folder containing the files, and run:

```bash
pip install streamlit matplotlib numpy
```

This only needs to be done once.

---

## Step 3: Run the app

In the terminal, navigate to the folder containing `app.py` and `solver.py`:

```bash
cd /path/to/your/folder
```

Then start the app:

```bash
streamlit run app.py
```

A browser window will open automatically at `http://localhost:8501`.  
If it does not open automatically, copy that address into your browser.

---

## Step 4: Using the app

### Solving the DP

1. Set the model parameters using the sliders in the **left panel**
2. Press the **▶ Solve DP** button (red button at the bottom of the panel)
3. Wait a few seconds for the solver to finish — a "DP solved!" message will appear
4. The plot will update automatically

> **Note:** You must press Solve DP every time you change a parameter.  
> The solver takes longer with larger state spaces (bigger I2_max, b1_max).

### Parameter panel

| Parameter | Description |
|-----------|-------------|
| T | Length of planning horizon |
| N | Number of discrete time periods (higher = more accurate, slower) |
| λ₁ | Demand arrival rate at Retailer 1 |
| λ₂ | Demand arrival rate at Retailer 2 |
| h | Holding cost rate |
| Cf | Fixed transshipment cost |
| cu | Unit transshipment cost |
| π₁ | Backlog penalty rate at Retailer 1 |
| π₂ | Backlog penalty rate at Retailer 2 |
| c₁, c₂ | Terminal purchasing costs |
| v₂ | Salvage value of remaining inventory at Retailer 2 |
| I2_max, I2_min | State space bounds for Retailer 2 inventory |
| b1_max | State space bound for Retailer 1 backlog |

### Dimensionless groups

The four boxes at the top of the page update automatically (no need to solve):

| Display | Formula | Meaning |
|---------|---------|---------|
| α₂ | λ₂cu/(h+π₂) | Key threshold parameter |
| γ | λ₂(π₁−π₂)/(h+π₂) | Asymmetry between retailers |
| Φ₂ | 2λ₂Cf/(h+π₂) | Fixed cost parameter |
| Region | — | Which of the 8 policy regions the current parameters fall into (A1, A2, B1a, etc.) |

### Choosing what to plot

Use the three dropdown menus to choose:

**X axis** — what to plot along the horizontal axis:
- `τ (remaining time)` — how the policy changes over the planning horizon
- `I₂ (inventory)` — how the policy changes with Retailer 2's inventory level
- `b₁ (backlog)` — how the policy changes with Retailer 1's backlog

**Y axis** — what to measure:
- `q* (optimal dispatch quantity)` — how many units to send
- `b₁* threshold (Case 1)` — minimum backlog required to trigger dispatch (when b₁ ≤ I₂)
- `Ī₂ threshold (Case 2)` — minimum inventory required to trigger dispatch (when b₁ ≥ I₂+1)
- `V^n (value function)` — expected total cost from current state

**Overlay analytical formula** — checkbox:
- Ticked: shows the closed-form analytical formula as a dashed line alongside the DP result
- Unticked: shows DP result only

### Fixed dimension sliders

When the X axis uses one dimension (e.g. τ), the other dimensions must 
be fixed to a specific value. Use these sliders to choose those values:

- **Fixed I₂** — the inventory level to use when I₂ is not on the X axis
- **Fixed b₁** — the backlog level to use when b₁ is not on the X axis
- **Fixed τ** — the remaining time to use when τ is not on the X axis

**Number of curves** — draw multiple lines by varying the fixed dimension.  
For example, if X = τ and you set curves = 3, the plot will show three lines 
for three different I₂ values.

> For the Ī₂ threshold (Case 2) plot, only one curve is shown regardless 
> of this setting, because Ī₂ is computed by scanning over I₂ and does 
> not depend on a fixed I₂ value.

---

## Typical use cases

### 1. Reproduce the Cf=0 threshold plot (I₂* vs τ)

- Set Cf = 0, π₁ = π₂
- Press Solve DP
- X axis: `τ (remaining time)`
- Y axis: `Ī₂ threshold (Case 2)`
- Fixed b₁: set to a large value (e.g. 45) to ensure the Case-2 regime
- Tick "Overlay analytical formula" to compare with I₂* = λ₂cu/(h+π)

### 2. See how q* changes with backlog

- X axis: `b₁ (backlog)`
- Y axis: `q* (optimal dispatch quantity)`
- Fixed I₂: choose a specific inventory level
- Fixed τ: choose a specific remaining time
- Number of curves: increase to compare across different I₂ values

### 3. Compare DP threshold with analytical formula (Theorem 3/4/5)

- X axis: `τ (remaining time)`
- Y axis: `b₁* threshold (Case 1)` or `Ī₂ threshold (Case 2)`
- Tick "Overlay analytical formula"
- The gap between the solid line (DP) and dashed line (analytical) 
  shows the discrepancy discussed in the validation report

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'streamlit'"**  
Run `pip install streamlit` in the terminal.

**"ModuleNotFoundError: No module named 'solver'"**  
Make sure `solver.py` and `app.py` are in the same folder, and that you 
ran `streamlit run app.py` from that folder (not a different directory).

**The browser does not open automatically**  
Manually go to `http://localhost:8501` in your browser.

**The plot does not update after changing a slider**  
Press the **▶ Solve DP** button. The solver must be re-run whenever 
parameters change.

**The solver takes too long**  
Reduce N (try 100 instead of 200), or reduce I2_max and b1_max.  
For quick exploration, N=100 is usually sufficient.

---

## Stopping the app

Press `Ctrl + C` in the terminal window where you ran `streamlit run app.py`.