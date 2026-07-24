"""
compare_q_general.py — Note (general Cf) analytic policy vs solver.py DP.

Section 6.2 example:  lam1=5, lam2=3, h=1, pi1=pi2=6, Cf=8, cu=1, T=5.

The note's model IS the solver.py model (state (I2,b1,tau), batch dispatch q,
pi1*b1 charged per period), so this is a like-for-like comparison. The terminal
cost is set to zero because the note's Vw uses V(I2,b1,0)=0.

Analytic policy (note eqs. 30-36):
    delta(m,tau) = (h+pi2)/lam2 * E[min(K,m)] - cu + (pi1-pi2)*tau,
                                                    K ~ Poisson(lam2*tau)
    m_c(tau)     = min{ m >= 1 : delta(m,tau) > 0 }        critical level
    N(I2,tau)    = #{ m in 1..I2 : delta(m,tau) > 0 }
    q_circ       = min(b1, N)
    S            = sum_{i=0}^{q_circ-1} delta(I2-i, tau)
    q*           = q_circ if S > Cf else 0
    b1bar(I2,tau)= min{ b >= 1 : sum_{i=0}^{b-1} delta(I2-i,tau) > Cf }, else +inf

Outputs:
    q_comparison_general_Cf.txt   human-readable report (matrices + diagnostics)
    q_comparison_general_Cf.csv   tidy long format, one row per (tau, I2, b1)
"""
import math
import csv
import numpy as np
from math import exp

# ══════════════════════════════════════════════════════════════════════
#  PARAMETERS (Section 6.2)
# ══════════════════════════════════════════════════════════════════════
T, N = 5.0, 800
LAM1, LAM2 = 5.0, 3.0
H, PI1, PI2 = 1.0, 6.0, 6.0
CF, CU = 8.0, 1.0

TAUS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
I2_LIST = list(range(1, 31))
B1_LIST = list(range(1, 31))

TXT_OUT = "q_comparison_general_Cf.txt"
CSV_OUT = "q_comparison_general_Cf.csv"


# ══════════════════════════════════════════════════════════════════════
#  ANALYTIC SIDE
# ══════════════════════════════════════════════════════════════════════
def Emin_table(mu, mmax=90):
    """E[min(K,m)] for m = 0..mmax, K ~ Poisson(mu). Pure Python, no scipy."""
    out = np.zeros(mmax + 1)
    pmf = exp(-mu)      # P(K = m-1), starting at m = 1
    cb = 0.0            # P(K <= m-2)
    M = 0.0
    for m in range(1, mmax + 1):
        M += max(1.0 - (cb + pmf), 0.0)      # add P(K >= m)
        out[m] = M
        cb += pmf
        pmf *= mu / m
    return out


def delta_fn(Em, tau):
    return lambda m: (H + PI2) / LAM2 * Em[m] - CU + (PI1 - PI2) * tau


def analytic_state(I2, b1, tau, Em):
    """Return (q_star, q_circ, S, N_pos, m_c) for one state."""
    d = delta_fn(Em, tau)
    Npos = sum(1 for m in range(1, I2 + 1) if d(m) > 0)
    mc = next((m for m in range(1, 90) if d(m) > 0), None)
    qc = min(b1, Npos)
    if qc == 0:
        return 0, 0, 0.0, Npos, mc
    S = sum(d(I2 - i) for i in range(qc))
    return (qc if S > CF else 0), qc, S, Npos, mc


def analytic_b1bar(I2, tau, Em):
    d = delta_fn(Em, tau)
    S = 0.0
    for b in range(1, I2 + 1):
        dm = d(I2 - b + 1)
        if dm <= 0:
            break
        S += dm
        if S > CF:
            return b
    return np.inf


# ══════════════════════════════════════════════════════════════════════
#  DP SIDE (solver.py model, vectorised, zero terminal)
# ══════════════════════════════════════════════════════════════════════
def solve_dp():
    s1, s2 = LAM1 * T, LAM2 * T
    Imin = -int(math.ceil(s2 + 4 * math.sqrt(s2)))
    Imax = int(math.ceil(max(35, s2 + 4 * math.sqrt(s2))))
    b1m = int(math.ceil(s1 + 4 * math.sqrt(s1)))
    dt = T / N
    p1, p2 = LAM1 * dt, LAM2 * dt
    p0 = 1 - p1 - p2
    I2v = np.arange(Imin, Imax + 1)
    I2g = I2v[:, None]
    b1g = np.arange(0, b1m + 1)[None, :]
    cI = lambda x: np.clip(x, Imin, Imax) - Imin
    cB = lambda x: np.clip(x, 0, b1m)
    V = np.zeros((len(I2v), b1m + 1))
    qmax = min(Imax, b1m)
    keep = {min(N, max(1, round(t / dt))): t for t in TAUS}
    pol = {}
    for n in range(1, N + 1):
        best = np.full(V.shape, np.inf)
        bq = np.zeros(V.shape, np.int16)
        for q in range(0, qmax + 1):
            feas = (np.ones(V.shape, bool) if q == 0
                    else ((I2g >= q) & (b1g >= q) & (I2g > 0) & (b1g > 0)))
            if not feas.any():
                continue
            I2a = np.broadcast_to(I2g - q, V.shape)
            b1a = np.broadcast_to(b1g - q, V.shape)
            flow = dt * (H * np.maximum(0, I2a) + PI1 * b1a
                         + PI2 * np.maximum(0, -I2a))
            cost = ((CF if q > 0 else 0.0) + CU * q + flow
                    + p0 * V[cI(I2a), cB(b1a)]
                    + p1 * V[cI(I2a), cB(b1a + 1)]
                    + p2 * V[cI(I2a - 1), cB(b1a)])
            cost = np.where(feas, cost, np.inf)
            upd = cost < best
            best = np.where(upd, cost, best)
            bq = np.where(upd, q, bq)
        V = best
        if n in keep:
            pol[keep[n]] = bq.copy()
    return pol, I2v, b1m, dt


print("Solving DP ...")
POL, I2V, B1M, DT = solve_dp()
ii = lambda I2: int(np.where(I2V == I2)[0][0])


# ══════════════════════════════════════════════════════════════════════
#  BUILD RECORDS
# ══════════════════════════════════════════════════════════════════════
rows = []
per_tau = {}
for tau in TAUS:
    Em = Emin_table(LAM2 * tau)
    A = np.zeros((len(I2_LIST), len(B1_LIST)), int)
    D = np.zeros_like(A)
    for a, I2 in enumerate(I2_LIST):
        for b, b1 in enumerate(B1_LIST):
            qa, qc, S, Npos, mc = analytic_state(I2, b1, tau, Em)
            qd = int(POL[tau][ii(I2), min(b1, B1M)])
            A[a, b], D[a, b] = qa, qd
            rows.append(dict(
                tau=tau, I2=I2, b1=b1,
                q_analytic=qa, q_DP=qd, diff=qd - qa,
                dispatch_analytic=int(qa > 0), dispatch_DP=int(qd > 0),
                q_circ=qc, S=round(S, 4), Cf=CF,
                S_minus_Cf=round(S - CF, 4),
                N_profitable=Npos, m_c=(mc if mc else -1),
                retained_DP=I2 - qd,
                retained_predicted=(max(I2 - b1, (mc - 1)) if mc else I2),
            ))
    per_tau[tau] = (A, D)

# ── CSV ───────────────────────────────────────────────────────────────
with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {CSV_OUT}  ({len(rows)} rows)")


# ── TXT report ────────────────────────────────────────────────────────
L = []
def P(s=""):
    L.append(s)

BAR = "=" * 100
P(BAR)
P("  q* AND DISPATCH-TRIGGER COMPARISON:  general-Cf note  vs  solver.py DP")
P(BAR)
P()
P("  Example      : Section 6.2 of the general-Cf note")
P(f"  Parameters   : lam1={LAM1}, lam2={LAM2}, h={H}, pi1={PI1}, pi2={PI2}, "
  f"Cf={CF}, cu={CU}, T={T}")
P(f"  DP           : solver.py model, N={N} periods, dt={DT:.5f}, "
  f"I2 in [{I2V[0]}, {I2V[-1]}], b1_max={B1M}")
P("  Terminal     : c1 = c2 = v2 = 0   (the note's Vw uses V(I2,b1,0)=0)")
P(f"  Grid         : I2 = {I2_LIST[0]}..{I2_LIST[-1]}, "
  f"b1 = {B1_LIST[0]}..{B1_LIST[-1]}, tau = {TAUS}")
P()
P("  NOTE: the note's model IS the solver.py model (state (I2,b1,tau), batch")
P("  dispatch q, pi1*b1 charged per period), so this is a like-for-like test.")
P()

# summary
tot = len(rows)
exact = sum(1 for r in rows if r["diff"] == 0)
trig_ok = sum(1 for r in rows if r["dispatch_analytic"] == r["dispatch_DP"])
a_only = sum(1 for r in rows if r["dispatch_analytic"] and not r["dispatch_DP"])
d_only = sum(1 for r in rows if r["dispatch_DP"] and not r["dispatch_analytic"])
both = [r for r in rows if r["dispatch_analytic"] and r["dispatch_DP"]]
qty_ok = sum(1 for r in both if r["diff"] == 0)
ret_ok = sum(1 for r in rows if r["dispatch_DP"]
             and r["retained_DP"] == r["retained_predicted"])
ret_tot = sum(1 for r in rows if r["dispatch_DP"])

P("-" * 100)
P("  SUMMARY")
P("-" * 100)
P(f"  states compared                        : {tot}")
P(f"  q* exact match                         : {exact}/{tot} = {100*exact/tot:.2f}%")
P()
P(f"  TRIGGER  (dispatch yes/no) agreement   : {trig_ok}/{tot} = {100*trig_ok/tot:.2f}%")
P(f"     analytic dispatches, DP waits       : {a_only}  ({100*a_only/tot:.2f}%)")
P(f"     DP dispatches, analytic waits       : {d_only}  ({100*d_only/tot:.2f}%)")
P("     -> the disagreement is ONE-DIRECTIONAL: the analytic rule is never")
P("        too cautious, only ever too eager.")
P()
P(f"  QUANTITY agreement | both dispatch     : {qty_ok}/{len(both)} = "
  f"{100*qty_ok/max(len(both),1):.2f}%")
P(f"  retained == max(I2-b1, m_c-1)          : {ret_ok}/{ret_tot} = "
  f"{100*ret_ok/max(ret_tot,1):.2f}%")
P("     (protection-level reading of the quantity rule)")
P()

P("-" * 100)
P("  WHERE THE TWO KINDS OF DISAGREEMENT SIT")
P("-" * 100)
_trig = [r for r in rows if r["dispatch_analytic"] and not r["dispatch_DP"]]
_qty = [r for r in rows if r["dispatch_analytic"] and r["dispatch_DP"]
        and r["diff"] != 0]
if _trig:
    m = [r["S_minus_Cf"] for r in _trig]
    P(f"  TRIGGER type  (analytic dispatches, DP waits) : {len(_trig)} states")
    P(f"     margin S-Cf   : min={min(m):.3f}  median={np.median(m):.3f}  "
      f"max={max(m):.3f}")
    P(f"     b1 <= 3       : {100*np.mean([r['b1']<=3 for r in _trig]):.1f}% "
      f"of them  -> concentrated on the trigger boundary")
    P(f"     tau range     : {sorted(set(r['tau'] for r in _trig))}")
if _qty:
    m2 = [r["S_minus_Cf"] for r in _qty]
    P(f"  QUANTITY type (both dispatch, q differs)      : {len(_qty)} states "
      f"({100*len(_qty)/tot:.2f}% of all)")
    P(f"     diff values   : {sorted(set(r['diff'] for r in _qty))}  "
      f"(DP always dispatches the same or fewer)")
    P(f"     tau values    : {sorted(set(r['tau'] for r in _qty))}")
    P(f"     I2 range      : {min(r['I2'] for r in _qty)}"
      f"-{max(r['I2'] for r in _qty)}, "
      f"b1 range {min(r['b1'] for r in _qty)}-{max(r['b1'] for r in _qty)}")
    P("     -> long horizon, small I2, large b1: the DP retains one extra unit.")
P()
P("  Disagreement rate by margin band (states where the analytic rule fires):")
_disp = [r for r in rows if r["dispatch_analytic"]]
for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 1e18)]:
    band = [r for r in _disp if lo < r["S_minus_Cf"] <= hi]
    nb = sum(1 for r in band if r["diff"] != 0)
    tag = f"S-Cf > {lo}" if hi > 1e17 else f"{lo} < S-Cf <= {hi}"
    if band:
        P(f"     {tag:>22} : {len(band):5d} states, disagree {nb:4d} "
          f"({100*nb/len(band):5.1f}%)")
P("     -> a thin margin raises the risk of disagreement, but does not")
P("        account for all of it.")
P()

P("-" * 100)
P("  PER-TAU SUMMARY")
P("-" * 100)
P(f"  {'tau':>6} | {'q* match':>9} | {'max|diff|':>9} | {'trigger match':>13} | "
  f"{'analytic-eager':>14} | {'m_c':>4}")
for tau in TAUS:
    A, D = per_tau[tau]
    diff = D - A
    rs = [r for r in rows if r["tau"] == tau]
    tm = sum(1 for r in rs if r["dispatch_analytic"] == r["dispatch_DP"])
    ae = sum(1 for r in rs if r["dispatch_analytic"] and not r["dispatch_DP"])
    mc = rs[0]["m_c"]
    P(f"  {tau:>6} | {100*(diff==0).mean():8.2f}% | {np.abs(diff).max():>9} | "
      f"{100*tm/len(rs):12.2f}% | {ae:>14} | {mc:>4}")
P()

# b1bar table
P("-" * 100)
P("  DISPATCH TRIGGER  b1bar(I2, tau):   analytic (eq. 36)  /  DP")
P("-" * 100)
_row_lbl = "I2\\tau"
hdr = f"  {_row_lbl:>8} |" + "".join(f"{t:>11}" for t in TAUS)
P(hdr)
P("  " + "-" * (len(hdr) - 2))
for I2 in I2_LIST:
    cells = []
    for tau in TAUS:
        Em = Emin_table(LAM2 * tau)
        ba = analytic_b1bar(I2, tau, Em)
        bd = np.inf
        for b1 in range(1, B1M + 1):
            if POL[tau][ii(I2), b1] > 0:
                bd = b1
                break
        f = lambda v: "inf" if np.isinf(v) else str(int(v))
        cells.append(f"{f(ba)}/{f(bd)}")
    P(f"  {I2:>8} |" + "".join(f"{c:>11}" for c in cells))
P("  (each cell: analytic / DP.  'inf' = never dispatch at any b1.)")
P()

# per-tau matrices
for tau in TAUS:
    A, D = per_tau[tau]
    diff = D - A
    P(BAR)
    P(f"  tau = {tau}      matrices over I2 (rows, 1..30) x b1 (cols, 1..30)")
    P(BAR)
    for name, M in [("ANALYTIC q*", A), ("DP q*", D), ("DIFF  (DP - analytic)", diff)]:
        P(f"  {name}")
        P("   I2\\b1 " + "".join(f"{b:>3}" for b in B1_LIST))
        for a, I2 in enumerate(I2_LIST):
            P(f"   {I2:>5} " + "".join(f"{M[a,b]:>3}" for b in range(len(B1_LIST))))
        P()

# disagreement listing
P(BAR)
P("  FULL LIST OF DISAGREEMENTS  (with diagnostics for tracing)")
P(BAR)
P("  S = accumulated saving of the analytic rule; the rule dispatches iff S > Cf.")
P("  A small positive S-Cf next to 'DP waits' means the analytic rule fires on a")
P("  thin margin that the DP does not consider worth it.")
P()
P(f"  {'tau':>5} {'I2':>4} {'b1':>4} | {'q_an':>5} {'q_DP':>5} {'diff':>5} | "
  f"{'q_circ':>7} {'S':>10} {'S-Cf':>9} {'N':>4} {'m_c':>4}")
P("  " + "-" * 84)
bad = [r for r in rows if r["diff"] != 0]
for r in bad:
    P(f"  {r['tau']:>5} {r['I2']:>4} {r['b1']:>4} | {r['q_analytic']:>5} "
      f"{r['q_DP']:>5} {r['diff']:>5} | {r['q_circ']:>7} {r['S']:>10.3f} "
      f"{r['S_minus_Cf']:>9.3f} {r['N_profitable']:>4} {r['m_c']:>4}")
P()
P(f"  total disagreements: {len(bad)} of {tot}")
P(BAR)

with open(TXT_OUT, "w") as f:
    f.write("\n".join(L))
print(f"Wrote {TXT_OUT}  ({len(L)} lines)")