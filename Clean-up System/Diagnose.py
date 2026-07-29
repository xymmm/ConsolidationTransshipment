"""
diagnose.py — separate three questions:
  (A) is the closed form Vw (eq.20) correct as the NO-DISPATCH value function?
  (B) is the analytic q* (eq.25-26) the exact argmin against that Vw?
  (C) how far is that policy from the FULL DP optimum, and in which direction?
"""
import numpy as np
from math import exp

T = 5.0
LAM1, LAM2 = 5.0, 3.0
H, PI1, PI2 = 1.0, 6.0, 6.0
CF, CU = 8.0, 1.0
N = 1600

TAUS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
I2_LIST = list(range(1, 31))
B1_LIST = list(range(1, 31))

IMAX, IMIN, B1M = 30, -60, 110
dt = T / N
p1, p2 = LAM1 * dt, LAM2 * dt
p0 = 1 - p1 - p2
assert p0 > 0

I2v = np.arange(IMIN, IMAX + 1)
b1v = np.arange(0, B1M + 1)
I2g, b1g = I2v[:, None], b1v[None, :]
cI = lambda x: np.clip(x, IMIN, IMAX) - IMIN
cB = lambda x: np.clip(x, 0, B1M)
ii = lambda I2: int(I2 - IMIN)

flow = dt * (H * np.maximum(0, I2g) + PI1 * b1g + PI2 * np.maximum(0, -I2g))
flow = np.broadcast_to(flow, (len(I2v), B1M + 1)).copy()

keep = {min(N, max(1, round(t / dt))): t for t in TAUS}

# ---------------------------------------------------------------- (1) no-dispatch DP
Vw = np.zeros((len(I2v), B1M + 1))
VW = {}
for n in range(1, N + 1):
    Vw = flow + (p0 * Vw[cI(I2g), cB(b1g)]
                 + p1 * Vw[cI(I2g), cB(b1g + 1)]
                 + p2 * Vw[cI(I2g - 1), cB(b1g)])
    if n in keep:
        VW[keep[n]] = Vw.copy()

# ---------------------------------------------------------------- (2) full DP
V = np.zeros((len(I2v), B1M + 1))
POL, VF = {}, {}
qmax = IMAX
for n in range(1, N + 1):
    best = flow + (p0 * V[cI(I2g), cB(b1g)]
                   + p1 * V[cI(I2g), cB(b1g + 1)]
                   + p2 * V[cI(I2g - 1), cB(b1g)])
    bq = np.zeros(best.shape, np.int16)
    for q in range(1, qmax + 1):
        feas = (I2g >= q) & (b1g >= q)
        if not feas.any():
            continue
        I2a = np.broadcast_to(I2g - q, best.shape)
        b1a = np.broadcast_to(b1g - q, best.shape)
        cost = (CF + CU * q
                + dt * (H * np.maximum(0, I2a) + PI1 * b1a + PI2 * np.maximum(0, -I2a))
                + p0 * V[cI(I2a), cB(b1a)]
                + p1 * V[cI(I2a), cB(b1a + 1)]
                + p2 * V[cI(I2a - 1), cB(b1a)])
        cost = np.where(feas, cost, np.inf)
        upd = cost < best - 1e-12
        best = np.where(upd, cost, best)
        bq = np.where(upd, q, bq)
    V = best
    if n in keep:
        POL[keep[n]] = bq.copy()
        VF[keep[n]] = V.copy()

# ---------------------------------------------------------------- analytic side
def Emin_table(mu, mmax=200):
    out = np.zeros(mmax + 1)
    pmf, cb, M = exp(-mu), 0.0, 0.0
    for m in range(1, mmax + 1):
        M += max(1.0 - (cb + pmf), 0.0)
        out[m] = M
        cb += pmf
        pmf *= mu / m
    return out

def beta(I2, tau):
    mu = LAM2 * tau
    s, pmf = 0.0, exp(-mu)
    for k in range(0, I2 + 1):
        s += pmf * (I2 - k) * (I2 - k + 1)
        pmf *= mu / (k + 1)
    return ((PI1 * LAM1 + PI2 * LAM2) / 2 * tau**2 - PI2 * I2 * tau
            + (H + PI2) * I2 * (I2 + 1) / (2 * LAM2) - (H + PI2) / (2 * LAM2) * s)

def Vw_formula(I2, b1, tau):
    return PI1 * b1 * tau + beta(I2, tau)

def analytic_q(I2, b1, tau, Em):
    d = lambda m: (H + PI2) / LAM2 * Em[m] - CU + (PI1 - PI2) * tau
    Npos = sum(1 for m in range(1, I2 + 1) if d(m) > 0)
    qc = min(b1, Npos)
    if qc == 0:
        return 0, 0.0
    S = sum(d(I2 - i) for i in range(qc))
    return (qc if S > CF else 0), S

# ---------------------------------------------------------------- TEST A: eq (20)
print("=" * 78)
print("TEST A   closed form Vw (eq.20)  vs  no-dispatch DP")
print("=" * 78)
errs = []
for tau in TAUS:
    e = 0.0
    for I2 in range(0, 31):
        for b1 in range(0, 31):
            e = max(e, abs(VW[tau][ii(I2), b1] - Vw_formula(I2, b1, tau)))
    errs.append(e)
    print(f"   tau={tau:>4}   max |DP - formula| = {e:.4f}   "
          f"(scale |Vw| ~ {abs(Vw_formula(15,15,tau)):.1f})")
print(f"   -> worst {max(errs):.4f}; dt = {dt:.5f}, so O(dt) discretisation only.")
print()

# ---------------------------------------------------------------- TEST B: eq (25)-(26)
print("=" * 78)
print("TEST B   analytic q* (eq.25-26)  vs  exact argmin against the SAME Vw")
print("=" * 78)
tot = mism = 0
for tau in TAUS:
    Em = Emin_table(LAM2 * tau)
    bad = 0
    for I2 in I2_LIST:
        for b1 in B1_LIST:
            qa, _ = analytic_q(I2, b1, tau, Em)
            cands = [(CU * q + (CF if q > 0 else 0.0) + VW[tau][ii(I2 - q), b1 - q])
                     for q in range(0, min(I2, b1) + 1)]
            qhat = int(np.argmin(cands))
            tot += 1
            if qa != qhat:
                bad += 1
                mism += 1
    print(f"   tau={tau:>4}   mismatches = {bad}/900")
print(f"   TOTAL {tot - mism}/{tot} = {100*(tot-mism)/tot:.2f}% agreement")
print()

# ---------------------------------------------------------------- TEST C: vs full DP
print("=" * 78)
print("TEST C   analytic q*  vs  FULL DP q*")
print("=" * 78)
rows = []
for tau in TAUS:
    Em = Emin_table(LAM2 * tau)
    for I2 in I2_LIST:
        for b1 in B1_LIST:
            qa, S = analytic_q(I2, b1, tau, Em)
            qd = int(POL[tau][ii(I2), b1])
            rows.append((tau, I2, b1, qa, qd, S))
eq = sum(1 for r in rows if r[3] == r[4])
ae = sum(1 for r in rows if r[3] > 0 and r[4] == 0)
de = sum(1 for r in rows if r[4] > 0 and r[3] == 0)
qty = sum(1 for r in rows if r[3] > 0 and r[4] > 0 and r[3] != r[4])
over = sum(1 for r in rows if r[3] > r[4])
under = sum(1 for r in rows if r[3] < r[4])
print(f"   exact q* match            : {eq}/{len(rows)} = {100*eq/len(rows):.2f}%")
print(f"   analytic dispatches, DP waits : {ae}")
print(f"   DP dispatches, analytic waits : {de}")
print(f"   both dispatch, q differs      : {qty}")
print(f"   q_analytic > q_DP : {over}      q_analytic < q_DP : {under}")
print()

# ---------------------------------------------------------------- marginal backorder cost
print("=" * 78)
print("SMOKING GUN   marginal cost of one backorder:  pi1*tau  vs  Delta_b1 W")
print("=" * 78)
print(f"   {'tau':>5} {'I2':>4} | {'pi1*tau':>9} {'D_b1 Vw':>9} {'D_b1 W(DP)':>11} {'option value':>13}")
for tau in [0.5, 1.0, 2.0, 5.0]:
    for I2 in [3, 10, 30]:
        b1 = 5
        dw = VF[tau][ii(I2), b1 + 1] - VF[tau][ii(I2), b1]
        dv = VW[tau][ii(I2), b1 + 1] - VW[tau][ii(I2), b1]
        print(f"   {tau:>5} {I2:>4} | {PI1*tau:>9.3f} {dv:>9.3f} {dw:>11.3f} {dv-dw:>13.3f}")
print()

# ---------------------------------------------------------------- b1bar tables
def b1bar_analytic(I2, tau, Em):
    d = lambda m: (H + PI2) / LAM2 * Em[m] - CU + (PI1 - PI2) * tau
    S = 0.0
    for b in range(1, I2 + 1):
        dm = d(I2 - b + 1)
        if dm <= 0:
            break
        S += dm
        if S > CF:
            return b
    return np.inf

def b1bar_dp(I2, tau):
    for b1 in range(1, B1M + 1):
        if POL[tau][ii(I2), b1] > 0:
            return b1
    return np.inf

print("=" * 78)
print("b1bar(I2,tau):   analytic / DP        ('-' = +inf)")
print("=" * 78)
hdr = f"  {'I2':>4} |" + "".join(f"{t:>11}" for t in TAUS)
print(hdr)
f = lambda v: "-" if np.isinf(v) else str(int(v))
BB_A, BB_D = {}, {}
for I2 in I2_LIST:
    cells = []
    for tau in TAUS:
        Em = Emin_table(LAM2 * tau)
        a, d_ = b1bar_analytic(I2, tau, Em), b1bar_dp(I2, tau)
        BB_A[(I2, tau)], BB_D[(I2, tau)] = a, d_
        cells.append(f"{f(a)}/{f(d_)}")
    print(f"  {I2:>4} |" + "".join(f"{c:>11}" for c in cells))
print()
viol = sum(1 for I2 in I2_LIST for tau in TAUS if BB_A[(I2, tau)] > BB_D[(I2, tau)])
print(f"   states where b1bar_analytic > b1bar_DP (analytic too cautious): {viol}")

# monotonicity of the DP threshold
bad_I, bad_t = 0, 0
for tau in TAUS:
    for k in range(len(I2_LIST) - 1):
        if BB_D[(I2_LIST[k + 1], tau)] > BB_D[(I2_LIST[k], tau)]:
            bad_I += 1
for I2 in I2_LIST:
    for k in range(len(TAUS) - 1):
        if BB_D[(I2, TAUS[k + 1])] > BB_D[(I2, TAUS[k])]:
            bad_t += 1
print(f"   DP b1bar monotone non-increasing in I2 : violations = {bad_I}")
print(f"   DP b1bar monotone non-increasing in tau: violations = {bad_t}")

np.save('/home/claude/pol.npy', np.array([POL[t] for t in TAUS]))
import pickle
pickle.dump({'BB_A': BB_A, 'BB_D': BB_D, 'TAUS': TAUS, 'I2_LIST': I2_LIST},
            open('/home/claude/bb.pkl', 'wb'))