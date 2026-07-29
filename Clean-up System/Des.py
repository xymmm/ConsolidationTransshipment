"""
des.py — exact continuous-time discrete-event simulation, common random numbers.

The arrival processes are EXOGENOUS (retailer 1 is stocked out, so all its demand
is backlogged; retailer 2's demand is served or backlogged).  Dispatch decisions
never change arrival times.  So one set of sample paths can be reused for every
policy, giving a paired comparison with zero discretisation error.

Policies evaluated on identical paths:
   NEVER      never dispatch                      -> should equal Vw   (eq. 20)
   ONESHOT    dispatch q* optimally at t=0, then never again
                                                  -> should equal Vd   (eq. 27)
   ANALYTIC   the note's threshold policy: dispatch q* whenever b1 >= b1bar
   DP         the backward-induction optimal policy
"""
import numpy as np
from math import exp

LAM1, LAM2, H, PI1, PI2, CF, CU = 5.0, 3.0, 1.0, 6.0, 6.0, 8.0, 1.0
TAU0 = 5.0
NGRID, IMAX, IMIN, B1M = 1600, 40, -60, 130
dtg = TAU0 / NGRID
R = 400_000
KMAX = 140
rng = np.random.default_rng(20260722)

# ────────────────────────────── analytic tables on the tau grid
def Em_table(mu, mmax=250):
    out = np.zeros(mmax + 1); pmf, cb, M = exp(-mu), 0.0, 0.0
    for m in range(1, mmax + 1):
        M += max(1.0 - (cb + pmf), 0.0); out[m] = M; cb += pmf; pmf *= mu / m
    return out

QAN = np.zeros((NGRID + 1, IMAX + 1, B1M + 1), np.int16)   # analytic q*(tau,I2,b1)
for n in range(1, NGRID + 1):
    tau = n * dtg
    Em = Em_table(LAM2 * tau)
    d = (H + PI2) / LAM2 * Em[:IMAX + 1] - CU + (PI1 - PI2) * tau
    pos = d[1:IMAX + 1] > 0
    for I2 in range(1, IMAX + 1):
        Npos = int(pos[:I2].sum())
        if Npos == 0:
            continue
        csum = np.cumsum([d[I2 - i] for i in range(Npos)])
        for b1 in range(1, B1M + 1):
            qc = min(b1, Npos)
            if csum[qc - 1] > CF:
                QAN[n, I2, b1] = qc

# ────────────────────────────── DP policy on the same grid
p1, p2 = LAM1 * dtg, LAM2 * dtg
p0 = 1 - p1 - p2
I2v = np.arange(IMIN, IMAX + 1)
I2g, b1g = I2v[:, None], np.arange(0, B1M + 1)[None, :]
cI = lambda x: np.clip(x, IMIN, IMAX) - IMIN
cB = lambda x: np.clip(x, 0, B1M)
sh = (len(I2v), B1M + 1)
flow = np.broadcast_to(dtg * (H * np.maximum(0, I2g) + PI1 * b1g
                              + PI2 * np.maximum(0, -I2g)), sh).copy()
V = np.zeros(sh)
QDP = np.zeros((NGRID + 1, IMAX + 1, B1M + 1), np.int16)
for n in range(1, NGRID + 1):
    best = flow + (p0 * V[cI(I2g), cB(b1g)] + p1 * V[cI(I2g), cB(b1g + 1)]
                   + p2 * V[cI(I2g - 1), cB(b1g)])
    bq = np.zeros(sh, np.int16)
    for q in range(1, IMAX + 1):
        feas = (I2g >= q) & (b1g >= q)
        I2a = np.broadcast_to(I2g - q, sh); b1a = np.broadcast_to(b1g - q, sh)
        cost = (CF + CU * q + dtg * (H * np.maximum(0, I2a) + PI1 * b1a
                                     + PI2 * np.maximum(0, -I2a))
                + p0 * V[cI(I2a), cB(b1a)] + p1 * V[cI(I2a), cB(b1a + 1)]
                + p2 * V[cI(I2a - 1), cB(b1a)])
        cost = np.where(feas, cost, np.inf)
        upd = cost < best - 1e-12
        best = np.where(upd, cost, best); bq = np.where(upd, q, bq)
    V = best
    QDP[n, :, :] = bq[cI(np.arange(0, IMAX + 1))[:, None],
                      np.arange(0, B1M + 1)[None, :]]
VDP_final = V

# ────────────────────────────── closed forms
def beta(I2, tau):
    mu = LAM2 * tau; s, pmf = 0.0, exp(-mu)
    for k in range(0, I2 + 1):
        s += pmf * (I2 - k) * (I2 - k + 1); pmf *= mu / (k + 1)
    return ((PI1 * LAM1 + PI2 * LAM2) / 2 * tau**2 - PI2 * I2 * tau
            + (H + PI2) * I2 * (I2 + 1) / (2 * LAM2) - (H + PI2) / (2 * LAM2) * s)

Vw_f = lambda I2, b1, tau: PI1 * b1 * tau + beta(I2, tau)

def Vd_f(I2, b1, tau):
    Em = Em_table(LAM2 * tau)
    d = lambda m: (H + PI2) / LAM2 * Em[m] - CU + (PI1 - PI2) * tau
    Npos = sum(1 for m in range(1, I2 + 1) if d(m) > 0)
    qc = min(b1, Npos)
    S = sum(d(I2 - i) for i in range(qc)) if qc else 0.0
    return Vw_f(I2, b1, tau) - max(S - CF, 0.0)

# ────────────────────────────── sample paths (shared)
lam = LAM1 + LAM2
gaps = rng.exponential(1.0 / lam, size=(R, KMAX))
times = np.cumsum(gaps, axis=1)
is1 = rng.random((R, KMAX)) < LAM1 / lam          # True -> retailer 1 arrival
assert (times[:, -1] > TAU0).all(), "KMAX too small"

def simulate(I0, b0, mode):
    """mode in {'never','oneshot','analytic','dp'}"""
    I2 = np.full(R, I0, np.int64)
    b1 = np.full(R, b0, np.int64)
    cost = np.zeros(R)
    t = np.zeros(R)
    fired = np.zeros(R, bool)

    def maybe_dispatch(tau_rem, allow):
        nonlocal I2, b1, cost, fired
        if mode == 'never':
            return
        n = np.clip(np.round(tau_rem / dtg).astype(np.int64), 1, NGRID)
        tab = QAN if mode in ('analytic', 'oneshot') else QDP
        q = tab[n, np.clip(I2, 0, IMAX), np.clip(b1, 0, B1M)].astype(np.int64)
        q = np.where((I2 > 0) & (b1 > 0), np.minimum(q, np.minimum(I2, b1)), 0)
        if mode == 'oneshot':
            q = np.where(fired, 0, q)
        q = np.where(allow, q, 0)
        act = q > 0
        cost += np.where(act, CF + CU * q, 0.0)
        I2 -= q; b1 -= q
        fired |= act

    maybe_dispatch(np.full(R, TAU0), np.ones(R, bool))
    for k in range(KMAX):
        tk = times[:, k]
        seg = np.minimum(tk, TAU0) - t
        seg = np.maximum(seg, 0.0)
        cost += seg * (H * np.maximum(I2, 0) + PI1 * b1 + PI2 * np.maximum(-I2, 0))
        t = np.minimum(tk, TAU0)
        alive = tk < TAU0
        if not alive.any():
            break
        step1 = alive & is1[:, k]
        step2 = alive & ~is1[:, k]
        b1 += step1.astype(np.int64)
        I2 -= step2.astype(np.int64)
        maybe_dispatch(TAU0 - t, step1)      # only a retailer-1 arrival can trigger
    return cost

print(f"replications = {R:,}   horizon tau = {TAU0}   terminal cost = 0")
print()
hdr = (f"{'(I2,b1)':>9} | {'Vw (eq20)':>10} {'sim NEVER':>18} | "
       f"{'Vd (eq27)':>10} {'sim ONESHOT':>18} | {'sim ANALYTIC':>18} | "
       f"{'V^DP':>9} {'sim DP':>18}")
print(hdr); print("-" * len(hdr))
for (I0, b0) in [(30, 0), (30, 2), (20, 5), (12, 3), (8, 1)]:
    cn = simulate(I0, b0, 'never')
    co = simulate(I0, b0, 'oneshot')
    ca = simulate(I0, b0, 'analytic')
    cd = simulate(I0, b0, 'dp')
    vdp = VDP_final[I0 - IMIN, b0]
    f = lambda c: f"{c.mean():10.3f}±{1.96*c.std(ddof=1)/np.sqrt(R):5.3f}"
    print(f"{str((I0,b0)):>9} | {Vw_f(I0,b0,TAU0):10.3f} {f(cn):>18} | "
          f"{Vd_f(I0,b0,TAU0):10.3f} {f(co):>18} | {f(ca):>18} | "
          f"{vdp:9.3f} {f(cd):>18}")

print()
print("paired differences (common random numbers, 95% CI):")
for (I0, b0) in [(30, 0), (30, 2), (20, 5), (12, 3)]:
    co = simulate(I0, b0, 'oneshot')
    ca = simulate(I0, b0, 'analytic')
    cd = simulate(I0, b0, 'dp')
    d1 = co - ca; d2 = ca - cd
    g = lambda d: f"{d.mean():8.3f} ± {1.96*d.std(ddof=1)/np.sqrt(R):.3f}"
    print(f"  ({I0},{b0}): ONESHOT - ANALYTIC = {g(d1)}    "
          f"ANALYTIC - DP = {g(d2)}")