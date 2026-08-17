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

Model alignment (audited):
  - cost integrand h*I2+ + pi2*I2- + pi1*b1 integrated EXACTLY between events;
    dispatch cost Cf + cu*q; terminal cost 0: matches TC in the note's Eq. (3)
    and the discrete model's g(.) as dt->0.
  - R1 arrival -> b1+1; R2 demand -> I2-1 (served or backlogged): matches
    I2(t) = I2^0 - D2(t) - Y(t), b1(t) = D1(t) - Y(t).
  - trigger times: the ANALYTIC policy is checked at t=0 and after R1
    arrivals only. This is EXACT for that policy: by Theorems 3-4 its
    threshold b1bar is non-increasing in I2 and tau, so R2 arrivals and the
    passage of time can only raise the threshold, never trigger it, and after
    a dispatch the profitable batch is exhausted. The DP policy is NOT
    monotone (e.g. (6,3) waits while (5,3) dispatches), so it is checked
    after EVERY arrival. The only triggers still missed for DP are the
    time-triggered ones inside the two tau-windows at (6,3),(7,3); their
    occupancy is tiny, and missing them makes sim-DP an upper bound on the
    optimal cost, i.e. the reported ANALYTIC-DP gap is conservative.

TRACE: set TRACE_PATHS > 0 to print, for a few sample paths, every event and
the side-by-side decision of ANALYTIC vs DP on identical arrivals.
"""
import numpy as np
from math import exp

LAM1, LAM2, H, PI1, PI2, CF, CU = 5.0, 3.0, 1.0, 6.0, 6.0, 8.0, 1.0
TAU0 = 5.0
NGRID, IMAX, IMIN, B1M = 1600, 40, -60, 130
dtg = TAU0 / NGRID
R = 400_000
KMAX = 140
TRACE_PATHS = 3          # event-by-event printout for this many paths
TRACE_STATE = (30, 2)    # (I2, b1) at tau = TAU0 for the traced paths
RUN_TABLES = True        # the big Monte-Carlo tables
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
        trig = alive if mode == 'dp' else step1   # DP: any arrival can trigger
        maybe_dispatch(TAU0 - t, trig)
    return cost

if not RUN_TABLES:
    pass
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


# ────────────────────────────── TRACE: event-by-event, side by side
def _analytic_q_exact(I2, b1, tau):
    """Eq. (33)-(36) evaluated at the exact continuous tau (no grid)."""
    if I2 < 1 or b1 < 1 or tau <= 0:
        return 0, 0.0
    Em = Em_table(LAM2 * tau, mmax=max(I2, 1))
    d = [0.0] + [(H + PI2) / LAM2 * Em[m] - CU + (PI1 - PI2) * tau
                 for m in range(1, I2 + 1)]
    Npos = sum(1 for m in range(1, I2 + 1) if d[m] > 0)
    if Npos == 0:
        return 0, 0.0
    qc = min(b1, Npos)
    Ssum = sum(d[I2 - i] for i in range(qc))
    return (qc if Ssum > CF else 0), Ssum


def trace_paths(n_paths=TRACE_PATHS, I0=TRACE_STATE[0], b0=TRACE_STATE[1],
                seed=20260731):
    trng = np.random.default_rng(seed)
    for path in range(1, n_paths + 1):
        n1 = trng.poisson(LAM1 * TAU0)
        n2 = trng.poisson(LAM2 * TAU0)
        ev = sorted([(t, 'R1') for t in np.sort(trng.random(n1)) * TAU0]
                    + [(t, 'R2') for t in np.sort(trng.random(n2)) * TAU0])
        st = {'AN': [I0, b0, 0.0, 0], 'DP': [I0, b0, 0.0, 0]}  # I2,b1,cost,nCf
        print("=" * 108)
        print(f"  PATH {path}   start (I2,b1)=({I0},{b0})  tau0={TAU0}   "
              f"{n1} R1 arrivals, {n2} R2 demands")
        print("=" * 108)
        print(f"  {'#':>3} {'t':>7} {'tau':>7} {'ev':>3} | "
              f"{'ANALYTIC':<34} {'cum':>8} | {'DP':<28} {'cum':>8}")

        def decide(tag, tau, trigger_ok):
            I2, b1, c, nc = st[tag]
            if tag == 'AN':
                q, _ = _analytic_q_exact(I2, b1, tau) if trigger_ok else (0, 0)
            else:
                n = int(np.clip(round(tau / dtg), 1, NGRID))
                q = int(QDP[n, min(max(I2, 0), IMAX), min(b1, B1M)]) \
                    if (trigger_ok and I2 > 0 and b1 > 0) else 0
                q = min(q, min(I2, b1)) if q > 0 else 0
            if q > 0:
                st[tag][2] += CF + CU * q
                st[tag][0] -= q
                st[tag][1] -= q
                st[tag][3] += 1
                return f"({I2:>2},{b1:>2}) DISPATCH q={q} -> ({I2-q},{b1-q})"
            return f"({I2:>2},{b1:>2}) wait"

        a0 = decide('AN', TAU0, True)
        d0 = decide('DP', TAU0, True)
        print(f"  {0:>3} {0.0:>7.3f} {TAU0:>7.3f} {'--':>3} | "
              f"{a0:<34} {st['AN'][2]:>8.2f} | {d0:<28} {st['DP'][2]:>8.2f}")
        t = 0.0
        for k, (te, kind) in enumerate(ev, start=1):
            seg = te - t
            for tag in ('AN', 'DP'):
                I2, b1 = st[tag][0], st[tag][1]
                st[tag][2] += seg * (H * max(I2, 0) + PI1 * b1
                                     + PI2 * max(-I2, 0))
            t = te
            for tag in ('AN', 'DP'):
                if kind == 'R1':
                    st[tag][1] += 1
                else:
                    st[tag][0] -= 1
            tau = TAU0 - t
            # ANALYTIC: only an R1 arrival can trigger (Theorems 3-4);
            # DP: any arrival can trigger (its threshold is not monotone)
            a = decide('AN', tau, kind == 'R1')
            d = decide('DP', tau, True)
            print(f"  {k:>3} {t:>7.3f} {tau:>7.3f} {kind:>3} | "
                  f"{a:<34} {st['AN'][2]:>8.2f} | {d:<28} {st['DP'][2]:>8.2f}")
        for tag in ('AN', 'DP'):
            I2, b1 = st[tag][0], st[tag][1]
            st[tag][2] += (TAU0 - t) * (H * max(I2, 0) + PI1 * b1
                                        + PI2 * max(-I2, 0))
        print("-" * 108)
        print(f"  totals: ANALYTIC cost = {st['AN'][2]:8.2f}  "
              f"({st['AN'][3]} dispatches, fixed cost {st['AN'][3]*CF:.0f})    "
              f"DP cost = {st['DP'][2]:8.2f}  "
              f"({st['DP'][3]} dispatches, fixed cost {st['DP'][3]*CF:.0f})")
        print()


if TRACE_PATHS > 0:
    print()
    print("#" * 108)
    print("#  EVENT-BY-EVENT TRACE: identical arrivals, ANALYTIC vs DP decisions")
    print("#" * 108)
    trace_paths()