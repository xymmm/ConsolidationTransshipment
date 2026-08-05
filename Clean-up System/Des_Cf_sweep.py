"""
des_Cf_sweep.py  —  issue 2: how DP and analytic decisions diverge, and how
                     the divergence grows with the fixed cost Cf.
=============================================================================
For each Cf in a sweep, on ONE shared set of continuous-time sample paths:

  * build the exact DP policy (backward induction on the note's model);
  * run the DP policy and the analytic threshold policy (Eq. 34+36) on the
    SAME arrivals, common random numbers;
  * report expected cost of each, the cost gap, the average number of
    dispatches (= how much each policy batches), and the fraction of
    decision points where the two policies disagree.

The headline is the last two columns: as Cf grows, the analytic policy keeps
dispatching in small batches while the DP waits and batches, so the decision
disagreement and the cost gap both grow.

Also prints, for one Cf, an event-by-event trace of a few paths with the two
policies side by side (set TRACE_PATHS > 0).

Standing example: lam1=5, lam2=3, h=1, pi1=pi2=6, cu=1, T=5, terminal 0.
Only Cf is swept. numpy only.
"""
import numpy as np
from math import exp

# ── fixed parameters; only CF is swept ───────────────────────────────
LAM1, LAM2, H, PI1, PI2, CU, T = 5.0, 3.0, 1.0, 6.0, 6.0, 1.0, 5.0
CF_LIST = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]     # the sweep
NGRID, IMAX, IMIN, B1M = 800, 40, -60, 130
R = 200_000
KMAX = 140
TRACE_CF = 8.0            # Cf at which to print event traces
TRACE_PATHS = 2           # 0 to skip
TRACE_STATE = (30, 2)     # (I2, b1) at tau = T for the traced paths
SEED = 20260803

dt = T / NGRID
p1, p2 = LAM1 * dt, LAM2 * dt
p0 = 1 - p1 - p2
I2v = np.arange(IMIN, IMAX + 1)
I2g = I2v[:, None]
b1g = np.arange(0, B1M + 1)[None, :]
sh = (len(I2v), B1M + 1)
cI = lambda x: np.clip(x, IMIN, IMAX) - IMIN
cB = lambda x: np.clip(x, 0, B1M)
flow = np.broadcast_to(dt * (H * np.maximum(0, I2g) + PI1 * b1g
                             + PI2 * np.maximum(0, -I2g)), sh).copy()


def Em_table(mu, mmax=250):
    E = np.zeros(mmax + 1)
    pmf, cdf, M = exp(-mu), exp(-mu), 0.0
    for m in range(1, mmax + 1):
        M += 1.0 - cdf
        E[m] = M
        pmf *= mu / m
        cdf += pmf
    return E


def build_QAN(cf):
    """Analytic dispatch quantity table q*(n, I2, b1) from Eq. (33)-(36)."""
    QAN = np.zeros((NGRID + 1, IMAX + 1, B1M + 1), np.int16)
    for n in range(1, NGRID + 1):
        tau = n * dt
        E = Em_table(LAM2 * tau, IMAX)
        d = (H + PI2) / LAM2 * E - CU + (PI1 - PI2) * tau
        Np = int((d[1:IMAX + 1] > 0).sum())
        if Np == 0:
            continue
        peel = np.array([d[IMAX - i] if False else 0.0 for i in range(0)])
        # cumulative peel of the top Np positive margins, per I2
        for I2 in range(1, IMAX + 1):
            npos = int((d[1:I2 + 1] > 0).sum())
            if npos == 0:
                continue
            cs = np.cumsum([d[I2 - i] for i in range(npos)])
            for b1 in range(1, B1M + 1):
                qc = min(b1, npos)
                if cs[qc - 1] > cf:
                    QAN[n, I2, b1] = qc
    return QAN


def build_QDP(cf):
    """Optimal DP policy table by backward induction."""
    V = np.zeros(sh)
    QDP = np.zeros((NGRID + 1, IMAX + 1, B1M + 1), np.int16)
    for n in range(1, NGRID + 1):
        best = flow + (p0 * V[cI(I2g), cB(b1g)]
                       + p1 * V[cI(I2g), cB(b1g + 1)]
                       + p2 * V[cI(I2g - 1), cB(b1g)])
        bq = np.zeros(sh, np.int16)
        for q in range(1, IMAX + 1):
            feas = (I2g >= q) & (b1g >= q)
            I2a = np.broadcast_to(I2g - q, sh)
            b1a = np.broadcast_to(b1g - q, sh)
            c = (cf + CU * q + dt * (H * np.maximum(0, I2a) + PI1 * b1a
                                     + PI2 * np.maximum(0, -I2a))
                 + p0 * V[cI(I2a), cB(b1a)]
                 + p1 * V[cI(I2a), cB(b1a + 1)]
                 + p2 * V[cI(I2a - 1), cB(b1a)])
            c = np.where(feas, c, np.inf)
            u = c < best - 1e-12
            best = np.where(u, c, best)
            bq = np.where(u, q, bq)
        V = best
        QDP[n] = bq[cI(np.arange(0, IMAX + 1))[:, None],
                    np.arange(0, B1M + 1)[None, :]]
    return QDP


# ── shared sample paths ──────────────────────────────────────────────
rng = np.random.default_rng(SEED)
lam = LAM1 + LAM2
times = np.cumsum(rng.exponential(1 / lam, size=(R, KMAX)), axis=1)
is1 = rng.random((R, KMAX)) < LAM1 / lam
assert (times[:, -1] > T).all(), "increase KMAX"


def simulate(Q, cf):
    """Run policy table Q on the shared paths; return (cost, n_dispatch)."""
    I2 = np.full(R, TRACE_STATE[0], np.int64)
    b1 = np.full(R, TRACE_STATE[1], np.int64)
    cost = np.zeros(R)
    ndisp = np.zeros(R, np.int64)
    t = np.zeros(R)

    def act(tau_rem, trig):
        nonlocal I2, b1, cost, ndisp
        n = np.clip(np.round(tau_rem / dt).astype(np.int64), 1, NGRID)
        q = Q[n, np.clip(I2, 0, IMAX), np.clip(b1, 0, B1M)].astype(np.int64)
        q = np.where((I2 > 0) & (b1 > 0) & trig, np.minimum(q, np.minimum(I2, b1)), 0)
        act_ = q > 0
        cost += np.where(act_, cf + CU * q, 0.0)
        ndisp += act_.astype(np.int64)
        I2 -= q
        b1 -= q

    act(T, np.ones(R, bool))       # decision at tau = T
    for k in range(KMAX):
        tk = times[:, k]
        seg = np.maximum(np.minimum(tk, T) - t, 0.0)
        cost += seg * (H * np.maximum(I2, 0) + PI1 * b1 + PI2 * np.maximum(-I2, 0))
        t = np.minimum(tk, T)
        alive = tk < T
        if not alive.any():
            break
        b1 += (alive & is1[:, k]).astype(np.int64)
        I2 -= (alive & ~is1[:, k]).astype(np.int64)
        act(T - t, alive)
    return cost, ndisp


def decision_disagreement(QDP, QAN):
    """Fraction of (n, I2, b1) states, over a representative grid, where the
    two policies make a different wait/dispatch choice. Restricted to the
    states the process actually reaches: I2 in 1..20, b1 in 1..10."""
    ns = np.linspace(1, NGRID, 60).astype(int)
    I2s = np.arange(1, 21)
    b1s = np.arange(1, 11)
    diff = tot = 0
    for n in ns:
        dp = QDP[n][np.ix_(I2s, b1s)] > 0
        an = QAN[n][np.ix_(I2s, b1s)] > 0
        diff += int((dp != an).sum())
        tot += dp.size
    return diff / tot


# ── the sweep ────────────────────────────────────────────────────────
print(f"shared paths: {R:,}   start (I2,b1)={TRACE_STATE}   tau0={T}")
print(f"{'Cf':>5} | {'DP cost':>9} | {'AN cost':>9} | {'gap':>7} | "
      f"{'DP disp':>7} | {'AN disp':>7} | {'decision disagree':>17}")
print("-" * 82)
for cf in CF_LIST:
    QDP = build_QDP(cf)
    QAN = build_QAN(cf)
    cdp, ndp = simulate(QDP, cf)
    can, nan = simulate(QAN, cf)
    dis = decision_disagreement(QDP, QAN)
    print(f"{cf:>5.0f} | {cdp.mean():>9.3f} | {can.mean():>9.3f} | "
          f"{can.mean()-cdp.mean():>7.3f} | {ndp.mean():>7.2f} | "
          f"{nan.mean():>7.2f} | {100*dis:>16.2f}%")

print("\nReading: as Cf grows, the DP batches (DP disp falls) while the")
print("analytic policy keeps dispatching in small lots (AN disp stays high),")
print("so both the decision disagreement and the cost gap widen. At Cf=0 the")
print("two policies nearly coincide.")


# ── optional event-by-event trace at TRACE_CF ────────────────────────
def _analytic_q_exact(I2, b1, tau, cf):
    if I2 < 1 or b1 < 1 or tau <= 0:
        return 0
    E = Em_table(LAM2 * tau, max(I2, 1))
    d = [0.0] + [(H + PI2) / LAM2 * E[m] - CU + (PI1 - PI2) * tau
                 for m in range(1, I2 + 1)]
    npos = sum(1 for m in range(1, I2 + 1) if d[m] > 0)
    if npos == 0:
        return 0
    qc = min(b1, npos)
    S = sum(d[I2 - i] for i in range(qc))
    return qc if S > cf else 0


if TRACE_PATHS > 0:
    print("\n" + "=" * 82)
    print(f"EVENT TRACE at Cf = {TRACE_CF}: identical arrivals, DP vs analytic")
    print("=" * 82)
    QDP_t = build_QDP(TRACE_CF)
    trng = np.random.default_rng(SEED + 1)
    for path in range(1, TRACE_PATHS + 1):
        n1 = trng.poisson(LAM1 * T)
        n2 = trng.poisson(LAM2 * T)
        ev = sorted([(x, 'R1') for x in np.sort(trng.random(n1)) * T]
                    + [(x, 'R2') for x in np.sort(trng.random(n2)) * T])
        st = {'DP': [TRACE_STATE[0], TRACE_STATE[1], 0.0, 0],
              'AN': [TRACE_STATE[0], TRACE_STATE[1], 0.0, 0]}
        print(f"\n  PATH {path}  start (I2,b1)={TRACE_STATE}, {n1} R1, {n2} R2")
        print(f"  {'t':>6} {'tau':>6} {'ev':>3} | {'DP':<26} | {'analytic':<26}")

        def decide(tag, tau, trig):
            I2, b1 = st[tag][0], st[tag][1]
            if tag == 'DP':
                n = int(np.clip(round(tau / dt), 1, NGRID))
                q = int(QDP_t[n, min(max(I2, 0), IMAX), min(b1, B1M)]) \
                    if (trig and I2 > 0 and b1 > 0) else 0
                q = min(q, min(I2, b1)) if q > 0 else 0
            else:
                q = _analytic_q_exact(I2, b1, tau, TRACE_CF) if trig else 0
            if q > 0:
                st[tag][2] += TRACE_CF + CU * q
                st[tag][0] -= q
                st[tag][1] -= q
                st[tag][3] += 1
                return f"({I2:>2},{b1:>2}) DISP q={q}->({I2-q},{b1-q})"
            return f"({I2:>2},{b1:>2}) wait"

        d0 = decide('DP', T, True)
        a0 = decide('AN', T, True)
        print(f"  {0.0:>6.3f} {T:>6.3f} {'--':>3} | {d0:<26} | {a0:<26}")
        t = 0.0
        for (te, kind) in ev:
            for tag in ('DP', 'AN'):
                I2, b1 = st[tag][0], st[tag][1]
                st[tag][2] += (te - t) * (H * max(I2, 0) + PI1 * b1
                                          + PI2 * max(-I2, 0))
            t = te
            for tag in ('DP', 'AN'):
                if kind == 'R1':
                    st[tag][1] += 1
                else:
                    st[tag][0] -= 1
            tau = T - t
            dd = decide('DP', tau, True)
            aa = decide('AN', tau, kind == 'R1')
            flag = "" if (dd.split()[1] == aa.split()[1]) else "   <-- differ"
            print(f"  {t:>6.3f} {tau:>6.3f} {kind:>3} | {dd:<26} | {aa:<26}{flag}")
        for tag in ('DP', 'AN'):
            I2, b1 = st[tag][0], st[tag][1]
            st[tag][2] += (T - t) * (H * max(I2, 0) + PI1 * b1
                                     + PI2 * max(-I2, 0))
        print(f"  totals: DP {st['DP'][2]:.2f} ({st['DP'][3]} disp)   "
              f"analytic {st['AN'][2]:.2f} ({st['AN'][3]} disp)")