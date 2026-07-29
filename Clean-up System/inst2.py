import numpy as np
from math import exp

def experiment(LAM1, LAM2, H, PI1, PI2, CF, CU, T, N, IMAX, IMIN, B1M, TAUS,
               I2LIST, B1LIST, label):
    dt = T / N
    p1, p2 = LAM1 * dt, LAM2 * dt
    p0 = 1 - p1 - p2
    I2v = np.arange(IMIN, IMAX + 1)
    I2g, b1g = I2v[:, None], np.arange(0, B1M + 1)[None, :]
    cI = lambda x: np.clip(x, IMIN, IMAX) - IMIN
    cB = lambda x: np.clip(x, 0, B1M)
    sh = (len(I2v), B1M + 1)
    flow = np.broadcast_to(dt * (H * np.maximum(0, I2g) + PI1 * b1g
                                 + PI2 * np.maximum(0, -I2g)), sh).copy()
    keep = {min(N, max(1, round(t / dt))): t for t in TAUS}
    V = np.zeros(sh); POL = {}
    for n in range(1, N + 1):
        best = flow + (p0 * V[cI(I2g), cB(b1g)] + p1 * V[cI(I2g), cB(b1g + 1)]
                       + p2 * V[cI(I2g - 1), cB(b1g)])
        bq = np.zeros(sh, np.int16)
        for q in range(1, IMAX + 1):
            feas = (I2g >= q) & (b1g >= q)
            I2a = np.broadcast_to(I2g - q, sh); b1a = np.broadcast_to(b1g - q, sh)
            cost = (CF + CU * q + dt * (H * np.maximum(0, I2a) + PI1 * b1a
                                        + PI2 * np.maximum(0, -I2a))
                    + p0 * V[cI(I2a), cB(b1a)] + p1 * V[cI(I2a), cB(b1a + 1)]
                    + p2 * V[cI(I2a - 1), cB(b1a)])
            cost = np.where(feas, cost, np.inf)
            upd = cost < best - 1e-12
            best = np.where(upd, cost, best); bq = np.where(upd, q, bq)
        V = best
        if n in keep: POL[keep[n]] = bq.copy()

    def Em_table(mu, mmax=300):
        out = np.zeros(mmax + 1); pmf, cb, M = exp(-mu), 0.0, 0.0
        for m in range(1, mmax + 1):
            M += max(1.0 - (cb + pmf), 0.0); out[m] = M; cb += pmf; pmf *= mu / m
        return out

    ii = lambda I2: int(I2 - IMIN)
    tot = eq = ov = un = qty = 0
    mcs = {}
    for tau in TAUS:
        Em = Em_table(LAM2 * tau)
        d = lambda m: (H + PI2) / LAM2 * Em[m] - CU + (PI1 - PI2) * tau
        mc = next((m for m in range(1, 300) if d(m) > 0), None)
        mcs[tau] = mc
        for I2 in I2LIST:
            Npos = sum(1 for m in range(1, I2 + 1) if d(m) > 0)
            for b1 in B1LIST:
                qc = min(b1, Npos)
                S = sum(d(I2 - i) for i in range(qc)) if qc else 0.0
                qa = qc if (qc and S > CF) else 0
                qd = int(POL[tau][ii(I2), b1])
                tot += 1; eq += (qa == qd); ov += (qa > qd and qd == 0)
                un += (qd > 0 and qa == 0)
                qty += (qa > 0 and qd > 0 and qa != qd)
    print(f"[{label}]  m_c by tau: {mcs}")
    print(f"   q* exact {eq}/{tot} = {100*eq/tot:.2f}% | analytic-fires-DP-waits {ov}"
          f" | DP-fires-analytic-waits {un} | both fire but q differs {qty}")
    return POL, ii


# instance A = Section 6.2
experiment(5, 3, 1, 6, 6, 8, 1, 5.0, 1600, 30, -60, 110,
           [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
           list(range(1, 31)), list(range(1, 31)), "A: sec 6.2")

# instance B: large cu, small pi -> m_c > 1, so the protection level bites
experiment(5, 3, 1, 3, 3, 8, 4, 5.0, 1600, 30, -60, 110,
           [0.5, 1.0, 2.0, 3.0, 5.0],
           list(range(1, 31)), list(range(1, 31)), "B: cu=4, pi=3")

# instance C: pi1 > pi2
experiment(5, 3, 1, 10, 3, 8, 2, 5.0, 1600, 30, -60, 110,
           [0.5, 1.0, 2.0, 3.0, 5.0],
           list(range(1, 31)), list(range(1, 31)), "C: pi1=10 > pi2=3")
