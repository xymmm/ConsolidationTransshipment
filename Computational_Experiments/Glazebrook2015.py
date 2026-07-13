"""
glazebrook_2015.py
===============================================================================
Replication of the hybrid transshipment heuristic in:
  Glazebrook, K., Paterson, C., Rauscher, S. & Archibald, T. (2015).
  "Benefits of Hybrid Lateral Transshipments in Multi-Item Inventory
  Systems under Periodic Replenishment." Production and Operations
  Management 24(2), 311-324.

SCOPE (agreed 20260713): replicate the quasi-myopic index policy of
Section 4 (Eqs. 1-21) for a SINGLE item type (X=1). The paper's own
generality is multi-item with a general joint demand distribution
(Eq. 3 additively decomposes cost across item types), so the X=1 case
is an exact special case of the general formulas, not an approximation
of them -- setting X=1 simply removes the outer sum in Eq. (3).

Cost model implemented: LOST SALES (Lix>0, bix=0), using the closed
form of Eq. (11) for shortage cost (the paper's own numerical study,
Section 5, "we include results only for [lost sales]" -- both models
gave comparable results per the paper). The BACKORDER branch (Eq. 13)
is not implemented; flagged as a TODO if the backorder model is later
needed.

Non-stationary demand (time-varying customer arrival rate lambda_i(t))
IS supported: pass a callable lam_i(t) rather than a constant.

This module is written for FIDELITY to the paper's equations. A smoke
test at the bottom (i) computes the transshipment index D(uji|...) on a
small worked example and (ii) checks it against Theorem 1(a): the index
must be non-increasing in the sending location's stock level ILj, for
fixed everything else. This is an internal-consistency check drawn
directly from the paper (not an external ground truth), used to confirm
the implementation is at least self-consistent with a proved property.
===============================================================================
"""

import math


# ═══════════════════════════════════════════════════════════════════════
#  Poisson helper
# ═══════════════════════════════════════════════════════════════════════
def poisson_pmf_upto(mu, kmax):
    if kmax < 0:
        return []
    p = math.exp(-mu)
    out = [p]
    for k in range(1, kmax + 1):
        p *= mu / k
        out.append(p)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Demand-size convolution powers (needed for both P^n_j and the
#  compound-Poisson interval-demand distribution)
# ═══════════════════════════════════════════════════════════════════════
class DemandModel:
    """
    Single-item-type demand model for one location: customers arrive as
    a (possibly non-homogeneous) Poisson process with rate lam(t); each
    customer's demand size is iid with pmf fd (a list, fd[d] = P(single
    customer demands d units), d=1,2,...; fd[0] should be 0 -- every
    customer demands at least 1 unit, per the paper's setup).

    Caches convolution powers of fd (the pmf of the sum of m iid
    customer demands) up to whatever order is requested, since both
    v{hold} (via P^n_j) and v{lost} (via the interval-demand pmf) need
    them.
    """

    def __init__(self, lam, fd):
        """
        lam : float OR callable t -> rate at time t (non-stationary demand)
        fd  : list, fd[d] = P(single customer demand = d), d=0..dmax_cust
              (fd[0] must be 0.0)
        """
        self.lam = lam if callable(lam) else (lambda t, _c=lam: _c)
        self.fd = fd
        self._conv_cache = {0: [1.0]}   # S_0 is degenerate at 0

    def _conv_power(self, m):
        """pmf of S_m = sum of m iid customer demands (S_0 = 0 w.p.1)."""
        if m in self._conv_cache:
            return self._conv_cache[m]
        prev = self._conv_power(m - 1)
        fd = self.fd
        out_len = len(prev) + len(fd) - 1
        out = [0.0] * out_len
        for i, pi in enumerate(prev):
            if pi == 0.0:
                continue
            for d, fdd in enumerate(fd):
                if fdd == 0.0:
                    continue
                out[i + d] += pi * fdd
        self._conv_cache[m] = out
        return out

    def S_cdf(self, m, x):
        """P(S_m <= x)."""
        if x < 0:
            return 0.0
        pmf = self._conv_power(m)
        return sum(pmf[: min(x, len(pmf) - 1) + 1])

    def K(self, t, s):
        """Eq. context: mean # customers in (t,t+s). Numeric integral of
        lam(u) du; exact for constant lam, else fine trapezoid."""
        if s <= 0:
            return 0.0
        # constant-rate fast path
        try:
            c = self.lam(t)
            if all(self.lam(t + frac * s) == c for frac in (0.0, 0.5, 1.0)):
                return c * s
        except Exception:
            pass
        n_steps = 200
        h = s / n_steps
        total = 0.5 * (self.lam(t) + self.lam(t + s))
        for i in range(1, n_steps):
            total += self.lam(t + i * h)
        return total * h

    def P_n_j(self, n, j):
        """
        P^n_j: probability the jth unit of stock is demanded by the nth
        customer after t, i.e. P(S_{n-1} <= j-1) - P(S_n <= j-1).
        (Section 3, "these quantities may be easily recovered from f^m_ixd".)
        """
        return self.S_cdf(n - 1, j - 1) - self.S_cdf(n, j - 1)

    def B(self, n, t, s):
        """
        Eq. (8). B(n,t,s) = 1 - integral_t^{t+s} q(n,t,sigma)dsigma
                           = P(fewer than n customers arrive in (t,t+s))
                           = sum_{m=0}^{n-1} K(t,t+s)^m/m! * exp(-K(t,t+s))
        """
        Kts = self.K(t, s)
        pmf = poisson_pmf_upto(Kts, n - 1) if n >= 1 else []
        return sum(pmf)

    def A(self, n, t, s):
        """
        Eq. (7). A(n,t,s) = integral_t^{t+s} (sigma-t) q(n,t,sigma) dsigma.

        CONSTANT-RATE FAST PATH (exact, not an approximation): for
        constant lam, q(n,t,t+sigma) is the Erlang(n,lam) density in
        sigma, and sigma*Erlang(n,lam)_pdf(sigma) = (n/lam)*Erlang(n+1,
        lam)_pdf(sigma) (standard Gamma shape-recursion). Integrating:
            integral_0^s sigma*q(n,t,t+sigma) dsigma
                = (n/lam) * P(Erlang(n+1,lam) <= s)
                = (n/lam) * P(Poisson(lam*s) >= n+1)
                = (n/lam) * (1 - poisson_cdf(n, lam*s))
        (an Erlang(n+1,lam) clock beats s iff at least n+1 Poisson
        arrivals have occurred by time s). Falls back to direct
        numerical integration for genuinely non-stationary lam(t).
        """
        if s <= 0:
            return 0.0
        try:
            c = self.lam(t)
            if c > 0 and all(self.lam(t + frac * s) == c for frac in (0.0, 0.5, 1.0)):
                return (n / c) * (1.0 - sum(poisson_pmf_upto(c * s, n)))
        except Exception:
            pass
        n_steps = 200
        h = s / n_steps

        def q(sigma):
            Kt = self.K(t, sigma)
            lam_t = self.lam(t + sigma)
            if n == 1:
                return lam_t * math.exp(-Kt)
            return lam_t * (Kt ** (n - 1)) / math.factorial(n - 1) * math.exp(-Kt)

        total = 0.5 * (0.0 * q(0.0) + s * q(s))
        for i in range(1, n_steps):
            sigma = i * h
            total += sigma * q(sigma)
        return total * h

    def interval_demand_pmf(self, t, s, dmax=None):
        """
        Compound-Poisson pmf of D(t,t+s) = total demand in (t,t+s):
            P(D=m) = sum_n Poisson_pmf(n; K(t,s)) * P(S_n = m)
        dmax defaults to the Eq.(12)-style mean+3sd cutoff.
        """
        Kts = self.K(t, s)
        # mean/var of a single customer's demand (for the Eq.12 cutoff)
        mu_d = sum(d * p for d, p in enumerate(self.fd))
        var_d = sum((d ** 2) * p for d, p in enumerate(self.fd)) - mu_d ** 2
        mean_D = mu_d * Kts
        var_D = (mu_d ** 2 + var_d) * Kts if Kts > 0 else 0.0
        if dmax is None:
            dmax = int(mean_D + 3 * math.sqrt(max(var_D, 1e-9))) + 5
        nmax = int(Kts + 8 * math.sqrt(max(Kts, 1e-9))) + 10
        pmf_n = poisson_pmf_upto(Kts, nmax)
        out = [0.0] * (dmax + 1)
        for n, pn in enumerate(pmf_n):
            if pn < 1e-14:
                continue
            Sn = self._conv_power(n)
            for m in range(0, min(dmax, len(Sn) - 1) + 1):
                out[m] += pn * Sn[m]
        return out  # out[dmax] absorbs no tail correction; fine for Lix use below


# ═══════════════════════════════════════════════════════════════════════
#  v_ix{IL,t,s;hold}  (Eq. 9)   and   v_ix{IL,t,s;lost}  (Eq. 11)
# ═══════════════════════════════════════════════════════════════════════
def v_hold(dm: DemandModel, IL, t, s, h_cost):
    """
    Eq. (9). Expected holding cost at this location over (t,t+s) given
    inventory level IL at t (IL<=0 contributes 0, matching the paper's
    ILix,t,s;hold definition summed over j=1..ILix).
    """
    if IL <= 0:
        return 0.0
    total = 0.0
    for j in range(1, IL + 1):
        for n in range(1, j + 1):
            Pnj = dm.P_n_j(n, j)
            if Pnj < 1e-15:
                continue
            total += h_cost * (dm.A(n, t, s) + s * dm.B(n, t, s)) * Pnj
    return total


def v_lost(dm: DemandModel, IL, t, s, L_cost, dmax=None):
    """
    Eq. (11) (the simplified closed form, equivalent to Eq.10):
        v{IL,t,s;lost} = sum_{j=IL+1}^{Mix} Lix * P(D(t,s) >= j)
    where IL+ = max(IL,0) (a currently-backordered/negative IL never
    happens in the lost-sales model, but we clip defensively).
    """
    IL_pos = max(IL, 0)
    pmf = dm.interval_demand_pmf(t, s, dmax=dmax)
    Mix = len(pmf) - 1
    if IL_pos >= Mix:
        return 0.0
    # P(D>=j) = 1 - P(D<=j-1); build once via reverse cumulative sum
    ccdf = [0.0] * (Mix + 2)
    running = 0.0
    for m in range(Mix, -1, -1):
        running += pmf[m]
        ccdf[m] = running  # ccdf[m] = P(D>=m)
    total = 0.0
    for j in range(IL_pos + 1, Mix + 1):
        total += L_cost * ccdf[j]
    return total


def v_total(dm: DemandModel, IL, t, s, h_cost, L_cost, dmax=None):
    """v_ix{ILix,t,s} = v{hold} + v{lost} (Eq.3 specialised to X=1, lost sales)."""
    return v_hold(dm, IL, t, s, h_cost) + v_lost(dm, IL, t, s, L_cost, dmax=dmax)


# ═══════════════════════════════════════════════════════════════════════
#  The transshipment index D(uji | di, ILi, ILj, t)  (Eq. 19) and the
#  no-transshipment index D(0 | di, ILi, t)  (Eq. 20), and the decision
#  rule of Eq. (21). Single item type, lost-sales model.
# ═══════════════════════════════════════════════════════════════════════
def IL_tilde_i(IL_i, u_ji, d_i):
    """
    ILtilde_ix(u_jix): inventory at i after demand d_i and a
    transshipment of u_ji units in. Lost-sales model:
        ILtilde = (IL_i + u_ji - d_i)+
    """
    return max(IL_i + u_ji - d_i, 0)


def index_with_transshipment(dm_i, dm_j, Rf_ji, Ru_ji, u_ji, d_i, IL_i, IL_j,
                              t, tau_i, tau_j, h_i, h_j, L_i, dmax=None):
    """
    Eq. (19), single item type, lost-sales model, ASSUMING j and i share
    the horizon-tail term v_j{t+tau_j(t), t+H} cancelling against the
    no-transshipment baseline (both indices are compared as differences
    from a common "no transshipment ever again" continuation, so the
    identical tail terms cancel in the Eq.(21) minimisation and we omit
    them here -- this is exactly what the paper's own index computation
    exploits: only the terms that DEPEND on the current decision matter).

        D(uji|di,ILi,ILj,t) =
            Rf_ji + Ru_ji*uji + Li*(ILi+ - di + uji)-        [Eq: lost-sales
                                                                term, uji
                                                                units reduce
                                                                the shortfall]
            + v_i{ILtilde_i(uji), t, tau_i(t)}
            + v_j{ILj - uji, t, tau_j(t)}
            - v_i{ILi, t, tau_i(t)}
            - v_j{ILj, t, tau_j(t)}

    Returns the index value (lower is better; compare against
    index_no_transshipment below per Eq.21).
    """
    IL_tilde = IL_tilde_i(IL_i, u_ji, d_i)
    residual_shortfall = max(IL_i - d_i + u_ji, 0)  # note: (IL_i+ - d_i+u_ji)- clipped >=0
    # the "(ILix+ - dix + ujix)-" term of Eq.(19)/(15): one-off lost-sale
    # cost for demand that STILL cannot be met after the transshipment.
    unmet = max(d_i - IL_i - u_ji, 0)
    lost_sale_term = L_i * unmet

    idx = (Rf_ji + Ru_ji * u_ji + lost_sale_term
           + v_total(dm_i, IL_tilde, t, tau_i, h_i, L_i, dmax=dmax)
           + v_total(dm_j, IL_j - u_ji, t, tau_j, h_j, 0.0, dmax=dmax)  # sender: no shortage term
           - v_total(dm_i, IL_i, t, tau_i, h_i, L_i, dmax=dmax)
           - v_total(dm_j, IL_j, t, tau_j, h_j, 0.0, dmax=dmax))
    return idx


def index_no_transshipment(dm_i, d_i, IL_i, t, tau_i, h_i, L_i, dmax=None):
    """
    Eq. (20), single item type, lost-sales model:
        D(0|di,ILi,t) = Li*(ILi+ - di)-
                        + v_i{ILtilde_i(0), t, tau_i(t)}
                        - v_i{ILi, t, tau_i(t)}
    """
    IL_tilde = IL_tilde_i(IL_i, 0, d_i)
    unmet = max(d_i - IL_i, 0)
    lost_sale_term = L_i * unmet
    return (lost_sale_term
            + v_total(dm_i, IL_tilde, t, tau_i, h_i, L_i, dmax=dmax)
            - v_total(dm_i, IL_i, t, tau_i, h_i, L_i, dmax=dmax))


def hybrid_decision(candidates, d_i, IL_i, t, tau_i, h_i, L_i,
                     no_transship_dmax=None):
    """
    Eq. (21): min over (j, uji) of D(uji|...) vs D(0|...).

    candidates : list of dicts, one per feasible sender j, each with keys
        'dm_j','Rf_ji','Ru_ji','IL_j','tau_j','h_j','u_range' (iterable of
        feasible uji values per the constraint (22): 0<=uji<=min(ILj,
        Six-ILi+di))
        dm_i (this location's DemandModel) is passed once, separately.

    Returns (decision, detail) where decision is None (no transshipment)
    or (j_index_in_candidates, u_star), and detail holds the winning index
    value for inspection.
    """
    dm_i = candidates[0]['dm_i'] if candidates else None
    # dm_i must be supplied identically in every candidate dict; take it
    # from the first if present, else require explicit arg (kept simple
    # here since our smoke test always supplies it).
    best_val = index_no_transshipment(candidates[0]['dm_i'], d_i, IL_i, t,
                                       tau_i, h_i, L_i, dmax=no_transship_dmax)
    best_choice = None
    for cidx, c in enumerate(candidates):
        for u in c['u_range']:
            val = index_with_transshipment(
                c['dm_i'], c['dm_j'], c['Rf_ji'], c['Ru_ji'], u, d_i, IL_i,
                c['IL_j'], t, tau_i, c['tau_j'], h_i, c['h_j'], L_i)
            if val < best_val:
                best_val, best_choice = val, (cidx, u)
    return best_choice, best_val


# ═══════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 78)
    print("SMOKE TEST 1: DemandModel building blocks run and look sane")
    # geometric demand size, mean 1.25 (matches the paper's own numerical
    # study, Section 5: fixd = 0.8*(1-0.8)^{d-1})
    fd = [0.0] + [0.8 * (0.2 ** (d - 1)) for d in range(1, 30)]
    s_ = sum(fd)
    fd = [p / s_ for p in fd]  # renormalise the truncated tail
    dm = DemandModel(lam=3.0, fd=fd)
    print(f" mean single-customer demand = {sum(d*p for d,p in enumerate(fd)):.3f} "
          f"(paper reports 1.25)")
    print(f" K(t=0,s=2) = {dm.K(0,2):.3f}  (expect lam*s = 6.0)")
    print(f" B(n=1,t=0,s=2) = {dm.B(1,0,2):.4f}  (expect exp(-K)= "
          f"{math.exp(-6.0):.4f})")
    pmf = dm.interval_demand_pmf(0, 2)
    print(f" interval demand pmf sums to {sum(pmf):.4f} (expect close to 1)")

    print("\n cross-check: closed-form A(n,t,s) vs direct numerical integration")
    def A_numerical(dm, n, t, s, n_steps=2000):
        h = s / n_steps
        def q(sigma):
            Kt = dm.K(t, sigma)
            lam_t = dm.lam(t + sigma)
            if n == 1:
                return lam_t * math.exp(-Kt)
            return lam_t * (Kt ** (n - 1)) / math.factorial(n - 1) * math.exp(-Kt)
        total = 0.5 * (0.0 * q(0.0) + s * q(s))
        for i in range(1, n_steps):
            sigma = i * h
            total += sigma * q(sigma)
        return total * h
    for n_ in (1, 2, 5):
        closed = dm.A(n_, 0.0, 2.0)
        numeric = A_numerical(dm, n_, 0.0, 2.0)
        print(f"  n={n_}: closed-form={closed:.6f}  numerical={numeric:.6f}  "
              f"match={abs(closed-numeric)<1e-4}")

    print("\n" + "=" * 78)
    print("SMOKE TEST 2: v_hold, v_lost run and are monotone in IL")
    h_cost, L_cost = 1.0, 20.0
    vh = [v_hold(dm, IL, 0, 2, h_cost) for IL in range(0, 8)]
    vl = [v_lost(dm, IL, 0, 2, L_cost) for IL in range(0, 8)]
    print(" v_hold(IL=0..7):", [round(x, 3) for x in vh])
    print(" v_lost(IL=0..7):", [round(x, 3) for x in vl])
    print(" v_hold nondecreasing in IL?", all(vh[i+1] >= vh[i]-1e-9 for i in range(len(vh)-1)))
    print(" v_lost nonincreasing in IL?", all(vl[i+1] <= vl[i]+1e-9 for i in range(len(vl)-1)))

    print("\n" + "=" * 78)
    print("SMOKE TEST 3: Theorem 1(a) check -- index D(uji|.) must be")
    print("non-increasing in the SENDER's stock level ILj, all else fixed.")
    dm_j = DemandModel(lam=3.0, fd=fd)
    Rf_ji, Ru_ji = 5.0, 1.0
    d_i, IL_i, t, tau_i, tau_j = 2, 1, 0.0, 3.0, 3.0
    u_ji = 1
    vals = []
    for IL_j in range(0, 10):
        idx = index_with_transshipment(dm, dm_j, Rf_ji, Ru_ji, u_ji, d_i,
                                        IL_i, IL_j, t, tau_i, tau_j,
                                        h_cost, h_cost, L_cost)
        vals.append(idx)
    print(" D(uji=1|.) for ILj=0..9:", [round(v, 3) for v in vals])
    nonincreasing = all(vals[i + 1] <= vals[i] + 1e-6 for i in range(len(vals) - 1))
    print(" Theorem 1(a) (non-increasing in ILj) holds?", nonincreasing)

    print("\n" + "=" * 78)
    print("SMOKE TEST 4: full decision rule (Eq.21) runs end-to-end")
    candidates = [{
        'dm_i': dm, 'dm_j': dm_j, 'Rf_ji': Rf_ji, 'Ru_ji': Ru_ji,
        'IL_j': 6, 'tau_j': tau_j, 'h_j': h_cost,
        'u_range': range(0, min(6, 8) + 1),
    }]
    choice, val = hybrid_decision(candidates, d_i=3, IL_i=0, t=0.0,
                                   tau_i=tau_i, h_i=h_cost, L_i=L_cost)
    print(f" decision: {choice}  (None=no transshipment; else (sender_idx,u*))"
          f"   index value = {val:.4f}")
    print(" (no crash, finite index returned -- pipeline is runnable)")

    print("\n" + "=" * 78)
    print("SMOKE TEST 5: lightweight NP vs CP vs H comparison (Table 2 style)")
    print(" Not a numeric replication of Table 2 (that needs the true 3-D")
    print(" optimal DP as the gap denominator, which is future work -- see")
    print(" README). This checks the QUALITATIVE ordering the paper reports")
    print(" in all its numerical studies: C*(H) < C*(CP) < C*(NP), on a")
    print(" small 3-location, single-item, weekly-review simulation with")
    print(" common random numbers (paired comparison, low noise).")

    import random

    def run_comparison(n_loc=3, lam_loc=2.0, h_loc=0.3, L_loc=20.0, S_loc=14,
                        Rf=5.0, Ru=0.5, T_review=7.0, n_cycles=12, R=15,
                        seed=0):
        rng = random.Random(seed)
        fd_unit = [0.0, 1.0]  # single-unit demand (X=1, d=1 always)
        dms = [DemandModel(lam=lam_loc, fd=fd_unit) for _ in range(n_loc)]
        horizon = n_cycles * T_review
        totals = {'NP': [], 'CP': [], 'H': []}

        for rep in range(R):
            # ---- generate ONE shared event stream (common random numbers) ----
            events = []  # (time, 'arrival', loc) or (time, 'replen', None)
            for i in range(n_loc):
                t = 0.0
                while True:
                    t += rng.expovariate(lam_loc)
                    if t >= horizon:
                        break
                    events.append((t, 'arrival', i))
            for c in range(n_cycles):
                events.append((c * T_review, 'replen', None))
            events.sort(key=lambda e: (e[0], e[1] != 'replen'))  # replen first at ties

            for policy in ('NP', 'CP', 'H'):
                IL = [S_loc] * n_loc
                cost = 0.0
                last_t = 0.0
                v_cache = {}  # (loc, IL_val, tau_rounded) -> v_total, memoised

                def v_cached(loc, IL_val, t_now, tau):
                    key = (loc, IL_val, round(tau, 2))
                    if key not in v_cache:
                        v_cache[key] = v_total(dms[loc], IL_val, 0.0, tau,
                                                h_loc, L_loc)
                    return v_cache[key]

                for (t, kind, loc) in events:
                    cost += sum(h_loc * max(IL[i], 0) for i in range(n_loc)) * (t - last_t)
                    last_t = t
                    if kind == 'replen':
                        IL = [S_loc] * n_loc
                        continue
                    # arrival at `loc`, single-unit demand
                    if IL[loc] >= 1:
                        IL[loc] -= 1
                        continue
                    # stockout at loc -> policy-specific decision
                    tau_loc = T_review - (t % T_review)
                    if policy == 'NP':
                        cost += L_loc
                    elif policy == 'CP':
                        senders = [j for j in range(n_loc) if j != loc and IL[j] >= 1]
                        if senders:
                            j = max(senders, key=lambda j: IL[j])  # any tie-break
                            IL[j] -= 1
                            cost += Rf + Ru * 1
                        else:
                            cost += L_loc
                    else:  # 'H'
                        senders = [j for j in range(n_loc) if j != loc and IL[j] >= 1]
                        if not senders:
                            cost += L_loc
                            continue
                        # No-transshipment baseline index (Eq.20 specialised to
                        # d=1,IL_i=0,u=0): unmet=1 so lost_sale_term=L_loc, and
                        # v_i{ILtilde(0)}-v_i{IL_i} = 0 (both states are IL=0).
                        best_val = L_loc
                        best_j = None
                        for j in senders:
                            tau_j = T_review - (t % T_review)
                            # Transshipment index (Eq.19 specialised to d=1,u=1,
                            # IL_i=0): unmet=max(1-0-1,0)=0 so lost_sale_term=0,
                            # and v_i{ILtilde(1)=0}-v_i{IL_i=0}=0 (cancels, both
                            # states are IL=0 -- the unit is consumed immediately).
                            idx = (Rf + Ru * 1
                                   + v_cached(j, IL[j] - 1, t, tau_j)
                                   - v_cached(j, IL[j], t, tau_j))
                            if idx < best_val:
                                best_val, best_j = idx, j
                        if best_j is None:
                            cost += L_loc
                        else:
                            IL[best_j] -= 1
                            cost += Rf + Ru * 1
                # final holding accrual to horizon end
                cost += sum(h_loc * max(IL[i], 0) for i in range(n_loc)) * (horizon - last_t)
                totals[policy].append(cost)
        return totals

    totals = run_comparison()
    import statistics as _stats
    means = {p: _stats.mean(v) for p, v in totals.items()}
    # paired standard errors (CRN -> compare within-replication differences)
    diffs_np_h = [a - b for a, b in zip(totals['NP'], totals['H'])]
    diffs_cp_h = [a - b for a, b in zip(totals['CP'], totals['H'])]
    def paired_ci(diffs):
        m = _stats.mean(diffs)
        se = _stats.stdev(diffs) / math.sqrt(len(diffs))
        return m, 1.96 * se
    m1, ci1 = paired_ci(diffs_np_h)
    m2, ci2 = paired_ci(diffs_cp_h)

    print(f"\n mean total cost over R=15 replications, 12 weekly cycles, 3 locations:")
    print(f"  NP: {means['NP']:.2f}   CP: {means['CP']:.2f}   H: {means['H']:.2f}")
    print(f" paired (CRN) NP-H = {m1:+.2f} +- {ci1:.2f}  "
          f"(positive & CI excludes 0 => H significantly cheaper than NP)")
    print(f" paired (CRN) CP-H = {m2:+.2f} +- {ci2:.2f}  "
          f"(positive & CI excludes 0 => H significantly cheaper than CP)")
    ordering_ok = means['H'] < means['CP'] < means['NP']
    print(f"\n Qualitative ordering C*(H) < C*(CP) < C*(NP) holds: {ordering_ok}")
    if means['NP'] > 0:
        print(f" H's improvement over NP: {(means['NP']-means['H'])/means['NP']:.1%}  "
              f"(paper's Table 2 range for policy H vs NP: up to ~250% in some cases,"
              f" typically tens of percent -- order of magnitude, not a direct match,"
              f" since parameters differ)")