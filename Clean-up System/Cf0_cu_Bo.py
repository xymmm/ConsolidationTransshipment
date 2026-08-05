"""
cu_demo_for_bo.py — a concrete cu-dependence example on the note's own model.

Keeps the note's example (lam1=3, h=1, pi1=4, pi2=5.5, T=5) and varies only
lam2. As lam2 falls, a state that is in the waiting region at t=0 becomes
cu-dependent, because the inventory now survives long enough for the falling
boundary to reach it. Shows V(I2, tau0=T) vs cu for several lam2, plus the
DP threshold curves that explain why.

Requires validation_Cf0.py + solver_cf0_2d.py in the same folder (for beta,
stair, solve). Writes cu_demo.pdf.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = open("validation_Cf0.py").read()
i = src.index("import numpy as np"); j = src.index("dt = T / N")
exec(src[i:j])
ii = lambda x: int(x - IMIN)

L1, H, PI1, PI2, T = 3.0, 1.0, 4.0, 5.5, 5.0     # note's example, fixed
LAM2_LIST = [5.0, 3.0, 2.0, 1.0]                 # only lam2 varies
I2_STATE = 8                                     # a waiting state at t=0
CU_GRID = np.arange(4.0, 9.01, 0.25)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.8))
colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(LAM2_LIST)))

for l2, c in zip(LAM2_LIST, colors):
    def st(tau, cu): return stair(tau, cu=cu, lam2=l2, h=H, pi1=PI1, pi2=PI2)
    # left: V(8, T) vs cu
    vals = []
    for cu_ in CU_GRID:
        _, V, _, _ = solve(cu=cu_, lam1=L1, lam2=l2, h=H, pi1=PI1, pi2=PI2,
                           T_=T, N_=800)
        vals.append(V[ii(I2_STATE)])
    span = max(vals) - min(vals)
    axL.plot(CU_GRID, vals, "o-", ms=3, lw=1.5, color=c,
             label=rf"$\lambda_2={l2:g}$  (span {span:.2f})")
    # right: threshold curve at cu=7, to show whether it reaches I2=8
    taus = np.linspace(0.05, T, 200)
    thr = [st(t, 7.0) for t in taus]
    axR.plot(taus, thr, lw=1.6, color=c, label=rf"$\lambda_2={l2:g}$")

axL.axhline(beta(I2_STATE, T, lam1=L1, lam2=5.0, h=H, pi1=PI1, pi2=PI2),
            color="0.6", ls=":", lw=1.0)
axL.set_xlabel(r"$c_u$")
axL.set_ylabel(rf"$V(I_2{{=}}{I2_STATE},\ \tau_0{{=}}{T:g})$")
axL.set_title(r"Value at a waiting state vs $c_u$", fontsize=10)
axL.legend(fontsize=7.5, loc="upper left")
axL.grid(True, alpha=0.3)

axR.axhline(I2_STATE, color="#B03A2E", ls="--", lw=1.0,
            label=rf"$I_2={I2_STATE}$")
axR.set_xlabel(r"$\tau$")
axR.set_ylabel(r"threshold $\bar I_2(\tau)$ at $c_u=7$")
axR.set_title(r"Why: does the boundary reach $I_2{=}8$?", fontsize=10)
axR.set_ylim(0, 20)
axR.legend(fontsize=7.5, loc="upper right")
axR.grid(True, alpha=0.3)

fig.suptitle(r"Note's example ($\lambda_1{=}3, h{=}1, \pi_1{=}4, "
             r"\pi_2{=}5.5, T{=}5$), varying only $\lambda_2$", fontsize=10)
fig.tight_layout()
fig.savefig("cu_demo.pdf")
fig.savefig("cu_demo.png", dpi=130)
print("saved cu_demo.pdf / .png")
for l2 in LAM2_LIST:
    vals = []
    for cu_ in CU_GRID:
        _, V, _, _ = solve(cu=cu_, lam1=L1, lam2=l2, h=H, pi1=PI1, pi2=PI2,
                           T_=T, N_=800)
        vals.append(V[ii(I2_STATE)])
    print(f"  lam2={l2}: V(8,5) span over cu = {max(vals)-min(vals):.4f}")