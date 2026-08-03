"""fig_cu.pdf: left = explicit pi1<pi2 counterexample (V slopes in cu while
the state is waiting at time zero for every cu shown), right = pi1>pi2 flat."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src=open('validation_Cf0.py').read()
i=src.index("import numpy as np"); j=src.index("dt = T / N")
exec(src[i:j])
ii=lambda x:int(x-IMIN)

# LEFT: lam1=15, lam2=1, h=1, pi1=0.8, pi2=6, T=8, state (6,8)
l1,l2,h,p1,p2,Tt,I2s = 15.,1.,1.,0.8,6.,8.0,6
def st(tau,cu): return stair(tau,cu=cu,lam2=l2,h=h,pi1=p1,pi2=p2)
bL=beta(I2s,Tt,lam1=l1,lam2=l2,h=h,pi1=p1,pi2=p2)
cusL=np.arange(0.6,7.01,0.4); vL=[]; wL=[]; c_star=None
for cu_ in cusL:
    _,V,_,_=solve(cu=cu_,lam1=l1,lam2=l2,h=h,pi1=p1,pi2=p2,T_=Tt,N_=6000)
    vL.append(V[ii(I2s)]); wL.append(I2s<st(Tt,cu_))
    fl=min(st(round(t,3),cu_) for t in np.arange(0.05,Tt+1e-9,0.02))
    if c_star is None and fl>I2s: c_star=cu_
print("left: waiting at t=0 for all cu:", all(wL), " c* (floor>6) =", c_star)
print("left span:", max(vL)-min(vL), " beta:", bL)

# RIGHT: pi1>pi2 flat (2, tau=1), cu 5..12
l1r,l2r,hr,p1r,p2r = 3.,5.,1.,5.5,4.
bR=beta(2,1.0,lam1=l1r,lam2=l2r,h=hr,pi1=p1r,pi2=p2r)
cusR=np.arange(5.0,12.01,0.5); vR=[]
for cu_ in cusR:
    _,V,_,_=solve(cu=cu_,lam1=l1r,lam2=l2r,h=hr,pi1=p1r,pi2=p2r,T_=1.0,N_=5000)
    vR.append(V[ii(2)])
print("right max dev:", max(abs(v-bR) for v in vR))

fig,(axL,axR)=plt.subplots(1,2,figsize=(9.6,3.4))
axL.axhline(bL,color="0.55",lw=1.0,ls=":",label=rf"Eq.(14)$={bL:.1f}$ (no $c_u$)")
axL.plot(cusL,vL,"o-",ms=3.5,lw=1.5,color="#1F618D",label=r"$V^{\rm DP}(6,\tau_0{=}8)$")
if c_star: axL.axvline(c_star,color="#B03A2E",lw=1.0,ls="--",
                       label=rf"floor rises past $6$ at $c_u\approx{c_star:.1f}$")
axL.set_xlabel(r"$c_u$"); axL.set_ylabel(r"$V(6,\tau_0{=}8)$")
axL.set_title(r"$\pi_1<\pi_2$: waiting at $t{=}0$ for every $c_u$ shown,"
              "\n"r"yet $V$ moves with $c_u$ until the sweep cannot reach",fontsize=9)
axL.legend(fontsize=7,loc="lower right"); axL.grid(True,alpha=0.3)
axR.axhline(bR,color="0.55",lw=1.0,ls=":",label=rf"Eq.(14)$={bR:.2f}$ (no $c_u$)")
axR.plot(cusR,vR,"o-",ms=3.5,lw=1.5,color="#1F618D",label=r"$V^{\rm DP}(2,\tau_0{=}1)$")
axR.set_xlabel(r"$c_u$"); axR.set_ylabel(r"$V(2,\tau_0{=}1)$")
axR.set_title(r"$\pi_1>\pi_2$: flat everywhere (boundary never falls)",fontsize=9)
axR.set_ylim(bR-1,bR+1); axR.legend(fontsize=7); axR.grid(True,alpha=0.3)
fig.tight_layout(); fig.savefig("fig_cu.pdf")
print("saved fig_cu.pdf")