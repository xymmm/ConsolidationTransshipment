"""Independent DES check of the crease at I2 = 6, 7.

For each I2, compare on identical sample paths:
  DISP0 : dispatch q0 = min(I2, b1) at t = 0, then follow the DP policy
  WAIT0 : do not dispatch at t = 0, then follow the DP policy from the
          first arrival onward
margin = E[WAIT0] - E[DISP0].  Positive means dispatching now is better.
No Delta_t in the cost accounting, no truncation in the paths.
"""
import numpy as np
LAM1,LAM2,H,PI1,PI2,CF,CU,T = 5.,3.,1.,6.,6.,8.,1.,5.
N,IMAX,IMIN,B1M = 2000,40,-70,130
dt=T/N; p1,p2=LAM1*dt,LAM2*dt; p0=1-p1-p2
I2v=np.arange(IMIN,IMAX+1); I2g,b1g=I2v[:,None],np.arange(0,B1M+1)[None,:]
cI=lambda x:np.clip(x,IMIN,IMAX)-IMIN; cB=lambda x:np.clip(x,0,B1M)
sh=(len(I2v),B1M+1)
flow=np.broadcast_to(dt*(H*np.maximum(0,I2g)+PI1*b1g+PI2*np.maximum(0,-I2g)),sh).copy()
V=np.zeros(sh); Q=np.zeros((N+1,IMAX+1,B1M+1),np.int16)
for n in range(1,N+1):
    best=flow+(p0*V[cI(I2g),cB(b1g)]+p1*V[cI(I2g),cB(b1g+1)]+p2*V[cI(I2g-1),cB(b1g)])
    bq=np.zeros(sh,np.int16)
    for q in range(1,IMAX+1):
        feas=(I2g>=q)&(b1g>=q)
        I2a=np.broadcast_to(I2g-q,sh); b1a=np.broadcast_to(b1g-q,sh)
        c=(CF+CU*q+dt*(H*np.maximum(0,I2a)+PI1*b1a+PI2*np.maximum(0,-I2a))
           +p0*V[cI(I2a),cB(b1a)]+p1*V[cI(I2a),cB(b1a+1)]+p2*V[cI(I2a-1),cB(b1a)])
        c=np.where(feas,c,np.inf); u=c<best-1e-12
        best=np.where(u,c,best); bq=np.where(u,q,bq)
    V=best
    Q[n]=bq[cI(np.arange(0,IMAX+1))[:,None],np.arange(0,B1M+1)[None,:]]

# check: at a fixed state, "dispatch" is an up-set in n, so waiting cannot be
# triggered by the passage of time alone
viol=0
for I2 in range(1,IMAX+1):
    for b1 in range(1,41):
        col=Q[1:,I2,b1]>0
        if col.any() and not col[int(np.argmax(col)):].all(): viol+=1
print(f"up-set check over n (dispatch cannot be triggered by time alone): "
      f"{viol} violations")

R,KMAX=600_000,160
rng=np.random.default_rng(7)
lam=LAM1+LAM2
times=np.cumsum(rng.exponential(1/lam,size=(R,KMAX)),axis=1)
is1=rng.random((R,KMAX))<LAM1/lam
assert (times[:,-1]>T).all()

def sim(I0,b0,mode):
    I2=np.full(R,I0,np.int64); b1=np.full(R,b0,np.int64)
    cost=np.zeros(R); t=np.zeros(R)
    if mode=='disp0':
        q0=min(I0,b0); cost+=CF+CU*q0; I2-=q0; b1-=q0
    for k in range(KMAX):
        tk=times[:,k]
        seg=np.maximum(np.minimum(tk,T)-t,0.0)
        cost+=seg*(H*np.maximum(I2,0)+PI1*b1+PI2*np.maximum(-I2,0))
        t=np.minimum(tk,T)
        alive=tk<T
        if not alive.any(): break
        b1+=(alive&is1[:,k]).astype(np.int64)
        I2-=(alive&~is1[:,k]).astype(np.int64)
        n=np.clip(np.round((T-t)/dt).astype(np.int64),1,N)
        q=Q[n,np.clip(I2,0,IMAX),np.clip(b1,0,B1M)].astype(np.int64)
        q=np.where((I2>0)&(b1>0)&alive,np.minimum(q,np.minimum(I2,b1)),0)
        act=q>0
        cost+=np.where(act,CF+CU*q,0.0); I2-=q; b1-=q
    return cost

print(f"\n{R:,} paths, tau = 5, b1 = 3, terminal cost 0")
print(f"{'I2':>4} | {'DP b1bar':>9} | {'sim WAIT0 - DISP0':>22}")
for I2 in [4,5,6,7,8,9]:
    w=sim(I2,3,'wait0'); d=sim(I2,3,'disp0')
    diff=w-d
    ci=1.96*diff.std(ddof=1)/np.sqrt(R)
    col=Q[1:,I2,1:41]  # b1bar at tau=5 -> n=N
    row=Q[N,I2,1:]>0
    bb=int(np.argmax(row))+1 if row.any() else -1
    print(f"{I2:>4} | {bb:>9} | {diff.mean():>13.4f} ± {ci:.4f} | ")
