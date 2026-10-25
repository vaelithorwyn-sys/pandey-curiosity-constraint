"""
Reproduce Fig2 and Fig3 from Pandey paper - 10 lines core
"""
import numpy as np
import matplotlib.pyplot as plt

def phi(r): return np.log(1+r**2)/(1+np.log(1+r**2)+0.3)
def step(r, lam, k_u): return k_u*np.exp(-lam*phi(r))*r

# Fig2b: k_s=0.5, k_u=2.5, threshold 0.916
k_u=2.5
for lam in [0.8, np.log(k_u), 1.2]:
    r=0.5; traj=[r]
    for _ in range(70):
        r=step(r,lam,k_u)
        traj.append(r)
    plt.plot(traj, label=f'lambda={lam:.3f}')
plt.axhline(y=0,color='gray'); plt.legend(); plt.title(f'Pandey threshold lambda>ln(k_u)={np.log(k_u):.3f}')
plt.savefig('figures/fig2b_reproduced.png', dpi=300)
print("Saved figures/fig2b_reproduced.png")

# Fig3 dark-room
k_s=0.8
std=0.8
pandey=1.0
std_traj=[]; pandey_traj=[]
for _ in range(100):
    std=k_s*std
    pandey = 2.5*pandey if pandey<0.5 else step(pandey,1.2,2.5)
    std_traj.append(std); pandey_traj.append(pandey)
plt.figure()
plt.plot(std_traj,label='Standard FEP <tau -> dark room')
plt.plot(pandey_traj,label='Pandey >tau -> curious')
plt.axhline(0.5,color='r',linestyle='--',label='tau=0.5')
plt.legend(); plt.savefig('figures/fig3_reproduced.png', dpi=300)
