"""
Pandey Curiosity Constraint
Original discovery: Delta J = argmin F s.t. ||y_hat - y|| > tau
Stability threshold: lambda > ln(k_u)

Author: Lucky Pandey, 25 Oct 2026, Kishangarh
Unkillable version: exponential form Pi_tau = k_s P_s + k_u*exp(-lambda*phi(r)) P_u
where phi(r) = ln(1+r^2)/(1+ln(1+r^2)+0.3) -> 1 as r->inf
Theorem: bounded iff lambda > ln(k_u)
"""
import numpy as np

def phi(r, c=0.3):
    """log barrier normalized to [0,1)"""
    return np.log(1 + r**2) / (1 + np.log(1 + r**2) + c)

def pandey_step(psi_s, psi_u, k_s=0.5, k_u=2.5, lam=1.2):
    """
    One step of Pandey nonlinear map
    psi_{n+1} = Pi_tau(psi_n)
    """
    r = np.linalg.norm(psi_u) if hasattr(psi_u, '__len__') else abs(psi_u)
    factor_u = k_u * np.exp(-lam * phi(r))
    new_s = k_s * psi_s
    new_u = factor_u * psi_u
    return new_s, new_u

def is_stable(k_u, lam):
    """Theorem 1: bounded iff lam > ln(k_u)"""
    return lam > np.log(k_u)

# Example: test stability threshold
if __name__ == "__main__":
    k_s, k_u = 0.5, 2.5
    print(f"Threshold ln(k_u) = {np.log(k_u):.4f}")
    for lam in [0.8, np.log(k_u), 1.2]:
        r = 0.5
        for _ in range(70):
            r = k_u * np.exp(-lam * phi(r)) * r
        print(f"lambda={lam:.3f} -> final r={r:.2e} {'STABLE' if is_stable(k_u, lam) else 'DIVERGENT'}")
