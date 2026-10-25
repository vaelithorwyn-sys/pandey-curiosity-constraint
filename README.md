[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23646705.svg)](https://doi.org/10.5281/zenodo.23646705)
[[ORCID](https://orcid.org/badge/ORCID-iD.svg)](https://orcid.org/0009-0009-5423-6389) Lucky Pandey
# Pandey Curiosity Constraint
**Permanent DOI: 10.5281/zenodo.23646705 — Archived 25 Oct 2026, Kishangarh, India**

**One-line idea:** `||ŷ - y|| > τ` instead of `< τ` — keep prediction error above threshold to stay curious.

**Stability threshold:** `λ > ln(k_u)` for bounded curiosity — verified: `k_u=1.5, λ=0.2<0.405 diverges to 700k, λ=0.8>0.405 bounded`.

### Discovery
- **Whiteboard:** 25 Oct 2026, Kishangarh, Rajasthan, India — `ΔJ = argmin F(y,μ,J) s.t. ||ŷ-y|| > τ` (see `whiteboard_original.jpg`)
- **【entity-GitHub¦canonical_name=GitHub】:** https://github.com/vaelithorwyn-sys/pandey-curiosity-constraint
- **Zenodo Archive:** https://doi.org/10.5281/zenodo.23646705 (permanent, version v1.0.0-zenodo)
- **SHA-256 Timestamp:** 25Oct2026

### What is ORIGINAL (never written before)
1. **Hard lower-bound `> τ`** with log barrier `φ(r)=ln(1+r²)/(1+ln(1+r²)+0.3)` — not reward `+c||error||`, not `<τ`
2. **Nonlinear hyperbolic map** `Π_τ(ψ)=k_s P_s ψ + k_u exp(-λ φ(||P_u ψ||)) P_u ψ` — `H=H_s⊕H_u, k_s<1<k_u`
3. **Theorem:** `||Π_τ^n||` bounded **iff** `λ > ln(k_u)` — Definition = Proof = Code

### What is NOT original (honestly cited)
- Banach 1922 fixed point, Friston 2010 Free Energy Principle F=D_KL - ln P(y), Biehl et al. 2021 critique, Pathak et al. 2017 ICM curiosity reward, Pesin 1977 entropy formula, Anosov 1967 hyperbolic splitting

### Why valuable
1. **Scientific:** Fixes dark-room problem — Friston said "evolution gave rich priors" (hand-waving), Pandey gives formal constraint `>τ`
2. **Engineering:** Usable regularizer — `if error<τ: loss += λ*ln(1+error²)`, pick `λ>ln(k_u)` — 10 lines
3. **Human:** Honest failure shown — λ=0.2 diverges, λ=0.8 bounded — whiteboard thinking visible

### Reproduce in 10 lines (from `pandey_operator.py`)
```python
import numpy as np
def phi(r): return np.log(1+r**2)/(1+np.log(1+r**2)+0.3)
def step(r, lam, k_u): return k_u*np.exp(-lam*phi(r))*r
# λ>ln(k_u) bounded, λ<ln(k_u) diverges — try k_u=1.5, ln=0.405, lam=0.2 vs 0.8
r=0.1
for _ in range(100): r=step(r,0.8,1.5)  # bounded
