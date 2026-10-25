# Pandey Curiosity Constraint

**One-line idea:** `||ŷ - y|| > τ` instead of `< τ` — keep prediction error above threshold to stay curious.

**Stability threshold:** `λ > ln(k_u)` for bounded curiosity.

### Discovery
- Whiteboard: 25 Oct 2026, Kishangarh, India
- Original anchor: ΔJ = argmin F(y,μ,J) s.t. ||ŷ-y|| > τ (photo in whiteboard_original.jpg)
- SHA-256: 25Oct2026 timestamp

### What is original (never written before)
1. Hard lower-bound constraint `> τ` with log barrier (not reward)
2. Nonlinear map `Π_τ(ψ)=k_sP_sψ + k_u exp(-λ φ(||P_u ψ||)) P_u ψ`, φ(r)=ln(1+r²)/(1+ln(1+r²))
3. Theorem: bounded iff `λ > ln(k_u)`

### What is NOT original (honestly cited)
- Banach 1922 contraction, Friston 2010 FEP, Biehl 2021 critique, Pathak 2017 ICM, Pesin 1977 entropy, Anosov 1967 splitting

### Why valuable
1. Scientific: fixes dark-room problem — Friston's "evolution gave rich priors" is hand-waving, `>τ` is formal
2. Engineering: usable regularizer `if error<τ: penalty λ ln(1+error²)`, pick λ>ln(k_u)
3. Human: honest failure cases shown, whiteboard thinking visible

### Reproduce in 10 lines
```python
def phi(r): return np.log(1+r**2)/(1+np.log(1+r**2)+0.3)
def step(r, lam, k_u): return k_u*np.exp(-lam*phi(r))*r
```

See `pandey_operator.py` and `experiment.py`

### Paper
`paper/Pandey_Curiosity_Constraint_2026.pdf` — 5 pages, unkillable version with corrected exponential form

### Cite as
Lucky Pandey (2026). The Pandey Curiosity Constraint: Maintaining Prediction Error Above Threshold with Hyperbolic Stability λ>ln(k_u). Independent Researcher, Bengaluru.

### License
MIT — use this brick, cite it as "Pandey constraint" when you need to avoid collapse.
