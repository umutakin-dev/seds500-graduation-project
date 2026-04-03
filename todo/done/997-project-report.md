# Project Report

**Source:** Faculty requirement
**Priority:** High

## Task

Write the graduation project report following the faculty's required structure.

## Thesis Framing (Updated 2026-01-11)

**Title:** Privacy-Preserving Synthetic Tabular Data Generation using Diffusion Models

**Research Question:** When generating synthetic tabular data for privacy/anonymization, do diffusion models produce more realistic data than traditional methods?

**Key Finding:** Diffusion-generated synthetic data preserves statistical properties (models trained on it work), while SMOGN-generated data can be catastrophically unrealistic (models fail).

## Report Structure

### 1. Introduction
- Problem: Organizations can't share sensitive data
- Solution: Generate realistic synthetic data
- Research question and contributions

### 2. Background / Literature Review
- Diffusion models (DDPM, TabDDPM)
- Traditional tabular augmentation (SMOTE, SMOGN)
- Privacy-preserving data synthesis

### 3. Methodology
- HybridDiffusion (Gaussian + Multinomial)
- Quality metric: ML utility (train on synthetic, test on real)
- Experimental setup

### 4. Experiments
- Exp 011: Manufacturing data (SMOGN catastrophic failure)
- Exp 012: Ozel simple (tie)
- Exp 013: Ozel rich (SMOGN catastrophic failure)

### 5. Results
- Diffusion maintains data quality
- SMOGN fails on complex feature spaces
- Pattern: feature complexity matters

### 6. Discussion
- Research journey (transparency about hypothesis evolution)
- Why SMOGN fails (interpolation in high dimensions)
- Why diffusion works (learns distribution)

### 7. Conclusion
- Diffusion is "safe default" for synthetic data
- Practical implications for data sharing
- Future work (differential privacy, quality metrics)

## Key Evidence Table

| Experiment | Features | Original R² | Diffusion R² | SMOGN R² |
|------------|----------|-------------|--------------|----------|
| 011 (Manufacturing) | 2 | 0.75 | 0.75 | **negative** |
| 012 (Ozel simple) | 5 | 0.55 | 0.47-0.55 | 0.39-0.55 |
| 013 (Ozel rich) | 29 | 0.65 | 0.64 | **-0.39** |

## Reference Materials

- `docs/thesis-summary.md` - Main summary document
- `experiments/*/RESULTS.md` - Detailed experiment results
- `docs/paper-analysis.md` - Literature review notes

## Status
- [ ] Waiting for faculty documentation on required format
