# Multinomial Diffusion for Categorical Features

**Source:** Implementation roadmap

## Task
Implement multinomial diffusion to handle categorical features (TabDDPM approach).

## Notes
- Gaussian diffusion only works for continuous numerical data
- Categorical features need multinomial diffusion
- Forward process: gradually corrupt one-hot vectors toward uniform distribution
- Reverse process: learn to predict original category probabilities
- Reference: TabDDPM paper Section 3.2

## Key Components
- [x] Multinomial forward process (add categorical noise)
- [x] Multinomial reverse process (denoise categories)
- [x] Hybrid model combining Gaussian + Multinomial
- [x] Test on dataset with mixed feature types (Adult)

## Implementation Details
- `MultinomialDiffusion` class in `src/diffusion.py`
- `HybridDiffusion` class combining Gaussian + Multinomial
- `HybridMLPDenoiser` in `src/models.py`

## Status
- [x] Core implementation complete (2026-01-11)
- [x] Validated on Adult dataset (Experiment 003)
