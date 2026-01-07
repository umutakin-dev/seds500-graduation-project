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
- [ ] Multinomial forward process (add categorical noise)
- [ ] Multinomial reverse process (denoise categories)
- [ ] Hybrid model combining Gaussian + Multinomial
- [ ] Test on dataset with mixed feature types (e.g., Adult)

## Status
- [ ] Not started
