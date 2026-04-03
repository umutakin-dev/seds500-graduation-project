# Technical Explanation for Report

**Source:** Planning session
**Priority:** Medium

## Task

Formalize the technical explanation of diffusion models for the project report.

## Thesis Framing (Updated 2026-01-11)

**Topic:** Privacy-Preserving Synthetic Tabular Data Generation using Diffusion Models

**Research Question:** When generating synthetic tabular data for privacy/anonymization, do diffusion models produce more realistic data than traditional methods (SMOGN)?

**Key Finding:** Diffusion preserves data quality, SMOGN can fail catastrophically on complex feature spaces.

## Content to Cover

### 1. What is a Diffusion Model?
- Forward process (adding noise gradually)
- Reverse process (learning to denoise)
- Training objective (predicting noise)

### 2. Why Diffusion for Tabular Data?
- Tables have mixed types (numeric + categorical)
- Traditional methods (SMOGN) use interpolation
- Diffusion learns the actual data distribution

### 3. HybridDiffusion (TabDDPM approach)
- Gaussian diffusion for numeric features
- Multinomial diffusion for categorical features
- Combined training loss

### 4. Privacy-Preserving Synthetic Data
- Generated samples are new (not copies)
- Statistical properties preserved
- Models trained on synthetic data work on real data

## Reference Materials

- Notebook: `notebooks/01_diffusion_explained.ipynb`
- Concepts doc: `docs/concepts.md`
- Paper analysis: `docs/paper-analysis.md`
- Thesis summary: `docs/thesis-summary.md`

## Status
- [ ] Not started
