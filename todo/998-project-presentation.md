# Project Presentation

**Source:** Faculty requirement
**Priority:** High

## Task

Prepare the graduation project presentation slides.

## Thesis Framing (Updated 2026-01-11)

**Title:** Privacy-Preserving Synthetic Tabular Data Generation using Diffusion Models

## Suggested Slide Structure

### 1. Title Slide
- Title, name, date, institution

### 2. The Problem (1-2 slides)
- Organizations can't share sensitive data (privacy, trade secrets)
- But data sharing is valuable (collaboration, ML, research)
- Need: Realistic synthetic data

### 3. Research Question (1 slide)
- "Do diffusion models produce more realistic synthetic data than traditional methods?"
- Quality metric: Can models trained on synthetic data work on real data?

### 4. What is Diffusion? (2-3 slides)
- Visual: Forward process (data → noise)
- Visual: Reverse process (noise → data)
- HybridDiffusion for tables (numeric + categorical)

### 5. Experiments (2-3 slides)
- Datasets: Manufacturing (18k rows), Ozel (2-3k rows)
- Setup: Train on synthetic, test on real
- Comparison: Diffusion vs SMOGN

### 6. Key Results (2-3 slides)
- **Visual table:** R² comparison across experiments
- **Highlight:** SMOGN catastrophic failure (R² goes negative!)
- **Highlight:** Diffusion maintains performance

### 7. Why Does This Happen? (1-2 slides)
- SMOGN: Interpolation fails in high dimensions
- Diffusion: Learns actual data distribution
- Visual: Feature space illustration

### 8. Research Journey (1 slide)
- Started with "augmentation for accuracy"
- Discovered "synthetic data quality" is the real story
- Transparency about hypothesis evolution

### 9. Practical Implications (1 slide)
- Companies can share synthetic data safely
- Models trained on synthetic work on real
- Diffusion = "safe default"

### 10. Conclusion & Future Work (1 slide)
- Diffusion produces realistic synthetic data
- Future: Differential privacy, quality metrics

## Visual Elements Needed

- [ ] Diffusion process diagram (forward/reverse)
- [ ] R² comparison chart (bar chart or table)
- [ ] Feature space visualization (why interpolation fails)
- [ ] Research journey timeline

## Status
- [ ] Not started
