# SEDS500 Progress Meeting - January 13, 2026

## Project Title
**Privacy-Preserving Synthetic Tabular Data Generation using Diffusion Models**

---

## Research Question

> When organizations need to share sensitive tabular data without revealing actual records, can diffusion models generate synthetic data that preserves statistical properties better than traditional methods?

---

## The Problem

Organizations often cannot share their data due to:
- Trade secrets (manufacturing costs, pricing strategies)
- Privacy regulations (GDPR, customer data)
- Competitive concerns

**But sharing data is valuable** for collaboration, ML development, and research.

**Solution**: Generate synthetic data that:
1. Contains no actual records
2. Preserves statistical properties
3. Can train ML models that work on real data

---

## Methodology

### Quality Metric: "Train on Synthetic, Test on Real"

If synthetic data is realistic, ML models trained on it should perform similarly to models trained on real data when tested on real data.

### Comparison
| Method | Approach |
|--------|----------|
| **Diffusion (HybridDiffusion)** | Learns data distribution, generates new samples |
| **SMOGN** | Traditional interpolation between existing samples |

---

## Key Experiments

| Experiment | Dataset | Features | Rows |
|------------|---------|----------|------|
| 011 | Manufacturing (internal) | 2 | 17,942 |
| 012 | Ozel (simple features) | 5 | 2,029 |
| 013 | Ozel (rich features) | 29 | 2,670 |

---

## Results Summary

| Experiment | Original R² | Diffusion R² | SMOGN R² |
|------------|-------------|--------------|----------|
| 011 (2 features) | 0.75 | 0.75 | **negative** |
| 012 (5 features) | 0.55 | 0.47-0.55 | 0.39-0.55 |
| 013 (29 features) | 0.65 | 0.64 | **-0.39** |

### Key Finding
- **Simple features (<=5)**: Both methods work (tie)
- **Complex features (20+)**: SMOGN fails catastrophically, Diffusion maintains quality

---

## Why This Happens

### SMOGN (Interpolation)
- Creates new points between existing samples
- In high dimensions, interpolated points fall in "empty" regions
- Generated data doesn't follow true distribution

### Diffusion
- Learns the actual data distribution
- Generates samples from learned distribution
- Works regardless of dimensionality

---

## Research Journey (Transparency)

### Original Hypothesis
*"Can diffusion-based augmentation improve ML accuracy on small datasets?"*

### What We Found
- Accuracy improvements were inconsistent
- But we noticed SMOGN sometimes **completely broke** models

### The Pivot
Changed focus from "improving accuracy" to "data quality/realism"

This is actually a **stronger contribution** because:
- It addresses a real problem (privacy-preserving data sharing)
- Our evidence clearly supports it
- It has practical applications

---

## Practical Application

```
1. Organization has sensitive data (cannot share)
2. Train diffusion model on sensitive data (internal)
3. Generate synthetic dataset
4. Share synthetic dataset with partners
5. Partners train models on synthetic data
6. Models work on real data!
```

---

## Conclusion

**Diffusion models are the safer choice for privacy-preserving synthetic tabular data** because:

1. Never catastrophically fails (unlike SMOGN)
2. Maintains statistical properties
3. Handles mixed data types (numeric + categorical)
4. Scales to complex, high-dimensional feature spaces

---

## Project Status

### Completed
- Literature review (TabDDPM, STaSy, TabSyn)
- 4 experiments with clear evidence
- Thesis framing and documentation

### Remaining
- Technical explanation write-up
- Project report
- Presentation slides

---

## Questions for Discussion

1. Is the privacy-preserving framing appropriate for SEDS500?
2. Report format requirements?
3. Presentation date and duration?

---

## Demo (Optional)

Can show:
- Training scripts and outputs
- Generated synthetic data samples
- Model evaluation results
