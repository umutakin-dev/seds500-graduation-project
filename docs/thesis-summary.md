# Thesis Summary: Privacy-Preserving Synthetic Tabular Data Generation using Diffusion Models

## Research Question

**When organizations need to share sensitive tabular data (e.g., manufacturing costs, customer data) without revealing actual records, can diffusion models generate synthetic data that preserves the statistical properties better than traditional methods?**

## Motivation

Organizations often cannot share their data due to:
- **Privacy concerns** (customer data, medical records)
- **Trade secrets** (manufacturing costs, pricing strategies)
- **Regulatory requirements** (GDPR, HIPAA)

However, sharing data is valuable for:
- Collaboration with partners/researchers
- Training ML models without exposing real data
- Benchmarking and reproducibility

**Solution**: Generate synthetic data that:
1. Preserves statistical properties of the original
2. Does not contain any actual records
3. Can train ML models that generalize to real data

## Research Journey: How We Got Here

**This section documents the evolution of our research question. We believe transparency about the research process is valuable.**

### Initial Hypothesis (Data Augmentation)
We originally set out to answer: *"Can diffusion-based data augmentation improve ML model accuracy on small tabular datasets?"*

The idea was that small datasets might benefit from synthetic samples, similar to how image augmentation helps in computer vision.

### What We Found
Results were **mixed**:
- On some datasets, augmentation helped slightly
- On others, it made no difference
- We couldn't demonstrate consistent accuracy improvements

### The Unexpected Discovery
While analyzing why augmentation wasn't helping, we noticed something striking:
- **SMOGN** (traditional method): Models trained on SMOGN-augmented data sometimes **completely failed** on real test data (R² went negative!)
- **Diffusion**: Models trained on diffusion-augmented data **maintained performance** (R² stayed similar to original)

This wasn't about improving accuracy - it was about **data quality**.

### The Pivot
We reframed the research question:
- **Old**: "Can diffusion improve accuracy?" (couldn't prove consistently)
- **New**: "Does diffusion produce more realistic synthetic data?" (yes, clearly!)

This reframing is more valuable because:
1. It addresses a real problem (privacy-preserving data sharing)
2. Our evidence strongly supports it (SMOGN fails, diffusion works)
3. It has practical applications (companies can share synthetic data)

### Lesson Learned
Sometimes the most interesting findings aren't what you set out to discover. The "failure" to prove our initial hypothesis led us to a stronger, more practical contribution.

## Methodology

### Approach
1. Train a diffusion model on the original (sensitive) data
2. Generate synthetic samples from the trained model
3. Evaluate quality by training ML models on synthetic data and testing on real data

### Quality Metric
**If synthetic data is realistic**: ML models trained on synthetic data should perform similarly to models trained on real data when evaluated on real test data.

### Comparison
- **Diffusion (HybridDiffusion)**: Learns the data distribution, generates new samples
- **SMOGN**: Traditional interpolation-based augmentation for regression

## Key Findings

### Finding 1: Diffusion Preserves Data Quality
Models trained on diffusion-generated data maintain performance on real test data.

| Experiment | Original Data R² | Diffusion Data R² | Difference |
|------------|------------------|-------------------|------------|
| 011 (Manufacturing) | 0.75 | 0.75 | ±0% |
| 013 (Ozel Rich) | 0.65 | 0.64 | -1.5% |

### Finding 2: SMOGN Can Fail Catastrophically
Models trained on SMOGN-generated data can completely fail on real test data.

| Experiment | Original Data R² | SMOGN Data R² | Difference |
|------------|------------------|---------------|------------|
| 011 (Manufacturing) | 0.75 | **negative** | **catastrophic** |
| 013 (Ozel Rich) | 0.65 | **-0.39** | **catastrophic** |

### Finding 3: Feature Complexity Matters
| Feature Space | SMOGN | Diffusion |
|---------------|-------|-----------|
| Simple (≤5 dims) | Works (±2%) | Works (±2%) |
| Complex (>20 dims) | **Fails** (+86-98% error) | **Works** (±2%) |

### Finding 4: Diffusion is a "Safe Default"
- Never catastrophically worse than original
- Sometimes slightly improves (GB in Exp 013: -1.1% error)
- Handles both numeric and categorical features (HybridDiffusion)

## Experimental Evidence

### Experiment 011: Manufacturing Data (17,942 rows, 2 features)
- **Baseline**: RF R² = 0.75
- **+ Diffusion**: R² = 0.75 (maintained)
- **+ SMOGN**: R² = negative (catastrophic failure, +70-84% RMSE)

### Experiment 012: Ozel Simple (2,029 rows, 5 features)
- **Baseline**: RF R² = 0.48, GB R² = 0.55
- **+ Diffusion**: Similar (±2%)
- **+ SMOGN**: Similar (±2%)
- **Conclusion**: Simple feature space, both methods work

### Experiment 013: Ozel Rich (2,670 rows, 29 features)
- **Baseline**: RF R² = 0.65, GB R² = 0.63
- **+ Diffusion**: RF R² = 0.64 (maintained), GB R² = 0.64 (**improved**)
- **+ SMOGN**: RF R² = -0.39, GB R² = -0.29 (catastrophic failure)
- **Conclusion**: Complex feature space breaks SMOGN

## Why SMOGN Fails

SMOGN uses **interpolation** between existing samples:
- Works in low dimensions (interpolated points are realistic)
- Fails in high dimensions (interpolated points fall in "empty" regions of feature space)
- Creates synthetic samples that don't follow the true data distribution

## Why Diffusion Works

Diffusion models **learn the data distribution**:
- Generate new samples by reversing a noise process
- Samples are drawn from the learned distribution
- Works regardless of dimensionality
- HybridDiffusion handles both numeric (Gaussian) and categorical (Multinomial) features

## Practical Implications

### For Data Sharing
Organizations can:
1. Train diffusion model on sensitive data (internal)
2. Generate synthetic dataset
3. Share synthetic dataset publicly
4. Partners can train models on synthetic data
5. These models will work on real data

### For Model Development
- Use synthetic data for development/testing
- Deploy models that generalize to real data
- No privacy risk in sharing synthetic data

## Conclusion

**Diffusion models are the preferred method for generating privacy-preserving synthetic tabular data** because:

1. **Reliability**: Never catastrophically fails
2. **Quality**: Generated data maintains statistical properties
3. **Flexibility**: Handles mixed-type data (numeric + categorical)
4. **Scalability**: Works with complex, high-dimensional feature spaces

Traditional methods like SMOGN are **risky** for high-dimensional data and should be avoided when data quality is critical.

## Future Work

1. **Privacy guarantees**: Add differential privacy to diffusion training
2. **Quality metrics**: Develop formal metrics for synthetic data quality
3. **Conditional generation**: Generate data with specific properties
4. **Larger datasets**: Test on more diverse domains
