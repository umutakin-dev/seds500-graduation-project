# Add Distribution Comparison Metrics

**Priority:** HIGH (Easy win)
**Effort:** 1 day
**Addresses:** "Single metric (R²) limitation"

## Problem

Currently only using "train on synthetic, test on real" R² as quality metric. Should add statistical distribution comparison to show synthetic data matches original distribution.

## Metrics to Add

### 1. Column-wise Statistics
- Mean, std, min, max comparison (real vs synthetic)
- Per-column correlation with target

### 2. Correlation Matrix Comparison
- Compute correlation matrix for real data
- Compute correlation matrix for synthetic data
- Compare (Frobenius norm of difference)

### 3. Distribution Metrics
- **KL Divergence** (per column, binned for continuous)
- **Wasserstein Distance** (per column)
- **Maximum Mean Discrepancy (MMD)** (overall)

### 4. Visual Comparison
- Histograms overlay (real vs synthetic) per feature
- Scatter plots for key feature pairs

## Implementation

```python
from scipy import stats
from scipy.spatial.distance import jensenshannon
import numpy as np

def compare_distributions(real_df, synthetic_df):
    """Compare statistical properties of real vs synthetic data."""
    results = {}

    for col in real_df.columns:
        real = real_df[col].values
        synth = synthetic_df[col].values

        # Basic stats
        results[col] = {
            'mean_diff': abs(real.mean() - synth.mean()),
            'std_diff': abs(real.std() - synth.std()),
            'ks_statistic': stats.ks_2samp(real, synth).statistic,
            'wasserstein': stats.wasserstein_distance(real, synth),
        }

    return results
```

## Apply To

- Experiment 011 (Manufacturing)
- Experiment 013 (Ozel Rich)
- Any new experiments

## Tasks

- [ ] Implement distribution comparison functions
- [ ] Run on existing experiments (011, 013)
- [ ] Generate comparison tables and visualizations
- [ ] Add to experiment RESULTS.md files
- [ ] Include in thesis findings
