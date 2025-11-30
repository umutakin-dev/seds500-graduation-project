# Paper Analysis Notes

This document contains detailed analysis of the three key papers for this project.

---

## Overview

| Paper | Venue | Year | Key Innovation |
|-------|-------|------|----------------|
| TabDDPM | ICML | 2023 | Hybrid diffusion (Gaussian + Multinomial) |
| STaSy | ICLR | 2023 | Self-paced learning for training stability |
| TabSyn | ICLR | 2024 | Latent diffusion with Transformer VAE |

---

## 1. TabDDPM (ICML 2023)

**Paper**: "TabDDPM: Modelling Tabular Data with Diffusion Models" - Kotelnikov et al.

**Core Problem**: Tabular data has heterogeneous features (numerical + categorical), making it challenging to model with standard diffusion approaches designed for continuous data.

### Key Ideas

1. **Hybrid diffusion approach**:
   - Gaussian diffusion for numerical features
   - Multinomial diffusion for categorical features (one-hot encoded)
2. **Class-conditional generation**: For classification tasks, learns p(x|y) conditioned on class labels
3. **Joint target modeling**: For regression, treats target as additional numerical feature

### Architecture

- Simple MLP backbone (not transformer/U-Net like image diffusion)
- Sinusoidal time embeddings (dimension 128)
- Class label embeddings added to input
- Input: `x = Linear(x_in) + t_emb + y_emb`
- Preprocessing: Gaussian quantile transformation for numerical features

### Loss Function

```
L_TabDDPM = L_simple (MSE for Gaussian) + (sum of KL divergences for multinomial) / C
```

### Key Hyperparameters

- Learning rate: LogUniform[0.00001, 0.003]
- Batch size: {256, 4096}
- Diffusion timesteps: {100, 1000}
- Training iterations: {5000, 10000, 20000}
- MLP layers: {2, 4, 6, 8}
- MLP width: {128, 256, 512, 1024}

### Evaluation Metrics

- ML efficiency (train classifier on synthetic, test on real)
- Wasserstein distance (numerical features)
- Jensen-Shannon divergence (categorical features)
- L2 distance between correlation matrices
- DCR (Distance to Closest Record) for privacy

### Results

- Outperforms TVAE, CTGAN, CTABGAN+ on most datasets
- Competitive with SMOTE on ML efficiency but much better privacy
- Better captures feature correlations than GAN/VAE methods

### Strengths

- Simple architecture (just MLP)
- Handles mixed data types naturally
- Better privacy than interpolation methods (SMOTE)
- State-of-the-art ML efficiency

### Weaknesses

- Multinomial diffusion may not be optimal (paper mentions alternatives)
- Hyperparameter sensitive (50 tuning trials recommended)
- Training time: 8-10 hours for hyperparameter search

### Code

https://github.com/rotot0/tab-ddpm

---

## 2. STaSy (ICLR 2023)

**Paper**: "STaSy: Score-based Tabular data Synthesis" - Kim, Lee, Park (Yonsei University)

**Core Problem**: Training score-based generative models on tabular data is difficult due to uneven loss distributions across records. Some records are "hard" and destabilize training.

### Key Ideas

1. **Score-based generative modeling (SGM)**: Uses continuous-time SDEs for diffusion/denoising
   - Forward SDE: `dx = f(x,t)dt + g(t)dw` (adds noise)
   - Reverse SDE: `dx = [f(x,t) - g²(t)∇x log pt(x)]dt + g(t)dw` (denoises)
   - Supports VE (Variance Exploding), VP (Variance Preserving), sub-VP variants
2. **Self-Paced Learning (SPL)**: Train from easy to hard records
   - Records with low loss are "easy", high loss are "hard"
   - Gradually include harder records as training progresses
   - Uses soft selection weights vi ∈ [0,1] instead of binary
3. **Fine-tuning with log-probability**: After main training, fine-tune on records with below-average log-probability to improve coverage

### Architecture

- MLP-based score network with skip connections
- Time embedding via layer-specific conditioning:
  - Squash: `FC(h) ⊙ σ(FC(t))`
  - Concat: `FC(t ⊕ h)`
  - Concatsquash: `FC(h) ⊙ σ(FC_gate(t) + FC_bias(t))`
- Only T=50 diffusion steps needed (vs T=1000 for images)
- Uses probability flow ODE for sampling (deterministic)

### Loss Function

```
L_STaSy = Σᵢ vᵢ·lᵢ + r(v; α, β)
```
where lᵢ is denoising score matching loss for record i, vᵢ is selection weight, and r is self-paced regularizer.

### Self-Paced Regularizer

```
r(v; α, β) = -[(Q(α) - Q(β))/2] Σvᵢ² - Q(β) Σvᵢ
```
where Q(p) is quantile function of loss distribution.

### Preprocessing

- Min-max scaling for numerical columns
- One-hot encoding for categorical columns
- Post-processing: reverse scaling, softmax + rounding for categorical

### Key Hyperparameters

- α₀, β₀: Initial thresholds for SPL (recommend α₀=0.2-0.25, β₀=0.9-0.95)
- SDE type: VE, VP, or sub-VP
- σ_min, σ_max: Noise schedule parameters
- S: Training steps before using full data (default 10,000)

### Evaluation

- TSTR framework (Train on Synthetic, Test on Real)
- Metrics: F1, AUROC, R², RMSE, coverage (diversity)
- 15 benchmark datasets, 7 baselines

### Results

| Method | Quality (F1/R²) | Diversity (coverage) | Runtime (sec) |
|--------|-----------------|---------------------|---------------|
| CTGAN | 0.569 | 0.352 | 0.704 |
| TableGAN | 0.501 | 0.434 | 0.046 |
| OCT-GAN | 0.567 | 0.381 | 26.926 |
| Naïve-STaSy | 0.717 | 0.637 | 8.855 |
| **STaSy** | **0.733** | **0.658** | 10.663 |

### Strengths

- Best sampling quality AND diversity among baselines
- Self-paced learning stabilizes training
- Handles multi-modal distributions well (see histograms)
- Good on small datasets and imbalanced classes
- Only 50 diffusion steps (faster than typical 1000)

### Weaknesses

- Slower than simple GAN methods (10s vs 0.05s for TableGAN)
- Requires hyperparameter tuning for SPL thresholds
- More complex training procedure than TabDDPM

### Code

https://github.com/JayoungKim408/STaSy

### Comparison to TabDDPM

- STaSy uses continuous SDEs vs TabDDPM's discrete diffusion
- STaSy has unified approach (all continuous) vs TabDDPM's hybrid (Gaussian + Multinomial)
- STaSy adds SPL and fine-tuning for training stability
- Both use MLP backbone
- STaSy evaluates diversity more explicitly (coverage metric)

---

## 3. TabSyn (ICLR 2024)

**Paper**: "Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space" - Zhang et al. (UIC & Amazon AWS)

**Core Problem**: Directly applying diffusion to tabular data is challenging due to mixed types (numerical + categorical) and complex inter-column dependencies. Existing methods either use suboptimal encodings or separate diffusion processes.

### Key Ideas

1. **Latent Diffusion for Tabular Data**: First encode tabular data into a continuous latent space via VAE, then apply diffusion in that space
   - Unifies numerical and categorical features
   - Captures inter-column relationships
2. **Transformer-based VAE**: Uses Transformer architecture for encoder/decoder to model column relationships
   - Column-wise tokenizer (embeddings per column)
   - Token-level representations enable inter-column attention
3. **Adaptive β-VAE Training**: Dynamically schedule β (KL weight) during training
   - Start with higher β, gradually decrease when reconstruction loss plateaus
   - Achieves good reconstruction while maintaining regularized latent space
4. **Linear Noise Schedule**: Use σ(t) = t (linear w.r.t. time)
   - Proven to minimize approximation errors in reverse process
   - Enables high-quality generation with <20 NFEs (vs 50-1000 for others)

### Architecture

```
Tabular Data → Tokenizer → Transformer Encoder → Latent z
     ↓
Diffusion in Latent Space (z₀ → zₜ → z₀)
     ↓
Transformer Decoder → Detokenizer → Synthetic Data
```

### Tokenizer/Detokenizer

- Numerical: `e_num = x_num · w + b` (linear projection)
- Categorical: `e_cat = OneHot(x) · W + b` (embedding lookup)
- Each column gets d-dimensional embedding
- Stacked into token matrix E ∈ R^(M×d)

### Loss Functions

- VAE: `L = ℓ_recon + β·ℓ_kl` with adaptive β scheduling
- Diffusion: `L = E[‖ε_θ(z_t, t) - ε‖²]` (denoising score matching)

### Key Hyperparameters

- β_max, β_min: Bounds for adaptive β (e.g., 0.01 to 10⁻⁵)
- λ: β decay factor (e.g., 0.7)
- d: Token embedding dimension
- NFEs: Number of function evaluations for sampling (<20)

### Evaluation Metrics (Comprehensive)

1. **Low-order statistics**:
   - Column-wise density (KST for numerical, TVD for categorical)
   - Pair-wise column correlation (Pearson, contingency similarity)
2. **High-order metrics**: α-precision, β-recall
3. **Privacy**: DCR (Distance to Closest Records)
4. **Downstream tasks**: MLE (train on synthetic, test on real)

### Results

| Method | Single Density Error | Pair Correlation Error | Avg MLE Gap |
|--------|---------------------|----------------------|-------------|
| CTGAN | 17.02% | 15.93% | 24.5% |
| STaSy | 7.72% | 7.77% | 10.9% |
| TabDDPM | 14.52%* | 5.34% | 9.14% |
| **TabSyn** | **1.08%** | **1.73%** | **7.23%** |

*TabDDPM fails on some datasets (News)

### Key Improvements

- 86% reduction in column-wise density error vs best baseline
- 67% reduction in pair-wise correlation error vs best baseline
- Only needs <20 NFEs vs 50-1000 for other diffusion methods

### Strengths

- Best overall quality metrics by large margin
- Unified handling of mixed types in latent space
- Very fast sampling (~20 steps)
- Captures inter-column correlations well (Transformer attention)
- Works on datasets where TabDDPM fails (News)
- Can be used for missing value imputation without retraining

### Weaknesses

- Two-stage training (VAE then diffusion) - more complex pipeline
- Requires tuning adaptive β schedule
- Latent space adds indirection (potential information loss)

### Code

https://github.com/amazon-science/tabsyn

---

## Summary Comparison

| Aspect | TabDDPM | STaSy | TabSyn |
|--------|---------|-------|--------|
| Diffusion space | Data space | Data space | Latent space |
| Categorical handling | Multinomial diffusion | Continuous (one-hot) | VAE embedding |
| Architecture | MLP | MLP + skip | Transformer VAE + MLP |
| NFEs needed | 1000 | 50-200 | <20 |
| Inter-column modeling | Implicit | Implicit | Explicit (attention) |
| Training stability | Standard | Self-paced learning | Adaptive β-VAE |
| SOTA status | 2023 baseline | Training improvements | Current SOTA |

---

## Key Takeaways for Implementation

1. **TabDDPM**: Good baseline, simple architecture. Start here for understanding.

2. **STaSy**: Self-paced learning is valuable for training stability. Could be combined with other approaches.

3. **TabSyn**: Current SOTA. Key insights:
   - Latent space diffusion > data space diffusion
   - Transformer attention captures inter-column relationships
   - Adaptive β-VAE training is crucial
   - Linear noise schedule enables fast sampling
