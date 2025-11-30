# Technical Concepts Guide

This document explains the key technical concepts for understanding tabular data augmentation with diffusion models. Use this as a reference for your presentation and Q&A.

---

## Table of Contents

1. [Diffusion Models Fundamentals](#1-diffusion-models-fundamentals)
2. [Tabular Data Challenges](#2-tabular-data-challenges)
3. [Model Architecture Terms](#3-model-architecture-terms)
4. [Training & Optimization](#4-training--optimization)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Baseline Methods](#6-baseline-methods)

---

## 1. Diffusion Models Fundamentals

### What is a Diffusion Model?

**Simple explanation**: A diffusion model learns to generate data by learning to reverse a noise-adding process.

**Analogy**: Imagine dropping ink into water. The ink slowly diffuses until it's evenly spread (noise). A diffusion model learns to reverse this - starting from the noisy state and recovering the original ink drop.

### Forward Process (Adding Noise)

The process of gradually adding Gaussian noise to data over T timesteps.

```
x_0 (real data) → x_1 → x_2 → ... → x_T (pure noise)
```

At each step, we add a small amount of noise according to a schedule (β_1, β_2, ..., β_T).

**Formula**: `x_t = √(1-β_t) * x_{t-1} + √(β_t) * ε`  where ε ~ N(0, I)

**Key property**: We can jump directly from x_0 to any x_t:
`x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε`  where ᾱ_t = α_1 * α_2 * ... * α_t

### Reverse Process (Denoising)

The learned process of removing noise step by step.

```
x_T (noise) → x_{T-1} → ... → x_1 → x_0 (generated data)
```

A neural network learns to predict the noise that was added, then subtracts it.

### DDPM (Denoising Diffusion Probabilistic Model)

The specific framework introduced by Ho et al. (2020) that made diffusion models practical.

**Key insight**: Instead of predicting x_{t-1} directly, predict the noise ε that was added, then compute x_{t-1}.

### Noise Schedule

How much noise to add at each timestep. Common schedules:

- **Linear**: β increases linearly from β_1 to β_T
- **Cosine**: Smoother schedule, often works better (used in TabDDPM)

### Timestep Embedding

A way to tell the neural network which timestep t we're at.

**Sinusoidal embedding**: Uses sine and cosine functions at different frequencies (borrowed from Transformers' positional encoding).

```
emb(t) = [sin(t/10000^0), cos(t/10000^0), sin(t/10000^{2/d}), cos(t/10000^{2/d}), ...]
```

---

## 2. Tabular Data Challenges

### Heterogeneous Features

Tabular data contains mixed types:

| Type | Example | How to handle |
|------|---------|---------------|
| **Numerical (continuous)** | Age=25.5, Income=50000 | Gaussian diffusion |
| **Categorical** | Color=Red, Country=Turkey | Multinomial diffusion |
| **Binary** | HasCar=True/False | Special case of categorical |
| **Ordinal** | Rating=1,2,3,4,5 | Can treat as numerical or categorical |

### Why is tabular data harder than images?

1. **No spatial structure**: Pixels have neighbors; tabular columns don't
2. **Feature independence**: Each column can have completely different distribution
3. **Small datasets**: Tabular datasets often have 1K-100K rows vs millions of images
4. **Mixed types**: Can't apply same operation to all features

### Gaussian Diffusion (for numerical features)

Standard DDPM approach:
- Add Gaussian noise during forward process
- Neural network predicts the noise
- Works because numerical values are continuous

### Multinomial Diffusion (for categorical features)

Modified diffusion for discrete/categorical data:

**Forward process**: Instead of adding Gaussian noise, we "corrupt" by randomly changing categories.

```
Original: [1, 0, 0, 0]  (category A)
    ↓ add noise
Noisy:    [0.7, 0.1, 0.1, 0.1]  (probably A, but uncertain)
    ↓ more noise
Very noisy: [0.25, 0.25, 0.25, 0.25]  (uniform - complete uncertainty)
```

**Reverse process**: Neural network outputs probability distribution over categories.

### One-Hot Encoding

Converting categorical variables to binary vectors:

```
Color: Red, Green, Blue

Red   → [1, 0, 0]
Green → [0, 1, 0]
Blue  → [0, 0, 1]
```

### Quantile Transformation (Gaussian Quantile Transformer)

Preprocessing step that transforms any numerical distribution to standard normal N(0,1).

**Why?** Diffusion models assume Gaussian noise. If your data is already Gaussian-like, it works better.

**How?**
1. Compute the percentile rank of each value
2. Map that percentile to the corresponding value in a standard normal

```
Original: [1, 2, 100, 1000, 10000]  (skewed)
After:    [-1.28, -0.52, 0, 0.52, 1.28]  (normal-ish)
```

---

## 3. Model Architecture Terms

### MLP (Multi-Layer Perceptron)

The simplest neural network: stack of fully-connected layers.

```
Input → Linear → ReLU → Linear → ReLU → ... → Output
```

**TabDDPM uses MLP** (not fancy architectures like U-Net used in image diffusion).

### MLPBlock

A single building block:
```
MLPBlock(x) = Dropout(ReLU(Linear(x)))
```

### Linear Layer

Matrix multiplication + bias: `y = Wx + b`

### ReLU (Rectified Linear Unit)

Activation function: `ReLU(x) = max(0, x)`

Introduces non-linearity so network can learn complex patterns.

### Dropout

During training, randomly set some neurons to 0.

**Why?** Prevents overfitting by forcing network to not rely on specific neurons.

### SiLU (Sigmoid Linear Unit) / Swish

Activation function: `SiLU(x) = x * sigmoid(x)`

Smoother than ReLU, often works better in modern architectures.

### Embedding Layer

Lookup table that converts discrete indices to continuous vectors.

```
Class 0 → [0.2, -0.5, 0.8, ...]
Class 1 → [-0.1, 0.3, 0.4, ...]
Class 2 → [0.7, 0.1, -0.2, ...]
```

Used for class labels in conditional generation.

### Softmax

Converts raw scores (logits) to probabilities that sum to 1.

```
logits: [2.0, 1.0, 0.5]
softmax: [0.59, 0.24, 0.17]  (sums to 1.0)
```

---

## 4. Training & Optimization

### Loss Function

Measures how wrong the model's predictions are. Training minimizes this.

### MSE (Mean Squared Error)

Average squared difference between prediction and target:
```
MSE = (1/n) * Σ(predicted - actual)²
```

Used for Gaussian diffusion (predicting noise).

### KL Divergence (Kullback-Leibler Divergence)

Measures how different two probability distributions are:
```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```

Used for multinomial diffusion (comparing predicted vs true category distributions).

### Variational Lower Bound (VLB / ELBO)

Mathematical framework for training probabilistic models. The loss function is derived from maximizing a lower bound on the data likelihood.

**You don't need to derive it** - just know that the simple MSE loss is a simplification of this.

### Learning Rate

How big of a step to take when updating model weights.
- Too high: Training unstable, diverges
- Too low: Training very slow

### Batch Size

Number of samples processed together before updating weights.
- Larger batch: More stable gradients, needs more memory
- Smaller batch: Noisier updates, can help generalization

### Training Iterations / Epochs

- **Iteration**: One batch of data
- **Epoch**: One pass through entire dataset

### Hyperparameter

Settings you choose before training (not learned):
- Learning rate
- Batch size
- Number of layers
- etc.

### Hyperparameter Tuning

Systematically searching for good hyperparameter values.

**Optuna**: Library that automates this search intelligently.

---

## 5. Evaluation Metrics

### ML Efficiency (Machine Learning Utility)

**The main metric for tabular generation!**

**Process**:
1. Train generator on real data
2. Generate synthetic dataset
3. Train classifier/regressor on synthetic data
4. Evaluate on real test set
5. Compare performance to model trained on real data

**Intuition**: If synthetic data is good, models trained on it should perform similarly to models trained on real data.

### F1 Score

Classification metric that balances precision and recall:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

- **Precision**: Of all predicted positives, how many are actually positive?
- **Recall**: Of all actual positives, how many did we predict?

### R² Score (Coefficient of Determination)

Regression metric: how much variance in target is explained by predictions.
```
R² = 1 - (SS_residual / SS_total)
```

- R² = 1: Perfect predictions
- R² = 0: As good as predicting the mean
- R² < 0: Worse than predicting the mean

### Wasserstein Distance (Earth Mover's Distance)

Measures distance between two probability distributions.

**Intuition**: Minimum "work" needed to transform one distribution into another.

Used to compare numerical feature distributions (real vs synthetic).

### Jensen-Shannon Divergence

Symmetric version of KL divergence, measures similarity between distributions.

Used to compare categorical feature distributions (real vs synthetic).

### Correlation Matrix

Table showing pairwise correlations between all features.

**L2 distance between correlation matrices**: Measures if synthetic data preserves feature relationships.

### DCR (Distance to Closest Record)

Privacy metric: For each synthetic sample, find distance to nearest real sample.

- **Low DCR**: Synthetic samples are very close to real ones (privacy risk!)
- **High DCR**: Synthetic samples are "new" (better privacy)

### Density and Coverage

**Density**: Are synthetic samples in high-density regions of real data? (quality)
**Coverage**: Does synthetic data cover all regions of real data? (diversity)

---

## 6. Baseline Methods

### SMOTE (Synthetic Minority Over-sampling Technique)

Simple interpolation method:
1. Pick a real sample
2. Find its k nearest neighbors
3. Create synthetic sample on the line between them

```
synthetic = real_sample + λ * (neighbor - real_sample)
```

where λ ∈ [0, 1]

**Pros**: Simple, fast, effective for ML utility
**Cons**: Poor privacy (synthetic points are interpolations of real ones)

### GAN (Generative Adversarial Network)

Two networks competing:
- **Generator**: Creates fake data
- **Discriminator**: Tries to distinguish real from fake

Training is a minimax game - generator tries to fool discriminator.

**CTGAN, CTABGAN, CTABGAN+**: GAN variants designed for tabular data.

### VAE (Variational Autoencoder)

- **Encoder**: Compresses data to latent space
- **Decoder**: Reconstructs data from latent space
- Sample from latent space to generate new data

**TVAE**: VAE variant for tabular data.

### Why diffusion models beat GANs/VAEs for tabular data?

1. **Training stability**: GANs are notoriously hard to train (mode collapse, etc.)
2. **Better coverage**: Diffusion models don't suffer from mode collapse
3. **Quality**: The iterative denoising produces higher quality samples

---

## Quick Reference Card

| Term | One-line explanation |
|------|---------------------|
| DDPM | Learn to denoise data step by step |
| Forward process | Gradually add noise to data |
| Reverse process | Gradually remove noise (learned) |
| Gaussian diffusion | For continuous/numerical features |
| Multinomial diffusion | For categorical features |
| MLP | Simple feedforward neural network |
| One-hot encoding | Category → binary vector |
| Quantile transform | Any distribution → Gaussian |
| ML efficiency | Train on synthetic, test on real |
| DCR | Privacy metric (higher = better) |
| SMOTE | Interpolation baseline |

---

## Potential Instructor Questions

1. **"Why not just use GANs?"**
   - Diffusion models are more stable to train and don't suffer from mode collapse.

2. **"How do you handle categorical features?"**
   - Multinomial diffusion: corrupt by mixing with uniform distribution, learn to reverse.

3. **"What's the main evaluation metric?"**
   - ML efficiency: train model on synthetic, test on real data.

4. **"How does this compare to simply oversampling?"**
   - SMOTE achieves similar ML utility but has poor privacy (interpolates real samples).

5. **"What preprocessing is needed?"**
   - Numerical: Quantile transformation to Gaussian
   - Categorical: One-hot encoding

6. **"How many diffusion steps?"**
   - Typically 100-1000. More steps = better quality but slower sampling.

---

## 7. Score-Based Generative Models (from STaSy)

### Score Function

The gradient of the log probability density: `∇x log p(x)`

**Intuition**: Points in the direction where data is more likely. If you follow the score, you move toward high-density regions.

### Score-Based Generative Modeling (SGM)

Alternative formulation of diffusion models using score functions instead of directly predicting noise.

**Key idea**: Instead of learning to denoise, learn the score function at each noise level, then use it to generate samples.

### Stochastic Differential Equation (SDE)

Continuous-time version of diffusion. Instead of discrete steps (t=1,2,3...), we have continuous time t ∈ [0, T].

**Forward SDE**: `dx = f(x,t)dt + g(t)dw`
- `f(x,t)`: Drift coefficient (deterministic part)
- `g(t)`: Diffusion coefficient (noise scale)
- `dw`: Wiener process (continuous random walk)

**Reverse SDE**: `dx = [f(x,t) - g²(t)∇x log p(x)]dt + g(t)dw̄`
- Uses the score function to reverse the noise process

### SDE Variants

| Type | Name | Behavior |
|------|------|----------|
| **VE** | Variance Exploding | Variance grows without bound |
| **VP** | Variance Preserving | Total variance stays constant |
| **sub-VP** | Sub-Variance Preserving | Variance bounded below VP |

### Probability Flow ODE

A deterministic (non-random) version of the reverse process:
```
dx = [f(x,t) - ½g²(t)∇x log p(x)]dt
```

**Advantage**: Allows exact likelihood computation and faster sampling.

### Denoising Score Matching

Training objective for score-based models:
```
L = E[λ(t) · ‖Sθ(x(t), t) - ∇x(t) log p(x(t)|x(0))‖²]
```

Learn a neural network Sθ to approximate the score function.

### Self-Paced Learning (SPL)

Training strategy: start with "easy" samples, gradually include "harder" ones.

**Why for tabular diffusion?** Some records have high loss and destabilize training. SPL helps by:
1. Training first on low-loss (easy) records
2. Gradually including high-loss (hard) records
3. Model becomes robust before seeing difficult cases

**Implementation**:
- Compute loss for each record
- Assign selection weight vi ∈ [0, 1] based on loss
- Records with loss < threshold get vi = 1 (fully included)
- Records with loss > threshold get vi = 0 (excluded)
- Thresholds gradually increase to include all records

### Quantile Function Q(p)

Returns the value at percentile p of a distribution.

```
Q(0.5) = median
Q(0.25) = first quartile
Q(0.9) = 90th percentile
```

Used in STaSy to set thresholds for self-paced learning.

### Fine-tuning with Log-Probability

After main training, fine-tune on records with below-average log-probability.

**Why?** These are records the model finds "hard" - focusing on them improves coverage/diversity.

---

## 8. Latent Diffusion Models (from TabSyn)

### Latent Space

A compressed representation of the data learned by an autoencoder.

```
Original data (high-dim) → Encoder → Latent space (lower-dim) → Decoder → Reconstructed data
```

**Advantages**:
- Lower dimensionality = faster diffusion
- Smoother space = easier to learn
- Can unify different data types

### Latent Diffusion Model (LDM)

Apply diffusion in latent space instead of data space:

```
Data x → Encode → z₀ → Diffuse → zT → Denoise → z₀ → Decode → x̂
```

**Key insight from TabSyn**: Diffusion works better in a well-structured latent space than directly on mixed-type tabular data.

### Variational Autoencoder (VAE)

Autoencoder with probabilistic latent space:
- Encoder outputs mean μ and variance σ² (not just a point)
- Sample z = μ + ε·σ where ε ~ N(0,1) (reparameterization trick)
- Decoder reconstructs from z

**Loss**: `L = L_reconstruction + β · L_KL`
- Reconstruction loss: How well can we reconstruct?
- KL divergence: How close is latent distribution to N(0,1)?

### β-VAE

VAE with adjustable weight β on KL divergence term.
- β > 1: More regularized latent space, potentially worse reconstruction
- β < 1: Better reconstruction, less regularized latent space

### Adaptive β Scheduling (TabSyn)

Dynamically adjust β during training:
1. Start with β = β_max (e.g., 0.01)
2. Monitor reconstruction loss
3. When loss plateaus, multiply β by λ < 1 (e.g., 0.7)
4. Continue until β reaches β_min (e.g., 10⁻⁵)

**Result**: Good reconstruction AND regularized latent space.

### Tokenizer (for tabular data)

Convert each column to a fixed-dimensional embedding:

**Numerical**: Linear projection
```
e_num = x · w + b
```

**Categorical**: Embedding lookup (like word embeddings)
```
e_cat = OneHot(x) · W + b
```

Each column becomes a d-dimensional "token".

### Transformer Architecture

Neural network using self-attention to model relationships between tokens.

**Key components**:
- **Self-attention**: Each token can "look at" all other tokens
- **Multi-head attention**: Multiple attention patterns in parallel
- **Layer normalization**: Stabilizes training
- **Feed-forward layers**: Process each token

**Why for tabular data?** Captures inter-column dependencies through attention.

### Token-Level Representation

Each column has its own embedding vector (token). The full record is a matrix of tokens:
```
E = [e₁, e₂, ..., eₘ] ∈ R^(M×d)
```

where M = number of columns, d = embedding dimension.

### Linear Noise Schedule

In TabSyn: σ(t) = t (noise level equals time)

**Theorem**: This minimizes approximation errors in the reverse process.

**Practical benefit**: High-quality samples with only ~20 steps (vs 50-1000 for other methods).

### NFE (Number of Function Evaluations)

How many times we call the denoising network during sampling.

More NFEs = better quality but slower.

| Method | Typical NFEs |
|--------|--------------|
| TabDDPM | 1000 |
| STaSy | 50-200 |
| TabSyn | <20 |

---

## 9. Additional Evaluation Metrics

### Kolmogorov-Smirnov Test (KST)

Statistical test comparing two distributions (for numerical columns).

**Returns**: Maximum difference between cumulative distributions.
- 0 = identical distributions
- 1 = completely different

### Total Variation Distance (TVD)

Distance between two categorical distributions:
```
TVD = ½ Σ |P(x) - Q(x)|
```

- 0 = identical distributions
- 1 = no overlap

### Pair-wise Column Correlation

Measures if synthetic data preserves relationships between columns.

**Process**:
1. Compute correlation matrix for real data
2. Compute correlation matrix for synthetic data
3. Measure difference (lower = better)

### α-Precision and β-Recall

**α-Precision**: What fraction of synthetic samples are realistic?
- High = synthetic samples look like real data

**β-Recall**: What fraction of real data modes are covered?
- High = synthetic data covers the full distribution

**Trade-off**: Can have high precision (few but good samples) or high recall (diverse but some bad samples).

### Classifier Two-Sample Test (C2ST)

Train a classifier to distinguish real vs synthetic data.

- Accuracy ≈ 50%: Cannot distinguish → synthetic data is realistic
- Accuracy ≈ 100%: Easily distinguished → synthetic data is unrealistic

### MLE (Machine Learning Efficiency)

Train on synthetic, test on real. Same as "ML Efficiency" but abbreviated.

### TSTR (Train on Synthetic, Test on Real)

The standard evaluation framework:
1. Generate synthetic training data
2. Train ML model on synthetic data
3. Evaluate on held-out real test data
4. Compare to model trained on real data

---

## 10. Updated Quick Reference

| Term | One-line explanation |
|------|---------------------|
| Score function | Gradient of log probability, points toward high-density |
| SDE | Continuous-time diffusion process |
| VE/VP/sub-VP | Different SDE formulations |
| Probability flow ODE | Deterministic reverse process |
| Self-paced learning | Train easy samples first, hard samples later |
| Latent diffusion | Apply diffusion in encoded space |
| VAE | Autoencoder with probabilistic latent space |
| β-VAE | VAE with tunable KL weight |
| Tokenizer | Convert columns to embeddings |
| Transformer | Attention-based architecture for sequences |
| NFE | Number of denoising steps during sampling |
| KST | Test for comparing numerical distributions |
| TVD | Distance between categorical distributions |
| C2ST | Can a classifier tell real from synthetic? |
| TSTR | Train on synthetic, test on real |

---

## Updated Potential Instructor Questions

7. **"What is the difference between TabDDPM and STaSy?"**
   - TabDDPM: Discrete diffusion, separate handling of numerical/categorical
   - STaSy: Continuous SDE, adds self-paced learning for training stability

8. **"Why use latent diffusion (TabSyn) instead of data-space diffusion?"**
   - Unified handling of mixed types
   - Better captures inter-column correlations (Transformer)
   - Faster sampling (~20 steps vs 1000)
   - More robust to dataset variations

9. **"What is self-paced learning?"**
   - Training strategy that starts with easy samples (low loss) and gradually includes harder ones. Stabilizes training for tabular diffusion.

10. **"Why does TabSyn need fewer sampling steps?"**
    - Linear noise schedule (σ(t) = t) minimizes discretization errors
    - Better latent space from adaptive β-VAE training
    - 20 steps achieves what others need 1000 for
