# Tabular Data Augmentation using Diffusion Models

Master's Graduation Project (SEDS500)

## Why This Project?

Our ML team has consistently struggled with **data acquisition**:
- Data was simply **not there** (no existing datasets for our use case)
- Data was **not shared** (organizational silos, privacy concerns, proprietary restrictions)
- Data was **limited** (small sample sizes, class imbalance, rare events)

**Classic data augmentation techniques did not improve model accuracy.**

Traditional approaches like SMOTE, random oversampling, or simple perturbations failed to capture the complex relationships in tabular data.

### The Goal

This study is a **series of experiments** to answer a critical question:

> Can we create a data augmentation system using diffusion models that actually increases model accuracy?

### Success Criteria
- Synthetic data that preserves statistical properties of real data
- Improved ML model performance when training on augmented datasets
- A practical, reproducible pipeline for tabular data augmentation

### Why Diffusion Models?
- State-of-the-art results on image generation (DALL-E, Stable Diffusion)
- Recent papers (TabDDPM, STaSy, TabSyn) show promise for tabular data
- Can capture complex feature dependencies
- Better privacy properties than GANs

## Background

Tabular data augmentation is challenging because:
- Mixed data types (continuous, categorical, ordinal)
- Complex feature dependencies
- No spatial/sequential structure to exploit (unlike images/text)

Diffusion models have recently shown promising results in this domain.

## Key Papers

| Paper | Venue | Year | Key Innovation |
|-------|-------|------|----------------|
| TabDDPM | ICML | 2023 | Hybrid diffusion (Gaussian + Multinomial) |
| STaSy | ICLR | 2023 | Self-paced learning for training stability |
| TabSyn | ICLR | 2024 | Latent diffusion with Transformer VAE (SOTA) |

## Project Structure

```
├── README.md
├── pyproject.toml    # Dependencies (managed by uv)
├── CLAUDE.md         # AI assistant instructions
├── docs/             # Documentation
│   ├── paper-analysis.md
│   └── concepts.md
├── papers/           # Reference papers (not tracked)
├── repos/            # Cloned reference implementations (not tracked)
├── src/              # Our implementation
│   ├── diffusion.py  # Gaussian diffusion process
│   ├── models.py     # MLP denoiser networks
│   └── train.py      # Training script
├── notebooks/        # Experiments and analysis
├── experiments/      # Organized experiment folders
├── data/             # Datasets (not tracked)
├── checkpoints/      # Model checkpoints (not tracked)
├── results/          # Experiment results
└── todo/             # Task tracking
```

## Development Guidelines

### Experiments Structure
Each experiment should be in its own folder for reproducibility:
```
experiments/
├── exp001-iris-baseline/
│   ├── config.yaml
│   ├── results/
│   └── README.md
├── exp002-california-gaussian/
├── exp003-adult-multinomial/
└── ...
```

### Methodology Notes

**Cross-Validation vs Dropout:**
- **Cross-validation (k-fold):** For robust performance estimation during evaluation
- **Dropout:** Regularization during training to prevent overfitting
- Different purposes but both address generalization

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python dependency management.

```bash
# Install dependencies
uv sync

# Run training
uv run python src/train.py --dataset iris --epochs 500
```

## Usage

```bash
# Train on Iris dataset (small, fast)
uv run python src/train.py --dataset iris --epochs 500

# Train on California Housing (larger)
uv run python src/train.py --dataset california --epochs 2000 --device cuda

# Options
#   --dataset: iris, california
#   --epochs: number of training epochs
#   --batch_size: batch size (default: 64)
#   --timesteps: diffusion timesteps (default: 1000)
#   --device: cpu or cuda
```

## References

- [TabDDPM](https://github.com/rotot0/tab-ddpm) - Kotelnikov et al., ICML 2023
- [STaSy](https://github.com/JayoungKim408/STaSy) - Kim et al., ICLR 2023
- [TabSyn](https://github.com/amazon-science/tabsyn) - Zhang et al., ICLR 2024

---

## Development Log

### 2025-12-01: Initial Implementation

**Setup:**
- Created project structure (papers/, src/, notebooks/, data/, results/, repos/)
- Downloaded papers: TabDDPM, STaSy, TabSyn (PDFs + extracted text)
- Cloned reference repositories to `repos/`
- Set up `uv` for dependency management with `pyproject.toml`

**Implementation - Minimal Gaussian Diffusion:**
- `src/diffusion.py`: Gaussian diffusion process
  - Linear and cosine beta schedules
  - Forward process q(x_t | x_0)
  - Reverse process p(x_{t-1} | x_t)
  - Training loss (simplified DDPM)
  - Sampling (full reverse process)

- `src/models.py`: MLP-based denoisers
  - `MLPDenoiser`: Simple MLP with timestep embeddings
  - `ResidualMLPDenoiser`: MLP with skip connections
  - Sinusoidal timestep embeddings (from Transformers)

- `src/train.py`: Training script
  - Supports Iris and California Housing datasets
  - Quantile transformation preprocessing
  - Basic evaluation (mean/std comparison, correlation matrix)

**First Test (Iris dataset):**
- 500 epochs, ~1 minute on CPU
- Results: Synthetic data statistics match real data well
- Mean absolute correlation difference: 0.037

**Next steps:**
- [x] Add GPU support (CUDA)
- [ ] Add multinomial diffusion for categorical features
- [ ] Implement ML efficiency evaluation
- [ ] Test on larger datasets

### 2025-12-02: GPU Support & Notebook Tutorial

**GPU Setup:**
- Configured `pyproject.toml` to use PyTorch with CUDA 12.4
- Verified GPU detection: NVIDIA GeForce RTX 4070 Ti SUPER

**Added Dependencies:**
- matplotlib (for visualizations)
- ipykernel (for Jupyter notebook support)

**Created Tutorial Notebook:**
- `notebooks/01_diffusion_explained.ipynb`
- Step-by-step explanation of diffusion concepts
- Visualizations: Gaussian noise, forward process, noise schedule
- Training example on Iris dataset
- Comparison of real vs synthetic distributions

**Current Status:**
- Basic Gaussian diffusion working on GPU
- Can generate synthetic numerical data
- Need to add categorical feature support next
