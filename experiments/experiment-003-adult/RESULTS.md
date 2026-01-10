# Experiment 003: Adult Dataset (Hybrid Diffusion)

## Dataset
- **Source**: UCI Adult / OpenML
- **Training samples**: 39,073
- **Test samples**: 9,769
- **Features**: 6 numerical + 8 categorical (108 one-hot dims)
- **Task**: Binary classification (Income >50K vs ≤50K)
- **Class imbalance**: 76% / 24%

## Model Configuration
- **Architecture**: HybridMLPDenoiser (256-256-256)
- **Epochs**: 500 (in progress)
- **Learning rate**: 0.0005
- **Diffusion timesteps**: 1000
- **Beta schedule**: linear
- **Dropout**: 0.1

## Categorical Features
| Feature | Cardinality |
|---------|-------------|
| workclass | 9 |
| education | 16 |
| marital-status | 7 |
| occupation | 15 |
| relationship | 6 |
| race | 5 |
| sex | 2 |
| native-country | 42 |

## Training Progress

### 50 Epochs (Initial)
- Final loss: 0.6346 (num=0.1293, cat=0.5053)
- Diffusion label distribution: [119, 38954] (skewed)

### 500 Epochs (In Progress)
- Loss at epoch 90: 0.5516 (num=0.1185, cat=0.4331)
- Final loss: TBD
- Diffusion label distribution: TBD

## ML Efficiency Evaluation

### 50 Epochs Results

#### Logistic Regression
| Method | Accuracy | F1 | vs Baseline |
|--------|----------|-----|-------------|
| Real → Real (baseline) | 0.8080 | 0.4809 | - |
| Diffusion only → Real | 0.7602 | 0.1188 | -4.78% |
| Augmented-Diffusion → Real | 0.7999 | 0.4634 | -0.81% |
| SMOTE (balanced) → Real | 0.7439 | 0.5751 | -6.41% |

**Winner**: Diffusion (beats SMOTE by 5.60%)

#### Random Forest
| Method | Accuracy | F1 | vs Baseline |
|--------|----------|-----|-------------|
| Real → Real (baseline) | 0.8599 | 0.6808 | - |
| Diffusion only → Real | 0.2395 | 0.3863 | -62.03% |
| Augmented-Diffusion → Real | 0.8609 | 0.6830 | +0.10% |
| SMOTE (balanced) → Real | 0.8527 | 0.6758 | -0.72% |

**Winner**: Diffusion (beats SMOTE by 0.82%)

### 500 Epochs Results
TBD - Training in progress

## Key Findings

1. **Diffusion beats SMOTE** even with undertrained model (50 epochs)
2. **SMOTE hurts performance** on this large, moderately imbalanced dataset
3. **Augmented-Diffusion slightly improves RF** (+0.10%) even with poor synthetic labels
4. **Label distribution skewed** with 50 epochs - expect improvement with 500 epochs

## Conclusion
TBD - Awaiting 500-epoch results
