# Experiment 018: TabDDPM-style Multinomial Diffusion

## Date
2026-01-14

## Motivation
Our current implementation differs from TabDDPM in several key ways that may affect performance. After analyzing the TabDDPM source code (`repos/tab-ddpm/tab_ddpm/gaussian_multinomial_diffsuion.py`), we identified these differences:

| Feature | TabDDPM | Our Implementation |
|---------|---------|-------------------|
| Categorical input | Log probabilities | Raw one-hot |
| Internal operations | Log-space (stable) | Direct probability |
| Loss function | KL divergence | Cross-entropy |
| Sampling | Gumbel-softmax | Direct softmax |
| Parameterization | Predict x0 → posterior | Predict noise |

## Hypothesis
Implementing TabDDPM-style multinomial diffusion will:
1. Improve numerical stability (log-space operations)
2. Better capture categorical distributions (KL loss on posteriors)
3. Lead to better replacement performance (target: >0.20, closer to CTGAN's 0.23)

## Background

### Current Approach (Cross-Entropy)
```python
# Our current loss
loss_cat = F.cross_entropy(predicted_logits, targets.argmax(dim=-1))
```
- Directly predicts original category from noised input
- Ignores diffusion process structure

### TabDDPM Approach (KL Divergence on Posteriors)
```python
# TabDDPM loss (from gaussian_multinomial_diffsuion.py:510-527)
log_true_prob = q_posterior(log_x_start, log_x_t, t)  # True posterior
log_model_prob = p_pred(model_out, log_x_t, t)       # Model prediction
kl = multinomial_kl(log_true_prob, log_model_prob)   # KL divergence
```
- Model predicts x0, then posterior is computed
- Loss is KL divergence between true and predicted posteriors
- Respects diffusion process structure

### Key Helper Functions (from utils.py)
```python
def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def index_to_log_onehot(x, num_classes):
    # Convert category indices to log one-hot
    ...
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot
```

### Gumbel-Softmax Sampling (from gaussian_multinomial_diffsuion.py:461-471)
```python
def log_sample_categorical(self, logits):
    uniform = torch.rand_like(one_class_logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample = (gumbel_noise + one_class_logits).argmax(dim=1)
```

## Implementation Plan

### Phase 1: Core Changes
1. **Add log-space helper functions** to `diffusion.py`
2. **Convert categorical input to log probabilities** during training
3. **Implement proper posterior computation** q(x_{t-1}|x_t, x_0)
4. **Add KL divergence loss** for categoricals
5. **Implement Gumbel-softmax sampling**

### Phase 2: Training
- Train on Ozel Rich dataset (same as V6)
- Same hyperparameters for fair comparison
- 1000 epochs

### Phase 3: Evaluation
- Test augmentation and replacement scenarios
- Compare with V6 baseline and CTGAN

## Baseline (Current V6 Results)
| Scenario | R² | % of Baseline |
|----------|-----|---------------|
| Augmentation | 0.6355 | 98.5% |
| Replacement | 0.1712 | 26.5% |
| CTGAN Replacement | 0.2292 | 35.5% |

## Expected Outcome
- Replacement R² > 0.20 (currently 0.1712)
- Closer to or exceeding CTGAN's 0.2292

## Files
- `src/diffusion_tabddpm.py` - TabDDPM-style diffusion implementation
- `src/train_experiment_018.py` - Training script
- `RESULTS.md` - Experiment results (created after running)

## Status
[x] Implementation
[x] Training
[x] Evaluation
[x] Documentation

## Results Summary
**BREAKTHROUGH**: Replacement R² = 0.5628 (87.3% of baseline)
- 3.3x better than V6 (0.1712)
- 2.5x better than CTGAN (0.2292)

See [RESULTS.md](RESULTS.md) for full details.

## References
- TabDDPM source: `repos/tab-ddpm/tab_ddpm/gaussian_multinomial_diffsuion.py`
- TabDDPM utils: `repos/tab-ddpm/tab_ddpm/utils.py`
