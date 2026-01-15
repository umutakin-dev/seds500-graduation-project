# Experiment 019: TabDDPM on Production Data

## Objective

Apply TabDDPM-style diffusion (same methodology as Experiment 018) to the Production dataset to enable fair comparison using consistent methodology.

## Background

- **Experiment 008**: Used "old-style" hybrid diffusion on Production data
- **Experiment 018**: TabDDPM-style diffusion on Ozel Rich achieved 87% baseline

This experiment applies TabDDPM-style to Production to verify the methodology works across different datasets.

## Dataset

- **Source**: `data/production/full.pt`
- **Samples**: 4,296 train / 1,074 test
- **Features**: 7 numerical + 35 categorical (117 total dimensions)
- **Target**: Quote amount (EUR)
- **Baseline R²**: 0.92

## TabDDPM-style Improvements

Same as Experiment 018:
1. Log-space operations for numerical stability
2. KL divergence loss (not cross-entropy)
3. Gumbel-softmax sampling
4. Proper posterior computation

## Training

```bash
cd src
uv run python train_production_tabddpm.py --epochs 1000 --device cuda
```

## Expected Outcomes

- Compare TabDDPM performance on Production vs Experiment 008 results
- Validate that TabDDPM methodology generalizes to different datasets
- Enable fair comparison: both Production and Ozel Rich using TabDDPM

## Files

- `src/train_production_tabddpm.py` - Training script
- `checkpoints/experiment_019/` - Model checkpoints
