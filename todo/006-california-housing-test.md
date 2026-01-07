# Test on California Housing Dataset

**Source:** Experiment 001 follow-up

## Task

Test Gaussian diffusion on a larger dataset (California Housing) to verify scalability.

## Notes

- California Housing: 20,640 samples, 9 features (all numerical)
- Much larger than Iris (150 samples)
- Will test if the approach scales
- Already supported in `src/train.py`

## Command

```bash
uv run python src/train.py --dataset california --epochs 2000 --device cuda
```

## Status
- [ ] Not started
