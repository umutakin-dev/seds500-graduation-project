# ML Efficiency Evaluation

**Source:** Experiment 001 follow-up, paper methodology

## Task

Implement ML efficiency evaluation - the standard metric used in TabDDPM/STaSy/TabSyn papers.

## What is ML Efficiency?

Train a classifier/regressor on synthetic data, test on real data. Compare performance to training on real data.

```
Real→Real:      Train on real, test on real (baseline)
Synthetic→Real: Train on synthetic, test on real (ML efficiency)
Augmented→Real: Train on real+synthetic, test on real (augmentation)
```

## Notes

- This directly answers "does synthetic data improve model accuracy?"
- Need to implement evaluation pipeline with sklearn classifiers
- Should test multiple ML models (LogisticRegression, RandomForest, XGBoost)
- Related to todo 001 (confusion matrix)

## Status
- [ ] Not started
