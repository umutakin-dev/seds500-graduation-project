# Production Data Experiment

**Source:** Real use case
**Depends on:** Previous experiments validating the approach

## Task

Test diffusion augmentation on the actual production efficiency dataset from the plant.

## Background

This is the dataset our ML team struggled with:
- Data was limited
- Classic augmentation (SMOTE) did not improve model accuracy
- This experiment will answer: can diffusion do better?

## Dataset Info

- **Source:** Internal production data (Excel file)
- **Domain:** Manufacturing production efficiency
- **Details:** TBD (need to analyze the Excel file)

## Data Preparation

- [ ] Analyze Excel file structure
- [ ] Identify feature types (numerical, categorical)
- [ ] Check for class imbalance
- [ ] Obfuscate sensitive columns if needed
- [ ] Create train/test split

## Experiment Plan

1. Understand the data and the original ML task
2. Establish baseline (Real -> Real)
3. Apply SMOTE (reproduce "didn't work" finding)
4. Apply diffusion augmentation
5. Compare results
6. Document findings

## Status
- [ ] Not started - need to analyze Excel file first
