# Experiment: Fill the Feature Gap (10-15 features)

**Priority:** HIGH (Easy win)
**Effort:** 1 day
**Addresses:** "Where exactly does SMOGN start failing?"

## Problem

Current experiments jump from 5 features (Exp 012) to 29 features (Exp 013). We need an experiment in the 10-15 feature range to show the transition point.

## Approach

Use the Ozel dataset but select a subset of features (~12-15).

### Feature Selection Options

From Ozel rich (29 features), select:
1. Target: `unit_cost`
2. Core features: `quantity`, `material_*`, `process_*` (pick ~12)

OR use California Housing (8 features) as a public dataset alternative.

## Expected Outcome

- If SMOGN works: Threshold is somewhere between 15-29 features
- If SMOGN fails: Threshold is below 15 features

Either result is informative and fills the gap.

## Experiment Number

Experiment 014

## Tasks

- [ ] Select 12-15 features from Ozel dataset
- [ ] Prepare data (train/test split)
- [ ] Train diffusion model
- [ ] Generate synthetic data
- [ ] Apply SMOGN
- [ ] Evaluate both on real test data
- [ ] Document results
