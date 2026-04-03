# Add Membership Inference Attack Test

**Priority:** HIGH
**Effort:** 1-2 days
**Addresses:** "No formal privacy analysis"

## Problem

We claim "privacy-preserving" but haven't tested if the synthetic data leaks information about training data. A membership inference attack tests whether an attacker can determine if a specific record was in the training set.

## What is Membership Inference?

Given:
- A trained generative model (or its outputs)
- A target record

Question: Was this record in the training data?

If the attack succeeds often, the model is "memorizing" training data → privacy risk.

## Simple Approach

### Shadow Model Attack (Simplified)

1. Split original data: Train (used for diffusion) / Holdout (not used)
2. Generate synthetic data from diffusion model
3. For each record in Train and Holdout:
   - Compute "similarity" to nearest synthetic sample
   - Train a classifier: "member" vs "non-member"
4. Evaluate: If classifier accuracy ≈ 50%, model doesn't leak membership

### Metrics
- **Attack Accuracy**: Should be close to 50% (random guess)
- **Attack AUC**: Should be close to 0.5
- **True Positive Rate at low FPR**: Should be low

## Implementation Sketch

```python
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def membership_inference_test(train_data, holdout_data, synthetic_data):
    """Test if synthetic data leaks membership information."""

    # Fit nearest neighbor on synthetic data
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(synthetic_data)

    # Compute distances for train (members) and holdout (non-members)
    train_distances, _ = nn.kneighbors(train_data)
    holdout_distances, _ = nn.kneighbors(holdout_data)

    # Create labels
    X = np.concatenate([train_distances, holdout_distances])
    y = np.concatenate([np.ones(len(train_data)), np.zeros(len(holdout_data))])

    # Train attack classifier
    clf = LogisticRegression()
    clf.fit(X, y)

    # Evaluate
    y_pred = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)

    return {
        'attack_auc': auc,
        'privacy_safe': auc < 0.6  # Close to 0.5 is good
    }
```

## Expected Result

- Diffusion AUC ≈ 0.5-0.55 (good, doesn't leak)
- If AUC > 0.7, there's a privacy concern

## Apply To

- Experiment 011 or 013 (pick one with most data)

## Tasks

- [ ] Implement membership inference attack
- [ ] Create holdout set from original data (not used in training)
- [ ] Run attack on diffusion-generated data
- [ ] Run attack on SMOGN-generated data (for comparison)
- [ ] Document results and interpretation
- [ ] Add to thesis privacy section

## References

- Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (2017)
- Stadler et al., "Synthetic Data - Anonymisation Groundhog Day" (2022)
