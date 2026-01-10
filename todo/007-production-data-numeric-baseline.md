# Experiment 003: Production Data - Numeric Baseline

## Objective
Test Gaussian diffusion on production data using only numeric features as a simplified baseline.

## Dataset
- Source: `6_Ay_Teklif_Dokumanlari_Sayfa2.xlsx` (processed)
- Rows: 5,533 quotations
- Target options:
  - `KAR MARJI` (Profit Margin) - 74% filled
  - `İlk Girilen Teklif Miktarı` (Initial Quote Amount) - 97% filled

## Approach
1. Extract only numeric columns (skip 74 categorical columns)
2. Parse Turkish number format ("123,00" → 123.0)
3. Handle missing values (drop rows or impute)
4. Train Gaussian diffusion on numeric features + target
5. Evaluate: Diffusion vs Noise augmentation

## Expected Numeric Features
- `Zorluk` (Difficulty): 1.2, 1.5
- `Dağıtım kanalı` (Distribution channel): 10, 20
- Various cost columns (after parsing)

## Limitations
- Ignores categorical features (74 columns)
- May lose important information
- Serves as baseline for comparison with full model

## Success Criteria
- Working pipeline for production data
- Baseline metrics for comparison
- Understanding of data quality issues

## Dependencies
- Create `src/prepare_production_data.py`
- Update training pipeline if needed

## Status
- [ ] Create data preparation script
- [ ] Extract and clean numeric features
- [ ] Train diffusion model
- [ ] Run evaluation
- [ ] Document findings
