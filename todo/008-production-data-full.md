# Experiment 004: Production Data - Full (with Multinomial Diffusion)

## Objective
Test hybrid diffusion (Gaussian + Multinomial) on production data with all features.

## Dataset
- Source: `6_Ay_Teklif_Dokumanlari_Sayfa2.xlsx` (processed)
- Rows: 5,533 quotations
- Features: ~30-40 selected (from 89 total)
- Targets:
  - `KAR MARJI` (Profit Margin)
  - `İlk Girilen Teklif Miktarı` (Initial Quote Amount)

## Approach
1. Feature selection: Pick most relevant ~30-40 features
2. Implement multinomial diffusion for categorical features
3. Create hybrid model (Gaussian for numeric, Multinomial for categorical)
4. Train on production data
5. Compare: Diffusion vs SMOTE vs Noise

## Key Categorical Features
- `Anma Ölçüsü` (Nominal Size): M6, M8, M10, M12, M16
- `Dayanım Sınıfı` (Strength Class): 8.8, 10.9, 12.9
- `Kaplama Tipi` (Coating Type): 26 types
- `Müşteri Numarası` (Customer): 104 unique
- Various process flags (VAR/YOK)

## Dependencies
- [x] Implement multinomial diffusion (todo/001) - DONE
- [x] Create hybrid diffusion model - DONE
- [x] Class-conditional diffusion (todo/009) - DONE
- [ ] Feature selection analysis
- [ ] Experiment 007 (numeric baseline) - recommended first

## Success Criteria
- Hybrid model handles mixed data types
- Compare with Experiment 007 baseline
- Demonstrate value of categorical features
- Outperform simple augmentation methods

## Status
- [x] Complete multinomial diffusion implementation
- [x] Hybrid diffusion model ready
- [ ] Feature selection analysis
- [ ] Train on production data
- [ ] Run evaluation for both targets
- [ ] Compare with baseline (Experiment 007)
- [ ] Document findings
