# Manufacturing Duration Dataset Analysis

**Date:** 2025-01-11
**Source:** `Teklif Maliyet SSEK data.xlsx` (from standart-muhendislik and ozel-muhendislik folders)

## Overview

This dataset contains manufacturing data for bolts/fasteners with the target being **machine duration for 100,000 units**. This is the dataset where the original AI engineer reported poor model performance, making it ideal for testing whether diffusion-based data augmentation can improve predictions.

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total Rows | 17,942 |
| Total Columns | 15 |
| Missing Values | 0% |
| Target Column | `Makine Süre (100.000 ADET) DK` |

## Target Variable

**`Makine Süre (100.000 ADET) DK`** - Machine duration in minutes for manufacturing 100,000 units

| Statistic | Value |
|-----------|-------|
| Min | 333 minutes |
| Max | 2,274 minutes |
| Mean | 1,008 minutes |
| Median | 914 minutes |
| Std Dev | 349 minutes |

### Distribution by Percentile
| Percentile | Value |
|------------|-------|
| 10% | 666 min |
| 25% | 741 min |
| 50% | 914 min |
| 75% | 1,178 min |
| 90% | 1,430 min |
| 95% | 1,668 min |

## Column Descriptions

### ID Columns (Do Not Use as Features)
| Column | Description | Unique Values |
|--------|-------------|---------------|
| Malzeme | Material code | 16,045 |
| Malzeme Tanımı | Material description (e.g., "M8x25 DIN 933 10.9 SSEK") | 11,832 |

### Cost Columns (DATA LEAKAGE - Do Not Use)
These columns are highly correlated with the target (0.44-0.86) because costs are likely **calculated from** machine time. Using them would be circular/cheating.

| Column | Correlation with Target |
|--------|------------------------|
| Endirekt Amortisman 10.000 ADET (TRY) | 0.86 |
| Direkt İşicilik 10.000 ADET (TRY) | 0.76 |
| GUG 10.000 ADET (TRY) | 0.74 |
| Enerji 10.000 ADET (TRY) | 0.63 |
| Direkt Amortisman 10.000 ADET (TRY) | 0.61 |
| Endirekt İşçilik 10.000 ADET (TRY) | 0.44 |

### Recommended Features

#### Numeric Features (Extracted from Malzeme Tanımı)
| Feature | Description | Correlation | Range |
|---------|-------------|-------------|-------|
| Çap | Diameter in mm (e.g., M8 → 8) | 0.69 | 4-30 mm |
| Boy | Length in mm | 0.64 | 1-300 mm |

**Extraction Success Rate:** 97.8% of rows

#### Categorical Features
| Feature | Description | Unique Values |
|---------|-------------|---------------|
| Sartname | DIN/ISO specification (e.g., "DIN 933", "OZEL") | 191 |
| Kısa tanım | Machine recipe code (e.g., "JBF 19B5SL", "SP57") | 136 |
| ÜY | Production location | 2 |
| İş Yeri | Workplace code | 144 |
| Msf.yeri | Cost center | 144 |

## Key Findings

### 1. Target Varies by Product Characteristics
**By Diameter (Anma ölçüsü):**
| Size | Mean Duration | Count |
|------|---------------|-------|
| M6 | 704 min | 2,112 |
| M8 | 791 min | 3,407 |
| M10 | 935 min | 3,018 |
| M12 | 1,004 min | 2,619 |
| M16 | 1,287 min | 1,954 |
| M24 | 1,584 min | 419 |

Larger bolts take ~2.2x longer to manufacture.

**By Specification (Sartname):**
| Specification | Mean Duration | Count |
|---------------|---------------|-------|
| ROTİL | 1,480 min | 546 |
| DIN 931-ISO | 1,216 min | 962 |
| DIN 960 | 1,094 min | 523 |
| DIN 912-ISO | 1,046 min | 1,958 |
| DIN 7984 | 866 min | 345 |

### 2. Machine Recipe is Important
The `Kısa tanım` column contains machine recipe codes (JBF, NEB, HBP, SP series) that likely encode the manufacturing process steps. Top recipes:
- JBF 19B5SL: 751 occurrences
- NEB524: 705 occurrences
- JBP 24B5S: 592 occurrences

### 3. Data Leakage Risk
The cost columns should NOT be used because:
- They have suspiciously high correlation (0.44-0.86) with target
- Cost per minute ratio is relatively stable (CV = 35.5%)
- Costs are likely calculated as: `cost = rate × machine_time`

## Recommended Feature Set for Experiments

### Numeric Features (2)
1. **Çap** - Diameter extracted from material description
2. **Boy** - Length extracted from material description

### Categorical Features (4)
1. **Sartname** - Specification (191 unique, but can group rare ones)
2. **Kısa tanım** - Machine recipe (136 unique)
3. **ÜY** - Production location (2 unique)
4. **İş Yeri** - Workplace (144 unique, consider grouping)

## Why This Dataset is Good for Thesis

1. **Real Production Data** - Actual manufacturing data, not synthetic
2. **Clear Business Problem** - Predicting manufacturing duration for quotes
3. **Moderate Feature-Target Correlation** - ~0.65 correlation leaves room for improvement
4. **Mixed Feature Types** - Both numeric and categorical features
5. **Reported Poor Baseline** - AI engineer reported unsatisfactory model performance
6. **Large Enough for Augmentation** - 18k rows, but augmentation might still help

## Data Preparation Notes

### Dimension Extraction Regex
```python
# Pattern 1: M8x1,00x50 (with pitch) -> cap=8, boy=50
match = re.match(r'[MØ]?(\d+(?:[.,]\d+)?)[xX]\d+[.,]\d+[xX](\d+(?:[.,]\d+)?)', text)

# Pattern 2: M8x25 (standard) -> cap=8, boy=25
match = re.match(r'[MØ]?(\d+(?:[.,]\d+)?)[xX](\d+(?:[.,]\d+)?)', text)
```

### Problematic Rows
- 3 rows have malformed `Malzeme Tanımı` (e.g., "M16x12015071" missing space)
- 2,633 rows have pitch in the middle (e.g., "M8x1,00x50")
- All handled by the extraction logic

## File Locations

**Source Data:**
- `C:\Users\ydran\Downloads\standart-muhendislik\standart-muhendislik\Genel_Sartname\Teklif Maliyet SSEK data.xlsx`
- `C:\Users\ydran\Downloads\ozel-muhendislik\ozel-muhendislik\Kopya Teklif Maliyet SSEK data.xlsx`
- (These are identical files - "Kopya" means "Copy")

**Note:** Data should be copied to `data/manufacturing/` for experiments (gitignored).
