# Feature Contract Specification (52 Features)

Generated from `MODEL/CAD_XGBoost_Metadata.pkl` and `MODEL/CAD_Categorical_Encoders.pkl`.

## Overview
- **Total Features Expected by Model**: 52
- **Feature Order**: Fixed. Must match the exact index order below.
- **Categorical Features**: 18 features (encoded via `LabelEncoder`).
- **Numeric Features**: 34 features (float conversion; range limits are `NEEDS_CONFIRMATION`).
- **Excluded Features (Never Accepted as Prediction Inputs)**: `Cath`, `LAD`, `LCX`, `RCA`, `Exertional CP`, `LowTH Ang`, `CHF`.

---

## Complete Feature Contract Table

| # | Feature Name | Data Type | Feature Type | Allowed Values / Range | Required |
| :-: | :--- | :--- | :--- | :--- | :-: |
| 1 | `Age` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 2 | `Weight` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 3 | `Length` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 4 | `Sex` | string | Categorical | `['Fmale', 'Male']` | Yes |
| 5 | `BMI` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 6 | `DM` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 7 | `HTN` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 8 | `Current Smoker` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 9 | `EX-Smoker` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 10 | `FH` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 11 | `Obesity` | string | Categorical | `['N', 'Y']` | Yes |
| 12 | `CRF` | string | Categorical | `['N', 'Y']` | Yes |
| 13 | `CVA` | string | Categorical | `['N', 'Y']` | Yes |
| 14 | `Airway disease` | string | Categorical | `['N', 'Y']` | Yes |
| 15 | `Thyroid Disease` | string | Categorical | `['N', 'Y']` | Yes |
| 16 | `DLP` | string | Categorical | `['N', 'Y']` | Yes |
| 17 | `BP` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 18 | `PR` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 19 | `Edema` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 20 | `Weak Peripheral Pulse` | string | Categorical | `['N', 'Y']` | Yes |
| 21 | `Lung rales` | string | Categorical | `['N', 'Y']` | Yes |
| 22 | `Systolic Murmur` | string | Categorical | `['N', 'Y']` | Yes |
| 23 | `Diastolic Murmur` | string | Categorical | `['N', 'Y']` | Yes |
| 24 | `Typical Chest Pain` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 25 | `Dyspnea` | string | Categorical | `['N', 'Y']` | Yes |
| 26 | `Function Class` | float | Numeric | `NEEDS_CONFIRMATION` (1 - 4) | Yes |
| 27 | `Atypical` | string | Categorical | `['N', 'Y']` | Yes |
| 28 | `Nonanginal` | string | Categorical | `['N', 'Y']` | Yes |
| 29 | `Q Wave` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 30 | `St Elevation` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 31 | `St Depression` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 32 | `Tinversion` | float | Numeric | `NEEDS_CONFIRMATION` (0 or 1) | Yes |
| 33 | `LVH` | string | Categorical | `['N', 'Y']` | Yes |
| 34 | `Poor R Progression` | string | Categorical | `['N', 'Y']` | Yes |
| 35 | `BBB` | string | Categorical | `['LBBB', 'N', 'RBBB']` | Yes |
| 36 | `FBS` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 37 | `CR` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 38 | `TG` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 39 | `LDL` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 40 | `HDL` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 41 | `BUN` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 42 | `ESR` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 43 | `HB` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 44 | `K` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 45 | `Na` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 46 | `WBC` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 47 | `Lymph` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 48 | `Neut` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 49 | `PLT` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 50 | `EF-TTE` | float | Numeric | `NEEDS_CONFIRMATION` | Yes |
| 51 | `Region RWMA` | float | Numeric | `NEEDS_CONFIRMATION` (0 - 4) | Yes |
| 52 | `VHD` | string | Categorical | `['Moderate', 'N', 'Severe', 'mild']` | Yes |

---

## Target Mapping & Classes
- Class `0`: `Normal`
- Class `1`: `CAD`
