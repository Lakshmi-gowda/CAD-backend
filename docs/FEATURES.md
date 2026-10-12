# Overall CAD Feature Contract (52 Features)

Generated from `backend/app/ml/artifacts/CAD_XGBoost_Metadata.pkl` and
`backend/app/ml/artifacts/CAD_Categorical_Encoders.pkl`.

## Overview
- **Total Features Expected by Model**: 52
- **Feature Order**: Fixed. Must match the exact index order below.
- **Categorical Features**: 18 features (encoded via `LabelEncoder`).
- **Numeric Features**: 34 features (float conversion; range limits are `NEEDS_CONFIRMATION`).
- The overall CAD model does not accept `Cath`, `CHF`, `Exertional CP`, or `LowTH Ang`.
- `LAD`, `LCX`, and `RCA` are target labels and are never accepted as prediction inputs.

## Artery Model Input Contract

The saved LAD, LCX, and RCA pipelines each require **56 input columns**, in the
order stored in `backend/app/ml/artifacts/artery_models/metadata.json`. They use
the 52 CAD features listed below plus these four additional required inputs:

| Feature | Type | Saved training categories / behavior |
| :--- | :--- | :--- |
| `CHF` | Categorical | `N`, `Y` |
| `Exertional CP` | Categorical | Training category: `N`; the saved `OneHotEncoder` uses `handle_unknown="ignore"` |
| `LowTH Ang` | Categorical | `N`, `Y` |
| `Cath` | Categorical | `CAD`, `Normal` |

All three artery pipelines have the same verified contract: 34 numeric columns
pass through unchanged after finite-number validation, and 22 categorical
columns are transformed by the fitted `OneHotEncoder`. The pipeline input
feature count is 56 (the classifier receives 80 columns after one-hot
encoding). Feature order is determined by `metadata.json`; callers must send
all 56 named values. The API rejects the target fields `LAD`, `LCX`, and `RCA`.

The existing Lovable frontend source is not present in this repository. Its
assessment form must explicitly collect or map the four additional fields
above before `POST /predict` can return artery results. The backend does not
infer or default these values. In particular, `Cath` must be supplied as its
own existing patient-data value; it is not populated from an overall CAD
prediction or an artery prediction.

The artery artifacts declare `Normal` as the negative class and `Stenotic` as
the positive class. Each pipeline's `classes_` is `[0, 1]`; the mapping comes
from that saved metadata, and inference locates the `Stenotic` class in
`classes_` instead of assuming a fixed probability-array position. Artery
scores are not described as clinically calibrated probabilities.

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
