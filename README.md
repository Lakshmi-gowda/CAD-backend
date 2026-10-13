# CAD Prediction Backend (FastAPI)

FastAPI backend serving the existing XGBoost classifier for overall CAD and
three saved pipeline models for LAD, LCX, and RCA artery predictions.

## Features
- **Separate Verified Feature Contracts**: The overall CAD model retains its original 52-feature contract; each artery pipeline uses its retrained 55-feature contract.
- **Categorical Encoding**: Preprocesses 18 categorical features using pre-fitted `LabelEncoder` objects.
- **Saved Artery Preprocessing**: Uses each artery pipeline's fitted `ColumnTransformer` and `OneHotEncoder`.
- **Direct Model Inference**: Uses each model's `predict()` and `predict_proba()` methods and maps probabilities in the trained model's class order.
- **Standardized Response Envelope**: Uniform success (`{"success": true, "data": ...}`) and error (`{"success": false, "error": ...}`) format.
- **Forbidden Input Validation**: Excludes `Cath`, `LAD`, `LCX`, and `RCA` from artery inputs.
- **Vercel Serverless Ready**: Compatible with Vercel serverless deployment (`@vercel/python`) with cold-start warmup.

---

## Directory Structure

```text
CAD backend/
├── api/
│   └── index.py                     # Vercel serverless entrypoint
├── backend/
│   ├── app/
│   │   ├── main.py                  # App factory, CORS, and exception handlers
│   │   ├── api/
│   │   │   ├── health.py            # GET /health
│   │   │   ├── model_info.py        # GET /model-info
│   │   │   └── predict.py           # POST /predict
│   │   ├── core/
│   │   │   ├── config.py            # Pydantic Settings
│   │   │   └── errors.py            # Error envelopes and exception handlers
│   │   ├── schemas/
│   │   │   ├── common.py            # Envelope and info schemas
│   │   │   └── predict.py           # Request and response schemas
│   │   ├── services/
│   │   │   ├── artifact_loader.py   # In-memory artifact registry with 5 self-checks
│   │   │   ├── preprocessing.py     # Feature validation, encoding, and ordering
│   │   │   └── model_service.py     # Inference execution
│   │   └── ml/
│   │       └── artifacts/
│   │           ├── CAD_XGBoost_*.pkl
│   │           └── artery_models/   # Three saved .joblib pipelines + metadata.json
├── docs/
│   └── FEATURES.md                  # Overall CAD and artery input contracts
├── scripts/
│   └── inspect_artifacts.py         # Verification and self-check inspection script
├── tests/
│   └── test_api.py                  # Automated API test suite
├── requirements.txt
├── requirements-dev.txt
├── .python-version
├── vercel.json
├── .env.example
└── .gitignore
```

---

## Getting Started

### 1. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```
Use the repository-pinned Python 3.12.7 runtime. The saved artery pipelines
were serialized with scikit-learn 1.6.1, and loading them with a different
scikit-learn version may fail; the backend reports that model as unavailable
instead of returning fallback output.

### 2. Environment Setup
```bash
cp .env.example .env
```

### 3. Verify Model Artifacts
Run the inspection script to verify the 5 existing CAD artifact checks and
the three artery pipeline contracts:
```bash
python3 scripts/inspect_artifacts.py
```

### 4. Run the Development Server
```bash
uvicorn backend.app.main:app --reload --port 8000
```
API Documentation is available at `http://localhost:8000/docs`.

### 5. Run Automated Tests
```bash
pytest tests/test_api.py -v
```

---

## API Endpoints

### 1. `GET /health/live`
Confirms the API process is responding; this does not indicate whether the model is loaded.

### 2. `GET /health`
Model readiness check. `model_status` reports whether the overall CAD model
and each artery model loaded. `status` is `degraded` if an artery model failed
to load; the overall CAD status remains separately visible.
```json
{
  "success": true,
  "data": {
    "status": "ok",
    "model_loaded": true,
    "model_status": {
      "CAD": { "loaded": true },
      "LAD": { "loaded": true },
      "LCX": { "loaded": true },
      "RCA": { "loaded": true }
    }
  }
}
```

### 3. `GET /model-info`
Provides the existing 52 overall CAD feature definitions, all 55 artery input
definitions, and per-model load status.
```json
{
  "success": true,
  "data": {
    "model_name": "XGBoost CAD classifier",
    "model_version": "1.0.0",
    "feature_count": 52,
    "artery_feature_count": 55,
    "model_status": {
      "CAD": { "loaded": true },
      "LAD": { "loaded": true },
      "LCX": { "loaded": true },
      "RCA": { "loaded": true }
    },
    "features": [
      { "name": "Age", "type": "numeric", "allowed_values": null, "required": true },
      { "name": "Sex", "type": "categorical", "allowed_values": ["Fmale", "Male"], "required": true }
    ],
    "classes": ["Normal", "CAD"]
  }
}
```

### 4. `POST /predict`
Runs overall CAD, all three artery predictions, or both independently. Supply
`cad_features` for the original CAD model's 52-feature contract and/or
`artery_features` for the artery pipelines' 55-feature contract. The backend
does not fill in missing values. The artery contract excludes `Cath`, `LAD`,
`LCX`, and `RCA`; these fields are never sent to an artery model.

For example, a CAD-only request is `{"cad_features": {...}}`, an artery-only
request is `{"artery_features": {...}}`, and a combined request supplies both
keys. The response includes `data.cad_prediction` only when CAD inputs were
provided and `data.artery_predictions.LAD`, `.LCX`, and `.RCA` only when artery
inputs were provided. Each artery result returns the predicted class,
class-labeled probabilities, and `stenotic_probability`. A failed requested
model returns an explicit error rather than a success-shaped response or
fallback score.

These artery outputs are model scores; clinical probability calibration has
not been evaluated in this integration. The existing Lovable frontend source
is not part of this repository. Its form/API integration should send separate
`cad_features` and/or `artery_features` maps and read the corresponding result
keys; the backend will not derive or default missing values.

No patient example or expected probability is shown because a verified source row
and its original training-notebook prediction are not present in this repository.
The automated inference test uses synthetic values only as an API smoke test, not
for clinical interpretation or prediction-parity validation.

**Validation Error Response (422 Unprocessable Content):**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Input feature validation failed",
    "fields": {
      "missing_features": ["Missing required feature: Age"],
      "LAD": ["Excluded feature 'LAD' must never be provided as artery model input"]
    }
  }
}
```

---

## Vercel Deployment

This backend is pre-configured for Vercel deployment:
- Entrypoint at [`api/index.py`](api/index.py)
- Configuration in [`vercel.json`](vercel.json)
- Ignored build artifacts in [`.vercelignore`](.vercelignore)

Deploy directly with Vercel CLI:
```bash
vercel --prod
```
Ensure that `CORS_ORIGINS` is configured in your Vercel Project Settings to match your frontend URL.

## Render Deployment

The service runs from the repository root with the existing ASGI app at
`backend.app.main:app`.

- **Build Command:** `pip install --only-binary=:all: -r requirements.txt`
- **Start Command:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- **Python version:** `.python-version` pins the service to Python 3.12.7. If
  configuring the runtime in Render instead, set `PYTHON_VERSION` to `3.12.7`.

Set these environment variables in Render (no committed `.env` file is needed):

| Variable | Value |
| --- | --- |
| `ENV` | `production` |
| `LOG_LEVEL` | `INFO` |
| `ARTIFACTS_DIR` | `backend/app/ml/artifacts` |
| `MAX_REQUEST_SIZE_BYTES` | `1048576` |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:8080,https://id-preview--4711e78b-8b24-4fd5-9d46-b7e957db5cb3.lovable.app` |

Render supplies `PORT` automatically. Use `/health/live` for process liveness
and `/health` for model readiness. A successful server start alone does not
mean the model artifacts loaded.

No third-party API keys or API base-URL environment variables are required.
After deployment, Render assigns the service URL; append `/health/live`,
`/health`, `/model-info`, `/predict`, or `/docs` to that URL to reach the
available endpoints. The `.env.example` file lists the application settings
and endpoint URL patterns.
