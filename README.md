# CAD Prediction Backend (FastAPI)

FastAPI backend serving a trained XGBoost classifier for Coronary Artery Disease (CAD) risk prediction, built for the Multimodal AI Hackathon 2026 (Track A).

## Features
- **Strict 52-Feature Contract**: Evaluates exactly 52 patient features matching training metadata.
- **Categorical Encoding**: Preprocesses 18 categorical features using pre-fitted `LabelEncoder` objects.
- **Direct Probability Inference**: Outputs unmodified probabilities directly from `predict_proba()` mapped to `Normal` and `CAD`.
- **Standardized Response Envelope**: Uniform success (`{"success": true, "data": ...}`) and error (`{"success": false, "error": ...}`) format.
- **Prohibited Field Validation**: Rejects restricted clinical features (`Cath`, `LAD`, `LCX`, `RCA`) with HTTP 422 `VALIDATION_ERROR`.
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
│   │       └── artifacts/           # Serialized model artifacts
├── docs/
│   └── FEATURES.md                  # Detailed 52-feature contract specification
├── scripts/
│   └── inspect_artifacts.py         # Verification and self-check inspection script
├── tests/
│   └── test_api.py                  # Automated test suite (13 tests)
├── requirements.txt
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
pip install -r requirements.txt
```

### 2. Environment Setup
```bash
cp .env.example .env
```

### 3. Verify Model Artifacts
Run the inspection script to verify all 5 startup self-checks:
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

### 1. `GET /health`
Liveness and readiness check. Returns HTTP 503 if artifacts are unavailable.
```json
{
  "success": true,
  "data": {
    "status": "ok",
    "model_loaded": true
  }
}
```

### 2. `GET /model-info`
Provides model metadata and all 52 feature definitions for frontend form construction.
```json
{
  "success": true,
  "data": {
    "model_name": "XGBoost CAD classifier",
    "model_version": "1.0.0",
    "feature_count": 52,
    "features": [
      { "name": "Age", "type": "numeric", "allowed_values": null, "required": true },
      { "name": "Sex", "type": "categorical", "allowed_values": ["Fmale", "Male"], "required": true }
    ],
    "classes": ["Normal", "CAD"]
  }
}
```

### 3. `POST /predict`
Executes CAD inference on the 52 patient features.

**Request:**
```json
{
  "features": {
    "Age": 58.0,
    "Sex": "Male",
    "...": "..."
  }
}
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "prediction": "Normal",
    "probabilities": {
      "CAD": 0.2538,
      "Normal": 0.7462
    },
    "model_version": "1.0.0",
    "disclaimer": "For decision support and educational purposes only. Not a substitute for formal diagnostic imaging."
  }
}
```

**Validation Error Response (422 Unprocessable Content):**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Input feature validation failed",
    "fields": {
      "missing_features": ["Missing required feature: Age"],
      "LAD": ["Excluded feature 'LAD' must never be provided as prediction input"]
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
