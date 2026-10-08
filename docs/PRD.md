# CAD Backend PRD Summary

This document references the full PRD specification located at `PRD/CAD_Backend_PRD.pdf`.

## Core Requirements Summary
- **Architecture**: FastAPI REST backend exposing XGBoost CAD model for React/Vite frontend.
- **Model Inputs**: Exactly 52 features. Excluded features (`Cath`, `LAD`, `LCX`, `RCA`) must never be accepted.
- **Inference Output**: Probabilities directly from `predict_proba()` mapped to `Normal` (class 0) and `CAD` (class 1).
- **Envelope Standard**: Uniform `{"success": true, "data": ...}` and `{"success": false, "error": ...}` envelopes.
- **Startup Verification**: 5 self-checks executed on startup. Fails loudly with HTTP 503 if artifacts fail to load.
- **Deployment**: Serverless-ready on Vercel (`api/index.py`, `vercel.json`).
