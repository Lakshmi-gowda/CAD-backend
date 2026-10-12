from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import app
from backend.app.services.artifact_loader import get_artifact_registry

client = TestClient(app)

@pytest.fixture(autouse=True)
def ensure_artifacts_loaded():
    registry = get_artifact_registry()
    if not registry.is_loaded:
        registry.load()

def get_synthetic_inference_features():
    """Builds a synthetic API smoke-test input, not a patient or parity fixture."""
    registry = get_artifact_registry()
    sample = {}
    for feat in registry.feature_names:
        if feat in registry.encoders:
            # Use the first valid categorical label
            sample[feat] = registry.encoders[feat].classes_[0]
        else:
            # Zeroes are synthetic placeholders, not source-row values.
            sample[feat] = 0.0
    return sample

# ==========================================
# 1. Health Endpoint Tests
# ==========================================
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["data"]["status"] == "ok"
    assert json_data["data"]["model_loaded"] is True

def test_liveness_endpoint():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["data"] == {"status": "ok"}

def test_artifact_path_is_resolved_from_project_root(monkeypatch, tmp_path):
    settings = Settings(ARTIFACTS_DIR="backend/app/ml/artifacts")
    monkeypatch.chdir(tmp_path)

    expected_path = Path(__file__).resolve().parents[1] / "backend/app/ml/artifacts"
    assert Path(settings.resolved_artifacts_dir()) == expected_path

# ==========================================
# 2. Model Info Endpoint Tests
# ==========================================
def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["model_name"] == "XGBoost CAD classifier"
    assert data["feature_count"] == 52
    assert len(data["features"]) == 52
    assert set(data["classes"]) == {"Normal", "CAD"}

    # Excluded features must not be in the list
    feature_names = [f["name"] for f in data["features"]]
    for excluded in ["Cath", "LAD", "LCX", "RCA"]:
        assert excluded not in feature_names

    # Check categorical features have allowed values
    sex_feature = next(f for f in data["features"] if f["name"] == "Sex")
    assert sex_feature["type"] == "categorical"
    assert set(sex_feature["allowed_values"]) == {"Fmale", "Male"}

# ==========================================
# 3. CORS Tests
# ==========================================
@pytest.fixture
def cors_client(monkeypatch):
    from backend.app import main as main_module

    settings = Settings(
        CORS_ORIGINS=(
            "http://localhost:8080, "
            "https://id-preview--4711e78b-8b24-4fd5-9d46-b7e957db5cb3.lovable.app"
        )
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    return TestClient(main_module.create_app())

@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8080",
        "https://id-preview--4711e78b-8b24-4fd5-9d46-b7e957db5cb3.lovable.app",
    ],
)
def test_cors_preflight_allows_configured_origins(cors_client, origin):
    requested_headers = "Content-Type, Authorization, X-Requested-With"
    response = cors_client.options(
        "/predict",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": requested_headers,
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
    assert "POST" in response.headers["access-control-allow-methods"]
    allowed_headers = {
        header.strip().lower()
        for header in response.headers["access-control-allow-headers"].split(",")
    }
    assert {header.lower() for header in requested_headers.split(",")} <= allowed_headers

def test_cors_preflight_rejects_unconfigured_origin(cors_client):
    response = cors_client.options(
        "/predict",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers

def test_cors_origins_are_parsed_as_comma_separated_values():
    settings = Settings(CORS_ORIGINS=" , https://one.example, , https://two.example ")

    assert settings.cors_origins_list == [
        "https://one.example",
        "https://two.example",
    ]

# ==========================================
# 4. Valid Predict Inference Tests
# ==========================================
def test_predict_success():
    sample = get_synthetic_inference_features()
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["prediction"] in ["Normal", "CAD"]
    probs = data["probabilities"]
    assert "CAD" in probs and "Normal" in probs
    assert 0.0 <= probs["CAD"] <= 1.0
    assert 0.0 <= probs["Normal"] <= 1.0
    assert abs((probs["CAD"] + probs["Normal"]) - 1.0) < 1e-4
    assert "disclaimer" in data
    assert "educational purposes only" in data["disclaimer"]

# ==========================================
# 5. Validation Error Tests
# ==========================================
def test_predict_missing_features():
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "missing_features" in json_data["error"]["fields"]
    # Check that missing features are listed
    missing = json_data["error"]["fields"]["missing_features"]
    assert len(missing) == 52

def test_predict_prohibited_key_lad():
    sample = get_synthetic_inference_features()
    sample["LAD"] = 1.0  # Prohibited key
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "LAD" in json_data["error"]["fields"]

def test_predict_prohibited_key_cath():
    sample = get_synthetic_inference_features()
    sample["Cath"] = 0
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Cath" in json_data["error"]["fields"]

def test_predict_unknown_key():
    sample = get_synthetic_inference_features()
    sample["unknown_custom_field"] = "foo"
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "unknown_custom_field" in json_data["error"]["fields"]

def test_predict_invalid_categorical_value():
    sample = get_synthetic_inference_features()
    sample["Sex"] = "UnknownGender"
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Sex" in json_data["error"]["fields"]
    assert "Allowed values" in json_data["error"]["fields"]["Sex"][0]

def test_predict_invalid_numeric_nan():
    sample = get_synthetic_inference_features()
    sample["Age"] = "NaN"  # String representation of NaN
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Age" in json_data["error"]["fields"]

def test_predict_invalid_numeric_boolean():
    sample = get_synthetic_inference_features()
    sample["Age"] = True  # Boolean should be rejected in numeric field
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Age" in json_data["error"]["fields"]

def test_predict_invalid_numeric_string():
    sample = get_synthetic_inference_features()
    sample["Age"] = "not_a_number"
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Age" in json_data["error"]["fields"]

def test_predict_malformed_json():
    response = client.post(
        "/predict",
        content="malformed{json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in [400, 422]
    json_data = response.json()
    assert json_data["success"] is False

def test_request_body_size_limit():
    from backend.app.core.config import get_settings

    limit = get_settings().MAX_REQUEST_SIZE_BYTES
    response = client.post(
        "/predict",
        content=b"x" * (limit + 1),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
