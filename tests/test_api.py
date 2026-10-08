import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.artifact_loader import get_artifact_registry

client = TestClient(app)

@pytest.fixture(autouse=True)
def ensure_artifacts_loaded():
    registry = get_artifact_registry()
    if not registry.is_loaded:
        registry.load()

def get_valid_sample_features():
    """Generates a valid set of all 52 features for sanity testing."""
    registry = get_artifact_registry()
    sample = {}
    for feat in registry.feature_names:
        if feat in registry.encoders:
            # Use the first valid categorical label
            sample[feat] = registry.encoders[feat].classes_[0]
        else:
            # Provide standard default numeric float
            sample[feat] = 0.0
    # Set typical clinical values for key numerics
    sample["Age"] = 58.0
    sample["Weight"] = 72.0
    sample["Length"] = 170.0
    sample["BMI"] = 24.9
    sample["BP"] = 120.0
    sample["PR"] = 75.0
    sample["FBS"] = 95.0
    sample["CR"] = 0.9
    sample["TG"] = 150.0
    sample["LDL"] = 110.0
    sample["HDL"] = 45.0
    sample["EF-TTE"] = 55.0
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
# 3. CORS Preflight Tests
# ==========================================
def test_cors_preflight():
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

# ==========================================
# 4. Valid Predict Inference Tests
# ==========================================
def test_predict_success():
    sample = get_valid_sample_features()
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
    # Only provide 2 features
    response = client.post("/predict", json={"features": {"Age": 60.0, "Sex": "Male"}})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "missing_features" in json_data["error"]["fields"]
    # Check that missing features are listed
    missing = json_data["error"]["fields"]["missing_features"]
    assert len(missing) == 50

def test_predict_prohibited_key_lad():
    sample = get_valid_sample_features()
    sample["LAD"] = 1.0  # Prohibited key
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "LAD" in json_data["error"]["fields"]

def test_predict_prohibited_key_cath():
    sample = get_valid_sample_features()
    sample["Cath"] = 0
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Cath" in json_data["error"]["fields"]

def test_predict_unknown_key():
    sample = get_valid_sample_features()
    sample["unknown_custom_field"] = "foo"
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "unknown_custom_field" in json_data["error"]["fields"]

def test_predict_invalid_categorical_value():
    sample = get_valid_sample_features()
    sample["Sex"] = "UnknownGender"
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Sex" in json_data["error"]["fields"]
    assert "Allowed values" in json_data["error"]["fields"]["Sex"][0]

def test_predict_invalid_numeric_nan():
    sample = get_valid_sample_features()
    sample["Age"] = "NaN"  # String representation of NaN
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Age" in json_data["error"]["fields"]

def test_predict_invalid_numeric_boolean():
    sample = get_valid_sample_features()
    sample["Age"] = True  # Boolean should be rejected in numeric field
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Age" in json_data["error"]["fields"]

def test_predict_invalid_numeric_string():
    sample = get_valid_sample_features()
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
