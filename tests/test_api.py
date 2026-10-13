from pathlib import Path

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import app
from backend.app.services.artifact_loader import (
    ARTERY_TARGETS,
    ArtifactRegistry,
    get_artifact_registry,
)
from backend.app.services.model_service import predict_arteries
from backend.app.services.preprocessing import validate_and_preprocess

client = TestClient(app)

@pytest.fixture(autouse=True)
def ensure_artifacts_loaded():
    registry = get_artifact_registry()
    if not registry.is_loaded:
        registry.load()

def get_synthetic_artery_features():
    """Builds a synthetic API smoke-test input, not a patient or parity fixture."""
    registry = get_artifact_registry()
    sample = {}
    for feat in registry.artery_feature_names:
        if feat in registry.artery_categorical_features:
            sample[feat] = registry.artery_category_values[feat][0]
        else:
            # Zeroes are synthetic placeholders, not source-row values.
            sample[feat] = 0.0
    return sample


def get_synthetic_cad_features():
    registry = get_artifact_registry()
    sample = {}
    for feature in registry.feature_names:
        if feature in registry.categorical_features:
            sample[feature] = list(registry.encoders[feature].classes_)[0]
        else:
            sample[feature] = 0.0
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
    assert set(json_data["data"]["model_status"]) == {"CAD", *ARTERY_TARGETS}
    assert all(
        status["loaded"] for status in json_data["data"]["model_status"].values()
    )

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
    assert data["artery_feature_count"] == 55
    assert len(data["artery_features"]) == 55
    assert set(data["model_status"]) == {"CAD", *ARTERY_TARGETS}
    assert set(data["classes"]) == {"Normal", "CAD"}
    cad_feature_names = [feature["name"] for feature in data["features"]]
    artery_feature_names = [feature["name"] for feature in data["artery_features"]]
    assert (
        [name for name in artery_feature_names if name in cad_feature_names]
        == cad_feature_names
    )
    assert set(artery_feature_names) - set(cad_feature_names) == {
        "CHF", "Exertional CP", "LowTH Ang"
    }
    artery_feature_info = {
        feature["name"]: feature for feature in data["artery_features"]
    }
    assert set(artery_feature_info["CHF"]["allowed_values"]) == {"N", "Y"}
    assert artery_feature_info["Exertional CP"]["allowed_values"] == ["N"]
    assert set(artery_feature_info["LowTH Ang"]["allowed_values"]) == {"N", "Y"}
    # Forbidden fields must not appear in either model's input contract.
    feature_names = cad_feature_names + artery_feature_names
    for excluded in ["Cath", "LAD", "LCX", "RCA"]:
        assert excluded not in feature_names

    # Check categorical features have allowed values
    sex_feature = next(f for f in data["features"] if f["name"] == "Sex")
    assert sex_feature["type"] == "categorical"
    assert set(sex_feature["allowed_values"]) == {"Fmale", "Male"}


def test_all_artery_artifacts_load_with_verified_contract():
    registry = get_artifact_registry()
    artery_dir = (
        Path(__file__).resolve().parents[1]
        / "backend/app/ml/artifacts/artery_models"
    )

    assert (artery_dir / "metadata.json").is_file()
    assert len(registry.artery_feature_names) == 55
    assert registry.artery_metadata["feature_columns"] == registry.artery_feature_names
    assert registry.artery_metadata["feature_count"] == 55
    assert set(registry.artery_metadata["excluded_input_columns"]) == {
        "Cath", *ARTERY_TARGETS
    }
    assert registry.artery_metadata["targets"] == list(ARTERY_TARGETS)
    assert not set(("Cath", *ARTERY_TARGETS)) & set(registry.artery_feature_names)
    for artery in ARTERY_TARGETS:
        assert (artery_dir / f"{artery}_model.joblib").is_file()
        assert registry.artery_model_status[artery]["loaded"] is True
        model = registry.artery_models[artery]
        assert model.n_features_in_ == 55
        assert list(model.feature_names_in_) == registry.artery_feature_names
        assert list(model.classes_) == ["Normal", "Stenotic"]
        assert not set(ARTERY_TARGETS) & set(model.feature_names_in_)
        assert "Cath" not in model.feature_names_in_
        preprocessing = model.named_steps["preprocessor"]
        transformers = {
            name: (transformer, list(columns))
            for name, transformer, columns in preprocessing.transformers_
        }
        assert len(transformers["numeric"][1]) == 34
        assert len(transformers["categorical"][1]) == 21
        encoder = transformers["categorical"][0].named_steps["onehot"]
        assert encoder.handle_unknown == "ignore"
        assert registry.artery_category_values["Exertional CP"] == ["N"]


def test_all_artery_pipelines_accept_their_saved_55_feature_vector():
    registry = get_artifact_registry()
    synthetic_features = get_synthetic_artery_features()
    input_frame = pd.DataFrame(
        [[
            synthetic_features[feature]
            for feature in registry.artery_feature_names
        ]],
        columns=registry.artery_feature_names,
    )

    for artery in ARTERY_TARGETS:
        result = registry.artery_models[artery].predict_proba(input_frame)
        assert len(result) == 1
        assert len(result[0]) == len(registry.artery_models[artery].classes_)


def test_stenotic_probability_uses_class_value_not_probability_position():
    """Use controlled synthetic outputs to test class-index mapping only."""
    class ReversedClassModel:
        classes_ = ["Stenotic", "Normal"]

        @staticmethod
        def predict_proba(_features):
            return [[0.8, 0.2]]

        @staticmethod
        def predict(_features):
            return ["Stenotic"]

    registry = ArtifactRegistry()
    for artery in ARTERY_TARGETS:
        registry.artery_models[artery] = ReversedClassModel()
        registry.artery_model_status[artery] = {"loaded": True, "error": None}

    results = predict_arteries(pd.DataFrame([{}]), registry)

    assert results["LAD"].stenotic_probability == 0.8
    assert results["LAD"].predicted_class == "Stenotic"

# ==========================================
# 3. CORS Tests
# ==========================================
@pytest.fixture
def cors_client(monkeypatch):
    from backend.app import main as main_module

    settings = Settings(
        CORS_ORIGINS=(
            "http://localhost:5173,"
            "http://127.0.0.1:5173,"
            "http://localhost:5174,"
            "http://127.0.0.1:5174,"
            "http://localhost:8080,"
            "https://id-preview--4711e78b-8b24-4fd5-9d46-b7e957db5cb3.lovable.app"
        )
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    return TestClient(main_module.create_app())

@pytest.mark.parametrize(
    ("origin", "method"),
    [
        (origin, method)
        for origin in [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:8080",
            "https://id-preview--4711e78b-8b24-4fd5-9d46-b7e957db5cb3.lovable.app",
        ]
        for method in ["GET", "POST"]
    ],
)
def test_cors_preflight_allows_configured_origins(cors_client, origin, method):
    requested_headers = "Content-Type, Authorization, X-Requested-With"
    response = cors_client.options(
        "/predict",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": requested_headers,
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
    assert method in response.headers["access-control-allow-methods"]
    assert "*" not in response.headers["access-control-allow-origin"]
    assert "access-control-allow-credentials" not in response.headers
    allowed_headers = {
        header.strip().lower()
        for header in response.headers["access-control-allow-headers"].split(",")
    }
    requested_header_names = {
        header.strip().lower() for header in requested_headers.split(",")
    }
    assert requested_header_names <= allowed_headers

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

def test_cors_origins_default_to_local_and_existing_frontend_origins(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    settings = Settings(_env_file=None)

    assert settings.cors_origins_list == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8080",
        "https://id-preview--4711e78b-8b24-4fd5-9d46-b7e957db5cb3.lovable.app",
    ]

def test_cors_origins_read_from_environment(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://frontend.example")
    settings = Settings(_env_file=None)

    assert settings.cors_origins_list == ["https://frontend.example"]

# ==========================================
# 4. Valid Predict Inference Tests
# ==========================================
def test_predict_success():
    registry = get_artifact_registry()
    cad_sample = get_synthetic_cad_features()
    artery_sample = get_synthetic_artery_features()
    response = client.post(
        "/predict",
        json={
            "cad_features": cad_sample,
            "artery_features": artery_sample,
        },
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    data = json_data["data"]
    cad_prediction = data["cad_prediction"]
    assert cad_prediction["prediction"] in ["Normal", "CAD"]
    probs = cad_prediction["probabilities"]
    assert "CAD" in probs and "Normal" in probs
    assert 0.0 <= probs["CAD"] <= 1.0
    assert 0.0 <= probs["Normal"] <= 1.0
    assert abs((probs["CAD"] + probs["Normal"]) - 1.0) < 1e-4

    cad_model = registry.models["cad"]
    cad_input = validate_and_preprocess(cad_sample, registry)
    cad_labels = {
        class_value: label
        for label, class_value in registry.target_mapping.items()
    }
    assert cad_prediction["prediction"] == cad_labels[cad_model.predict(cad_input)[0]]
    expected_cad_probabilities = {
        cad_labels[class_value]: float(probability)
        for class_value, probability in zip(
            cad_model.classes_, cad_model.predict_proba(cad_input)[0]
        )
    }
    assert cad_prediction["probabilities"] == pytest.approx(expected_cad_probabilities)

    artery_predictions = data["artery_predictions"]
    assert set(artery_predictions) == set(ARTERY_TARGETS)
    artery_input = pd.DataFrame(
        [[artery_sample[feature] for feature in registry.artery_feature_names]],
        columns=registry.artery_feature_names,
    )
    for artery in ARTERY_TARGETS:
        prediction = artery_predictions[artery]
        assert set(prediction) == {
            "stenotic_probability", "predicted_class", "probabilities"
        }
        assert isinstance(prediction["stenotic_probability"], (int, float))
        assert 0.0 <= prediction["stenotic_probability"] <= 1.0
        assert prediction["predicted_class"] in {"Normal", "Stenotic"}
        assert set(prediction["probabilities"]) == {"Normal", "Stenotic"}
        assert prediction["stenotic_probability"] == prediction["probabilities"]["Stenotic"]
        assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
        model = registry.artery_models[artery]
        assert prediction["predicted_class"] == model.predict(artery_input)[0]
        expected_probabilities = {
            label: float(probability)
            for label, probability in zip(
                model.classes_, model.predict_proba(artery_input)[0]
            )
        }
        assert prediction["probabilities"] == pytest.approx(expected_probabilities)
    assert "disclaimer" in cad_prediction


def test_cad_prediction_runs_without_artery_features():
    response = client.post(
        "/predict", json={"cad_features": get_synthetic_cad_features()}
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert set(data) == {"cad_prediction"}
    assert data["cad_prediction"]["prediction"] in {"Normal", "CAD"}


def test_artery_predictions_run_without_cad_features():
    response = client.post(
        "/predict", json={"artery_features": get_synthetic_artery_features()}
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert set(data) == {"artery_predictions"}
    assert set(data["artery_predictions"]) == set(ARTERY_TARGETS)

# ==========================================
# 5. Validation Error Tests
# ==========================================
def test_predict_missing_features():
    response = client.post("/predict", json={"artery_features": {}})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "missing_features" in json_data["error"]["fields"]
    # Check that missing features are listed
    missing = json_data["error"]["fields"]["missing_features"]
    assert len(missing) == 55
    assert {
        f"Missing required feature: {feature}"
        for feature in ("CHF", "Exertional CP", "LowTH Ang")
    } <= set(missing)


def test_predict_failed_artery_model_returns_named_error_not_fallback(monkeypatch):
    registry = get_artifact_registry()
    monkeypatch.setitem(
        registry.artery_model_status,
        "LCX",
        {"loaded": False, "error": "test artifact load failure"},
    )

    response = client.post(
        "/predict", json={"artery_features": get_synthetic_artery_features()}
    )

    assert response.status_code == 503
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "MODEL_UNAVAILABLE"
    assert body["error"]["fields"]["model"] == ["LCX"]
    assert "LCX" in body["error"]["message"]
    assert "data" not in body

@pytest.mark.parametrize("target", ARTERY_TARGETS)
def test_predict_prohibited_artery_target(target):
    sample = get_synthetic_artery_features()
    sample[target] = 1.0
    response = client.post("/predict", json={"artery_features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert target in json_data["error"]["fields"]

def test_predict_rejects_cath_as_artery_input():
    sample = get_synthetic_artery_features()
    sample["Cath"] = "CAD"
    response = client.post("/predict", json={"artery_features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Cath" in json_data["error"]["fields"]

def test_predict_unknown_key():
    sample = get_synthetic_artery_features()
    sample["unknown_custom_field"] = "foo"
    response = client.post("/predict", json={"artery_features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "unknown_custom_field" in json_data["error"]["fields"]

def test_predict_unknown_categorical_value_uses_saved_encoder_behavior():
    sample = get_synthetic_artery_features()
    sample["Sex"] = "UnknownGender"
    response = client.post("/predict", json={"artery_features": sample})
    assert response.status_code == 200
    assert response.json()["success"] is True

def test_predict_invalid_numeric_nan():
    sample = get_synthetic_artery_features()
    sample["Age"] = "NaN"  # String representation of NaN
    response = client.post("/predict", json={"artery_features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Age" in json_data["error"]["fields"]

def test_predict_invalid_numeric_boolean():
    sample = get_synthetic_artery_features()
    sample["Age"] = True  # Boolean should be rejected in numeric field
    response = client.post("/predict", json={"artery_features": sample})
    assert response.status_code == 422
    json_data = response.json()
    assert json_data["success"] is False
    assert json_data["error"]["code"] == "VALIDATION_ERROR"
    assert "Age" in json_data["error"]["fields"]

def test_predict_invalid_numeric_string():
    sample = get_synthetic_artery_features()
    sample["Age"] = "not_a_number"
    response = client.post("/predict", json={"artery_features": sample})
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
