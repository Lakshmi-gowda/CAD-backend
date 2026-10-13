import os
import pickle
import json
import logging
from typing import Dict, List, Any, Optional

from backend.app.core.config import get_settings

logger = logging.getLogger("cad_backend")

EXCLUDED_FEATURES = ["Cath", "LAD", "LCX", "RCA"]
ARTERY_TARGETS = ("LAD", "LCX", "RCA")
ARTERY_FEATURE_COUNT = 55

class ArtifactRegistry:
    def __init__(self):
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.numeric_features: List[str] = []
        self.target_mapping: Dict[str, int] = {}
        self.model_version: str = "1.0.0"
        self.artery_metadata: Dict[str, Any] = {}
        self.artery_models: Dict[str, Any] = {}
        self.artery_feature_names: List[str] = []
        self.artery_categorical_features: List[str] = []
        self.artery_numeric_features: List[str] = []
        self.artery_category_values: Dict[str, List[str]] = {}
        self.artery_unknown_category_behavior: Dict[str, str] = {}
        self.artery_model_status: Dict[str, Dict[str, Any]] = {
            artery: {"loaded": False, "error": "Model has not been loaded"}
            for artery in ARTERY_TARGETS
        }

    def load(self, artifacts_dir: Optional[str] = None) -> None:
        settings = get_settings()
        target_dir = artifacts_dir or settings.resolved_artifacts_dir()
        logger.info(f"Loading artifacts from directory: {target_dir}")
        self.is_loaded = False
        self.models.clear()
        self.load_error = None
        self.artery_models = {}
        self.artery_metadata = {}
        self.artery_feature_names = []
        self.artery_categorical_features = []
        self.artery_numeric_features = []
        self.artery_category_values = {}
        self.artery_unknown_category_behavior = {}
        self.artery_model_status = {
            artery: {"loaded": False, "error": "Model has not been loaded"}
            for artery in ARTERY_TARGETS
        }

        cad_load_error: Optional[Exception] = None
        try:
            self._load_cad_artifacts(target_dir)
        except Exception as exc:
            self.is_loaded = False
            self.load_error = str(exc)
            cad_load_error = exc
            logger.critical(f"FATAL: Artifact loading failed: {exc}", exc_info=True)

        self._load_artery_models(target_dir)

        if cad_load_error is not None:
            raise cad_load_error

    def _load_cad_artifacts(self, target_dir: str) -> None:
        meta_path = os.path.join(target_dir, "CAD_XGBoost_Metadata.pkl")
        enc_path = os.path.join(target_dir, "CAD_Categorical_Encoders.pkl")
        model_path = os.path.join(target_dir, "CAD_XGBoost_Model.pkl")

        for path in (meta_path, enc_path, model_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required artifact not found: {path}")

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        try:
            import joblib
            self.encoders = joblib.load(enc_path)
        except Exception as e_joblib:
            logger.warning(
                f"Native joblib.load failed ({e_joblib}), attempting pickle.load"
            )
            with open(enc_path, "rb") as f:
                self.encoders = pickle.load(f)

        with open(model_path, "rb") as f:
            self.models["cad"] = pickle.load(f)

        self.feature_names = self.metadata.get("feature_names", [])
        self.target_mapping = self.metadata.get(
            "target_mapping", {"Normal": 0, "CAD": 1}
        )
        self.categorical_features = list(self.encoders.keys())
        self.numeric_features = [
            feature
            for feature in self.feature_names
            if feature not in self.encoders
        ]
        self.model_version = str(self.metadata.get("model_version", "1.0.0"))

        self._run_startup_self_checks()
        self.is_loaded = True
        self.load_error = None
        logger.info("Successfully loaded CAD artifacts and passed startup self-checks.")

    def _load_artery_models(self, target_dir: str) -> None:
        artery_dir = os.path.join(target_dir, "artery_models")
        metadata_path = os.path.join(artery_dir, "metadata.json")

        try:
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(
                    f"Required artery metadata not found: {metadata_path}"
                )
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            feature_names = metadata.get("feature_columns")
            targets = metadata.get("targets")
            if not isinstance(feature_names, list) or not feature_names:
                raise ValueError("metadata.json must contain feature_columns")
            if any(not isinstance(feature, str) or not feature for feature in feature_names):
                raise ValueError("metadata.json feature_columns must contain feature names")
            if len(feature_names) != ARTERY_FEATURE_COUNT:
                raise ValueError(
                    f"metadata.json must define exactly {ARTERY_FEATURE_COUNT} features; "
                    f"got {len(feature_names)}"
                )
            if metadata.get("feature_count") != ARTERY_FEATURE_COUNT:
                raise ValueError(
                    f"metadata.json feature_count must be {ARTERY_FEATURE_COUNT}"
                )
            if targets != list(ARTERY_TARGETS):
                raise ValueError(
                    f"metadata.json targets must be {list(ARTERY_TARGETS)}, got {targets}"
                )
            if len(feature_names) != len(set(feature_names)):
                raise ValueError("metadata.json feature_columns contains duplicates")
            included_excluded = [
                feature for feature in EXCLUDED_FEATURES if feature in feature_names
            ]
            if included_excluded:
                raise ValueError(
                    "Excluded columns cannot be artery model inputs: "
                    f"{included_excluded}"
                )
            excluded_columns = metadata.get("excluded_input_columns")
            if (
                not isinstance(excluded_columns, list)
                or set(excluded_columns) != set(EXCLUDED_FEATURES)
                or len(excluded_columns) != len(EXCLUDED_FEATURES)
            ):
                raise ValueError(
                    "metadata.json excluded_input_columns must list exactly "
                    f"{EXCLUDED_FEATURES}"
                )
            target_classes = metadata.get("target_classes")
            if not isinstance(target_classes, dict) or any(
                set(target_classes.get(artery, [])) != {"Normal", "Stenotic"}
                for artery in ARTERY_TARGETS
            ):
                raise ValueError(
                    "metadata.json target_classes must list Normal and Stenotic "
                    "for each artery"
                )

            self.artery_metadata = metadata
            self.artery_feature_names = list(feature_names)
        except Exception as exc:
            for artery in ARTERY_TARGETS:
                self.artery_model_status[artery] = {
                    "loaded": False,
                    "error": f"Artery metadata unavailable: {exc}",
                }
            logger.error("Failed to load artery model metadata: %s", exc, exc_info=True)
            return

        for artery in ARTERY_TARGETS:
            model_path = os.path.join(artery_dir, f"{artery}_model.joblib")
            try:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"Required artery artifact not found: {model_path}"
                    )

                import joblib
                model = joblib.load(model_path)
                contract = self._validate_artery_pipeline(model, artery)

                if not self.artery_categorical_features:
                    self.artery_categorical_features = contract["categorical_features"]
                    self.artery_numeric_features = contract["numeric_features"]
                    self.artery_category_values = contract["category_values"]
                    self.artery_unknown_category_behavior = contract[
                        "unknown_category_behavior"
                    ]
                elif (
                    contract["categorical_features"] != self.artery_categorical_features
                    or contract["numeric_features"] != self.artery_numeric_features
                    or contract["category_values"] != self.artery_category_values
                    or contract["unknown_category_behavior"]
                    != self.artery_unknown_category_behavior
                ):
                    raise ValueError(
                        "Preprocessing contract differs from the other artery models"
                    )

                self.artery_models[artery] = model
                self.models[artery.lower()] = model
                self.artery_model_status[artery] = {
                    "loaded": True,
                    "error": None,
                    "feature_count": len(self.artery_feature_names),
                    "classes": [str(value) for value in model.classes_],
                }
                logger.info(
                    "Loaded %s artery model from %s with %d input features",
                    artery,
                    model_path,
                    len(self.artery_feature_names),
                )
            except Exception as exc:
                self.artery_models.pop(artery, None)
                self.models.pop(artery.lower(), None)
                self.artery_model_status[artery] = {
                    "loaded": False,
                    "error": str(exc),
                }
                logger.error(
                    "Failed to load %s artery model from %s: %s",
                    artery,
                    model_path,
                    exc,
                    exc_info=True,
                )

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        cad_classes = list(self.target_mapping.keys()) if self.target_mapping else []
        statuses: Dict[str, Dict[str, Any]] = {
            "CAD": {
                "loaded": self.is_loaded,
                "error": self.load_error,
                "feature_count": len(self.feature_names) if self.is_loaded else None,
                "classes": [str(value) for value in cad_classes],
            }
        }
        statuses.update(self.artery_model_status)
        return statuses

    def _validate_artery_pipeline(self, model: Any, artery: str) -> Dict[str, Any]:
        expected_count = len(self.artery_feature_names)
        input_count = getattr(model, "n_features_in_", None)
        if input_count != expected_count:
            raise ValueError(
                f"{artery} pipeline expects {input_count} features; "
                f"metadata defines {expected_count}"
            )

        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None and list(model_features) != self.artery_feature_names:
            raise ValueError(
                f"{artery} pipeline feature names/order do not match metadata.json"
            )
        if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
            raise ValueError(f"{artery} pipeline must expose predict_proba() and classes_")
        if not hasattr(model, "predict"):
            raise ValueError(f"{artery} pipeline must expose predict()")

        named_steps = getattr(model, "named_steps", {})
        preprocessor = named_steps.get("preprocessor")
        if preprocessor is None:
            raise ValueError(f"{artery} pipeline has no fitted 'preprocessor' step")
        preprocessing_features = getattr(preprocessor, "feature_names_in_", None)
        if (
            preprocessing_features is not None
            and list(preprocessing_features) != self.artery_feature_names
        ):
            raise ValueError(
                f"{artery} preprocessing feature names/order do not match metadata.json"
            )

        transformers = {
            name: (transformer, list(columns))
            for name, transformer, columns in getattr(preprocessor, "transformers_", [])
            if columns is not None
        }
        if "numeric" not in transformers or "categorical" not in transformers:
            raise ValueError(
                f"{artery} preprocessing must define numeric and categorical transformers"
            )
        numeric_transformer, numeric_features = transformers["numeric"]
        categorical_transformer, categorical_features = transformers["categorical"]
        if (
            set(numeric_features) | set(categorical_features)
        ) != set(self.artery_feature_names) or set(numeric_features) & set(categorical_features):
            raise ValueError(
                f"{artery} preprocessing columns do not cover the metadata feature contract"
            )
        categorical_steps = getattr(categorical_transformer, "named_steps", {})
        one_hot_encoder = categorical_steps.get("onehot")
        categories = getattr(one_hot_encoder, "categories_", None)
        if categories is None or len(categories) != len(categorical_features):
            raise ValueError(f"{artery} categorical encoder has no fitted categories")

        category_values = {
            feature: [str(value) for value in values]
            for feature, values in zip(categorical_features, categories)
        }
        unknown_behavior = {
            "categorical": str(
                getattr(one_hot_encoder, "handle_unknown", "error")
            )
        }
        classes = list(getattr(model, "classes_", []))
        if set(classes) != {"Normal", "Stenotic"}:
            raise ValueError(
                f"{artery} classes_ must contain Normal and Stenotic; got {classes}"
            )
        if set(classes) != set(self.artery_metadata["target_classes"][artery]):
            raise ValueError(
                f"{artery} classes_ do not match metadata target_classes"
            )

        return {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "category_values": category_values,
            "unknown_category_behavior": unknown_behavior,
        }

    def _run_startup_self_checks(self) -> None:
        # Check 1: Feature list from metadata has exactly 52 entries
        if len(self.feature_names) != 52:
            raise ValueError(
                f"Self-check 1 failed: Expected exactly 52 features, got {len(self.feature_names)}"
            )

        # Check 2: No excluded feature is in the list
        found_excluded = [f for f in EXCLUDED_FEATURES if f in self.feature_names]
        if found_excluded:
            raise ValueError(
                f"Self-check 2 failed: Excluded features present in metadata: {found_excluded}"
            )

        # Check 3: All 18 encoder features appear in the metadata
        unknown_encoders = [f for f in self.encoders if f not in self.feature_names]
        if unknown_encoders:
            raise ValueError(
                f"Self-check 3 failed: Encoders reference unknown features: {unknown_encoders}"
            )
        if len(self.encoders) != 18:
            raise ValueError(
                f"Self-check 3 failed: Expected 18 categorical encoders, found {len(self.encoders)}"
            )

        # Check 4: Model's expected feature count equals 52 & order matches
        cad_model = self.models.get("cad")
        if cad_model is None:
            raise ValueError("Self-check 4 failed: CAD model is not registered")

        booster = cad_model.get_booster() if hasattr(cad_model, "get_booster") else None
        model_feat_count = getattr(cad_model, "n_features_in_", None)
        if model_feat_count is None and booster is not None:
            model_feat_count = booster.num_features()

        if model_feat_count is not None and model_feat_count != 52:
            raise ValueError(
                f"Self-check 4 failed: Model expected feature count is {model_feat_count}, expected 52"
            )

        model_feature_names = getattr(cad_model, "feature_names_in_", None)
        if model_feature_names is None and booster is not None:
            model_feature_names = booster.feature_names
        if (
            model_feature_names is not None
            and list(model_feature_names) != self.feature_names
        ):
            raise ValueError(
                "Self-check 4 failed: Model feature names/order do not match metadata"
            )

        # Check 5: Target mapping and model.classes_ agree on CAD vs Normal
        # model.classes_ should be [0, 1] matching Normal: 0, CAD: 1
        model_classes = getattr(cad_model, "classes_", [0, 1])
        if list(model_classes) != [0, 1]:
            raise ValueError(
                f"Self-check 5 failed: Unexpected model.classes_ {model_classes}"
            )
        if self.target_mapping.get("Normal") != 0 or self.target_mapping.get("CAD") != 1:
            raise ValueError(
                f"Self-check 5 failed: Target mapping {self.target_mapping} does not agree with Normal=0, CAD=1"
            )

_registry_instance: Optional[ArtifactRegistry] = None

def get_artifact_registry() -> ArtifactRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ArtifactRegistry()
    return _registry_instance
