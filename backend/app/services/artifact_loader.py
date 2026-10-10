import os
import pickle
import logging
from typing import Dict, List, Any, Optional

from backend.app.core.config import get_settings

logger = logging.getLogger("cad_backend")

EXCLUDED_FEATURES = ["Cath", "LAD", "LCX", "RCA"]

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

    def load(self, artifacts_dir: Optional[str] = None) -> None:
        settings = get_settings()
        target_dir = artifacts_dir or settings.resolved_artifacts_dir()
        logger.info(f"Loading artifacts from directory: {target_dir}")
        self.is_loaded = False

        try:
            # 1. Verify files exist
            meta_path = os.path.join(target_dir, "CAD_XGBoost_Metadata.pkl")
            enc_path = os.path.join(target_dir, "CAD_Categorical_Encoders.pkl")
            model_path = os.path.join(target_dir, "CAD_XGBoost_Model.pkl")

            for p in [meta_path, enc_path, model_path]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Required artifact not found: {p}")

            # 2. Load metadata (pickle)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

            # 3. Load encoders (joblib or pickle fallback)
            try:
                import joblib
                self.encoders = joblib.load(enc_path)
            except Exception as e_joblib:
                logger.warning(f"Native joblib.load failed ({e_joblib}), attempting pickle.load")
                with open(enc_path, "rb") as f:
                    self.encoders = pickle.load(f)

            # 4. Load XGBoost Model (pickle)
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            self.models["cad"] = model

            # 5. Extract configuration
            self.feature_names = self.metadata.get("feature_names", [])
            self.target_mapping = self.metadata.get("target_mapping", {"Normal": 0, "CAD": 1})
            self.categorical_features = list(self.encoders.keys())
            self.numeric_features = [f for f in self.feature_names if f not in self.encoders]
            self.model_version = str(self.metadata.get("model_version", "1.0.0"))

            # 6. Run the 5 Startup Self-checks (PRD Section 5)
            self._run_startup_self_checks()

            self.is_loaded = True
            self.load_error = None
            logger.info("Successfully loaded all artifacts and passed startup self-checks.")

        except Exception as exc:
            self.is_loaded = False
            self.load_error = str(exc)
            logger.critical(f"FATAL: Artifact loading failed: {exc}", exc_info=True)
            raise

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
