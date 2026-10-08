import math
from typing import Any, Dict, List, Tuple
import pandas as pd

from backend.app.core.errors import ValidationError
from backend.app.services.artifact_loader import ArtifactRegistry, EXCLUDED_FEATURES

def validate_and_preprocess(
    raw_features: Dict[str, Any], registry: ArtifactRegistry
) -> pd.DataFrame:
    """
    Validates input features strictly against the 52-feature contract and
    encodes them into a single-row DataFrame ordered by registry.feature_names.
    """
    fields: Dict[str, List[str]] = {}

    expected_features = registry.feature_names
    categorical_features = set(registry.categorical_features)
    numeric_features = set(registry.numeric_features)

    # 1. Check for missing features
    missing_features = [f for f in expected_features if f not in raw_features]
    if missing_features:
        fields["missing_features"] = [
            f"Missing required feature: {feat}" for feat in missing_features
        ]

    # 2. Check for unknown or prohibited features
    unknown_features = [f for f in raw_features if f not in expected_features]
    for feat in unknown_features:
        if feat in EXCLUDED_FEATURES:
            fields[feat] = [f"Excluded feature '{feat}' must never be provided as prediction input"]
        else:
            fields[feat] = [f"Unknown feature '{feat}' is not recognized by the model"]

    # 3. Validate feature values
    processed_row: Dict[str, float] = {}

    for feat_name, raw_val in raw_features.items():
        if feat_name in unknown_features:
            continue

        # Categorical feature validation
        if feat_name in categorical_features:
            encoder = registry.encoders[feat_name]
            allowed_classes = list(getattr(encoder, "classes_", []))

            # Strictly require string type (no silent coercion of numbers)
            if not isinstance(raw_val, str):
                fields[feat_name] = [
                    f"Expected string categorical label, got {type(raw_val).__name__}. "
                    f"Allowed values: {allowed_classes}"
                ]
                continue

            if raw_val not in allowed_classes:
                fields[feat_name] = [
                    f"Invalid categorical value '{raw_val}'. Allowed values are: {allowed_classes}"
                ]
                continue

            try:
                # Transform using saved encoder without fitting
                encoded_num = float(encoder.transform([raw_val])[0])
                processed_row[feat_name] = encoded_num
            except Exception as e:
                fields[feat_name] = [f"Encoding error: {str(e)}"]

        # Numeric feature validation
        elif feat_name in numeric_features:
            # Booleans are subclasses of int in Python; explicitly reject them
            if isinstance(raw_val, bool):
                fields[feat_name] = ["Expected numeric float or int, got boolean"]
                continue

            try:
                num_val = float(raw_val)
                if math.isnan(num_val) or math.isinf(num_val):
                    fields[feat_name] = ["Numeric value cannot be NaN or infinity"]
                else:
                    processed_row[feat_name] = num_val
            except (ValueError, TypeError):
                fields[feat_name] = [
                    f"Value '{raw_val}' could not be converted to a valid finite float"
                ]

    # If any validation errors occurred, raise 422 with all collected errors
    if fields:
        raise ValidationError(
            message="Input feature validation failed",
            fields=fields,
        )

    # 4. Construct DataFrame and enforce exact feature ordering
    df = pd.DataFrame([processed_row])
    df = df[registry.feature_names]
    return df
