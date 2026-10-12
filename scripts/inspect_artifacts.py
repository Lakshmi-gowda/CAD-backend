#!/usr/bin/env python3
"""
scripts/inspect_artifacts.py

Verification script for CAD backend ML artifacts.
Reads and validates:
  1. CAD_XGBoost_Metadata.pkl
  2. CAD_Categorical_Encoders.pkl
  3. CAD_XGBoost_Model.pkl
  4. LAD, LCX, and RCA joblib pipelines plus artery metadata

Performs the 5 CAD startup self-checks and verifies each artery pipeline's
feature count, feature order, and fitted categorical preprocessing.
"""

import os
import sys
import pickle
import pickletools
import json

def find_artifacts_dir():
    env_dir = os.environ.get("ARTIFACTS_DIR")
    candidates = [
        env_dir,
        os.path.join(os.path.dirname(__file__), "..", "backend", "app", "ml", "artifacts"),
        os.path.join(os.path.dirname(__file__), "..", "MODEL"),
        os.path.join(os.getcwd(), "MODEL"),
        os.path.join(os.getcwd(), "backend", "app", "ml", "artifacts"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            if os.path.exists(os.path.join(c, "CAD_XGBoost_Metadata.pkl")):
                return os.path.abspath(c)
    raise FileNotFoundError("Could not locate artifacts directory containing CAD_XGBoost_Metadata.pkl")

def inspect_metadata(artifacts_dir):
    meta_path = os.path.join(artifacts_dir, "CAD_XGBoost_Metadata.pkl")
    print(f"\n[1/3] Inspecting Metadata: {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    print("=" * 60)
    print("METADATA CONTENTS:")
    for k, v in meta.items():
        if k != "feature_names":
            print(f"  {k}: {v}")
    
    features = meta.get("feature_names", [])
    print(f"  Feature count: {len(features)}")
    print(f"  First 5 features: {features[:5]}")
    print(f"  Last 5 features:  {features[-5:]}")
    return meta

def inspect_encoders(artifacts_dir):
    enc_path = os.path.join(artifacts_dir, "CAD_Categorical_Encoders.pkl")
    print(f"\n[2/3] Inspecting Encoders: {enc_path}")
    print("=" * 60)
    
    encoders = {}
    sklearn_version = "Unknown"
    
    # Try native joblib loading first if installed
    try:
        import joblib
        loaded = joblib.load(enc_path)
        print("  Loaded via joblib natively.")
        for k, enc in loaded.items():
            encoders[k] = list(enc.classes_)
            if hasattr(enc, "_sklearn_version"):
                sklearn_version = enc._sklearn_version
    except Exception:
        # Fallback to parser to inspect without requiring dependencies
        with open(enc_path, "rb") as f:
            data = f.read()
        
        # Check scikit-learn version
        ver_idx = data.find(b"_sklearn_version")
        if ver_idx != -1:
            ver_part = data[ver_idx:ver_idx+40]
            # Pattern after _sklearn_version
            import re
            m = re.search(rb'([0-9]+\.[0-9]+\.[0-9]+)', ver_part)
            if m:
                sklearn_version = m.group(1).decode("utf-8")
        
        offset = 0
        while offset < len(data):
            try:
                gen = pickletools.genops(data[offset:])
                last_pos = 0
                ops = []
                for op, arg, pos in gen:
                    last_pos = pos
                    ops.append((op.name, arg))
                    if op.name == "STOP":
                        break
                offset += last_pos + 1
                
                feat_name = None
                classes = []
                for op_name, arg in ops:
                    if op_name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                        if arg in (
                            "Sex", "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease",
                            "DLP", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur",
                            "Diastolic Murmur", "Dyspnea", "Atypical", "Nonanginal", "LVH",
                            "Poor R Progression", "BBB", "VHD"
                        ):
                            feat_name = arg
                        elif arg in ("Fmale", "Male", "Y", "N", "Moderate", "Severe", "mild", "LBBB", "RBBB"):
                            classes.append(arg)
                if feat_name:
                    encoders[feat_name] = classes
            except Exception:
                break

    print(f"  Encoder type: sklearn.preprocessing.LabelEncoder")
    print(f"  scikit-learn version in artifact: {sklearn_version}")
    print(f"  Total categorical encoders found: {len(encoders)}")
    for feat, classes in encoders.items():
        print(f"    - {feat:<25}: classes_ = {classes}")
    
    return encoders, sklearn_version

def inspect_model(artifacts_dir, meta):
    model_path = os.path.join(artifacts_dir, "CAD_XGBoost_Model.pkl")
    print(f"\n[3/3] Inspecting XGBoost Model: {model_path}")
    print("=" * 60)
    
    model_type = "Unknown"
    classes = [0, 1]
    booster_features = []
    
    # Try native pickle if xgboost installed
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_type = f"{type(model).__module__}.{type(model).__name__}"
        classes = list(getattr(model, "classes_", [0, 1]))
        if hasattr(model, "feature_names_in_"):
            booster_features = list(model.feature_names_in_)
        elif hasattr(model, "get_booster"):
            booster_features = model.get_booster().feature_names
        print(f"  Loaded model natively via pickle: {model_type}")
    except Exception:
        # Fallback to inspecting booster header buffer
        with open(model_path, "rb") as f:
            data = f.read()
        
        class FakeBooster:
            def __setstate__(self, state): self.state = state
        class FakeXGB:
            def __setstate__(self, state): self.__dict__.update(state)
        class Unp(pickle.Unpickler):
            def find_class(self, mod, name):
                if mod == "xgboost.sklearn" and name == "XGBClassifier": return FakeXGB
                if mod == "xgboost.core" and name == "Booster": return FakeBooster
                return super().find_class(mod, name)
        
        f = open(model_path, "rb")
        m = Unp(f).load()
        f.close()
        
        model_type = "xgboost.sklearn.XGBClassifier"
        classes = [0, 1]
        
        handle = m._Booster.state.get("handle", b"")
        idx = handle.find(b"feature_names")
        if idx != -1:
            pos = idx + len(b"feature_names[#L") + 8
            p = pos
            for _ in range(52):
                if handle[p:p+2] == b"SL":
                    p += 2
                    slen = int.from_bytes(handle[p:p+8], "big")
                    p += 8
                    name = handle[p:p+slen].decode("utf-8")
                    p += slen
                    booster_features.append(name)

    print(f"  Model class: {model_type}")
    print(f"  Classes: {classes} (mapped to: {meta.get('target_mapping')})")
    print(f"  Booster internal feature count: {len(booster_features)}")
    return model_type, classes, booster_features

def run_self_checks(meta, encoders, booster_features, classes):
    print("\n" + "=" * 60)
    print("STARTUP SELF-CHECKS (PRD Section 5):")
    print("=" * 60)
    
    features = meta.get("feature_names", [])
    
    # Check 1: Feature list has exactly 52 entries
    c1 = len(features) == 52
    print(f"  Check 1 [52 features in metadata]:       {'PASS' if c1 else 'FAIL'} ({len(features)} features)")
    
    # Check 2: No excluded feature in list
    excluded = ["Cath", "LAD", "LCX", "RCA"]
    found_excluded = [x for x in excluded if x in features]
    c2 = len(found_excluded) == 0
    print(f"  Check 2 [No excluded features]:          {'PASS' if c2 else 'FAIL'} (Excluded: {found_excluded or 'None'})")
    
    # Check 3: Every categorical feature has an encoder
    missing_encoders = [f for f in encoders.keys() if f not in features]
    c3 = len(encoders) == 18 and len(missing_encoders) == 0
    print(f"  Check 3 [18 categorical encoders match]:  {'PASS' if c3 else 'FAIL'} ({len(encoders)}/18 encoders found)")
    
    # Check 4: Model's expected feature count equals 52 & order matches
    c4 = len(booster_features) == 52 and booster_features == features
    print(f"  Check 4 [Model features match metadata]: {'PASS' if c4 else 'FAIL'} (Exact order match: {c4})")
    
    # Check 5: Target mapping and model.classes_ agree
    tm = meta.get("target_mapping", {})
    c5 = tm.get("Normal") == 0 and tm.get("CAD") == 1 and classes == [0, 1]
    print(f"  Check 5 [Target mapping & classes agree]: {'PASS' if c5 else 'FAIL'} (Normal=0, CAD=1)")
    
    all_passed = all([c1, c2, c3, c4, c5])
    print("-" * 60)
    print(f"OVERALL SELF-CHECK RESULT: {'ALL CHECKS PASSED (5/5)' if all_passed else 'SOME CHECKS FAILED'}")
    return all_passed

def inspect_artery_models(artifacts_dir):
    artery_dir = os.path.join(artifacts_dir, "artery_models")
    metadata_path = os.path.join(artery_dir, "metadata.json")
    targets = ["LAD", "LCX", "RCA"]
    print("\n" + "=" * 60)
    print("ARTERY MODEL PIPELINE CHECKS:")
    print("=" * 60)

    missing = [
        path
        for path in [
            metadata_path,
            *[
                os.path.join(artery_dir, f"{target}_model.joblib")
                for target in targets
            ],
        ]
        if not os.path.isfile(path)
    ]
    if missing:
        for path in missing:
            print(f"  FAIL: Required artifact is missing: {path}")
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    features = metadata.get("feature_columns", [])
    metadata_targets = metadata.get("targets", [])
    if metadata_targets != targets:
        print(f"  FAIL: Expected targets {targets}, found {metadata_targets}")
        return False
    if any(target in features for target in targets):
        print("  FAIL: An artery target is included in feature_columns")
        return False

    import joblib

    all_passed = True
    print(f"  Metadata inputs: {len(features)}")
    print(f"  Exact feature order: {features}")
    print(
        f"  Class semantics: {metadata.get('negative_class')} (negative), "
        f"{metadata.get('positive_class')} (positive)"
    )

    for target in targets:
        model_path = os.path.join(artery_dir, f"{target}_model.joblib")
        try:
            model = joblib.load(model_path)
        except Exception as exc:
            print(f"  FAIL {target}: Could not load {model_path}: {exc}")
            all_passed = False
            continue

        model_features = list(getattr(model, "feature_names_in_", []))
        feature_count = getattr(model, "n_features_in_", None)
        pipeline_matches = (
            feature_count == len(features)
            and model_features == features
            and hasattr(model, "predict_proba")
            and hasattr(model, "classes_")
        )
        print(
            f"  {'PASS' if pipeline_matches else 'FAIL'} {target}: "
            f"type={type(model).__module__}.{type(model).__name__}, "
            f"n_features_in_={feature_count}, classes_="
            f"{list(getattr(model, 'classes_', []))}, "
            f"feature_order_matches={model_features == features}"
        )
        preprocessor = getattr(model, "named_steps", {}).get("preprocessing")
        if preprocessor is not None:
            for name, transformer, columns in preprocessor.transformers_:
                if columns is None:
                    continue
                print(f"    {name} columns ({len(columns)}): {list(columns)}")
                if hasattr(transformer, "categories_"):
                    print(
                        "    fitted categories: "
                        f"{dict(zip(columns, [list(values) for values in transformer.categories_]))}"
                    )
                    print(
                        "    handle_unknown: "
                        f"{getattr(transformer, 'handle_unknown', 'error')}"
                    )
        all_passed = all_passed and pipeline_matches

    print(
        "ARTERY PIPELINE RESULT: "
        f"{'ALL THREE PASSED' if all_passed else 'ONE OR MORE CHECKS FAILED'}"
    )
    return all_passed

def main():
    try:
        artifacts_dir = find_artifacts_dir()
        print(f"Artifacts located at: {artifacts_dir}")
        meta = inspect_metadata(artifacts_dir)
        encoders, sklearn_ver = inspect_encoders(artifacts_dir)
        model_type, classes, booster_feats = inspect_model(artifacts_dir, meta)
        passed = run_self_checks(meta, encoders, booster_feats, classes)
        artery_passed = inspect_artery_models(artifacts_dir)
        if not passed or not artery_passed:
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Inspection failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
