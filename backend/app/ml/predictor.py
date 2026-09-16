from pathlib import Path

import joblib
import pandas as pd

from .features import extract_url_features


MODEL_PATH = Path(__file__).parent / "url_model.joblib"

_model = None
_feature_names = None


def load_model():
    global _model, _feature_names

    if _model is not None:
        return

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        _model = artifact["model"]
        _feature_names = artifact["feature_names"]
    else:
        _model = artifact
        _feature_names = None


def predict_url(url: str) -> dict:
    load_model()

    extracted_features = extract_url_features(url)

    if _feature_names:
        row = {
            feature_name: extracted_features.get(feature_name, 0)
            for feature_name in _feature_names
        }

        features_df = pd.DataFrame(
            [row],
            columns=_feature_names
        )
    else:
        features_df = pd.DataFrame([extracted_features])

    probabilities = _model.predict_proba(features_df)[0]

    classes = list(_model.classes_)
    phishing_index = classes.index(1)

    phishing_probability = float(probabilities[phishing_index])
    score = round(phishing_probability * 100)

    if score >= 70:
        classification = "Likely phishing"
    elif score >= 40:
        classification = "Suspicious"
    else:
        classification = "Likely safe"

    indicator = (
        f"ML model estimates a {score}% phishing probability."
    )

    return {
        "score": score,
        "classification": classification,
        "probability": round(phishing_probability, 4),
        "indicator": indicator,
    }