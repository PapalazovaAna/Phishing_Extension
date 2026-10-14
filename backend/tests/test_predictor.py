import json

from app.ml.predictor import (
    load_preprocessing,
    preprocess_url,
)


def test_url_is_unchanged_without_preprocessing():
    assert preprocess_url(
        "https://www.google.com/",
        {},
    ) == "https://www.google.com/"


def test_scheme_is_removed_when_configured():
    preprocessing = {"strip_scheme": True}

    assert preprocess_url(
        "https://www.google.com/",
        preprocessing,
    ) == "www.google.com/"

    assert preprocess_url(
        "HTTP://example.com/login",
        preprocessing,
    ) == "example.com/login"

    assert preprocess_url(
        "example.com/login",
        preprocessing,
    ) == "example.com/login"


def test_missing_preprocessing_file_means_no_preprocessing(tmp_path):
    assert load_preprocessing(tmp_path) == {}


def test_preprocessing_file_is_loaded(tmp_path):
    (tmp_path / "preprocessing.json").write_text(
        json.dumps({"strip_scheme": True}),
        encoding="utf-8",
    )

    assert load_preprocessing(tmp_path) == {"strip_scheme": True}
