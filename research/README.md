# Research: training and evaluation of the URL model

This folder contains everything about the machine learning model that is
not needed to run the backend: training notebooks, PyTorch models,
evaluation scripts and results. The backend only uses the exported ONNX
model in `backend/models/`.

```
research/
  notebooks/
    train_urlbert_vonpower.ipynb    first model (vonpower only)
    train_urlbert_combined.ipynb    current model (vonpower + ealvaradob, no scheme)
  models/
    urlbert_combined_v2/            current model (PyTorch)
    urlbert_vonpower_full/          previous model (PyTorch, for comparison)
  data/                             datasets, not in the repository
  results/                          evaluation metrics (JSON)
  export_onnx.py                    PyTorch model -> backend/models/ (ONNX)
  evaluate_urlbert_vonpower.py      model evaluation on the vonpower test split
  evaluate_urlbert_ealvaradob.py    model evaluation on the ealvaradob test split
  evaluate_pipeline.py              evaluation of the complete backend scoring
  download_ealvaradob_dataset.py
  inspect_vonpower_dataset.py
  inspect_ealvaradob_dataset.py
```

## Setup

All scripts are run from the repository root as modules
(`python -m research.<script>`), because they reuse code from the backend.
They need the packages from `requirements.txt`, which include PyTorch:

```bash
pip install -r research/requirements.txt
```

Datasets in `research/data/`:

- `vonDataset20180426.dill`: the vonpower dataset (train, validation and
  test splits)
- `ealvaradob/urls.json`: download with
  `python -m research.download_ealvaradob_dataset`

## Training

The models were trained on Google Colab (free T4 GPU) with the notebooks
in `notebooks/`. They contain the complete output of the training run.
`train_urlbert_combined.ipynb` reads the datasets from Google Drive and
saves the model to `MyDrive/phishing_models/urlbert_combined_v2/`; that
folder is `models/urlbert_combined_v2/` here. The notebooks are kept
exactly as they were run, so file paths mentioned in them refer to the
earlier folder layout (`backend/ml/`, `backend/app/ml/`).

## Exporting a model for the backend

```bash
python -m research.export_onnx --name urlbert_combined_v2
```

This writes `model.onnx`, `tokenizer.json` and `preprocessing.json` to
`backend/models/urlbert_combined_v2/` and checks that the ONNX model gives
the same results as the PyTorch model.

## Evaluation

```bash
python -m research.evaluate_urlbert_vonpower
python -m research.evaluate_urlbert_ealvaradob --scheme https
python -m research.evaluate_urlbert_ealvaradob --scheme http
python -m research.evaluate_pipeline
```

Without options the scripts evaluate the deployed ONNX model. Other models:

```bash
python -m research.evaluate_urlbert_vonpower --model-dir research/models/urlbert_vonpower_full
python -m research.evaluate_urlbert_vonpower --model base
```

`--model base` is the original `CrabInHoney/urlbert-tiny-v4-phishing-classifier`
from Hugging Face. The metrics are written to `results/` as JSON, the
predictions for every URL as CSV (not in the repository). The results
are described in the main [README](../README.md#evaluation).
