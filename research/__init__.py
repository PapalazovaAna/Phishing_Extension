"""Training, evaluation and dataset scripts for the URL model.

Run the scripts from the repository root, for example:

    python -m research.evaluate_urlbert_vonpower

The evaluation reuses the backend code (URL preprocessing, chunking,
reputation and scoring), so the backend directory is added to the
import path here.
"""

from pathlib import Path
import sys


RESEARCH_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESEARCH_DIR.parent
BACKEND_DIR = REPOSITORY_DIR / "backend"

# Datasets, not part of the repository (see research/README.md).
DATA_DIR = RESEARCH_DIR / "data"

# PyTorch versions of the models, used for evaluation.
MODELS_DIR = RESEARCH_DIR / "models"

RESULTS_DIR = RESEARCH_DIR / "results"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
