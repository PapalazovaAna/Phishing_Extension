from pathlib import Path

from huggingface_hub import hf_hub_download


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ml"
    / "data"
    / "ealvaradob"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

path = hf_hub_download(
    repo_id="ealvaradob/phishing-dataset",
    repo_type="dataset",
    filename="urls.json",
    local_dir=OUTPUT_DIR,
)

print(f"Downloaded to: {path}")