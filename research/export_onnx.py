"""Export a trained URL model to ONNX for the backend.

The backend runs the model with ONNX Runtime instead of PyTorch,
which needs much less memory. This script converts the PyTorch model
in research/models/ and writes the files the backend needs:

    backend/models/<name>/model.onnx
    backend/models/<name>/tokenizer.json
    backend/models/<name>/preprocessing.json

Afterwards it checks that both versions give the same probabilities.

Run from the repository root:

    python -m research.export_onnx
    python -m research.export_onnx --name urlbert_combined_v2
"""

import argparse
import json
import shutil

import numpy as np
import onnxruntime
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from research import BACKEND_DIR, MODELS_DIR


# URLs for comparing the PyTorch and ONNX predictions.
CHECK_URLS = [
    "https://www.google.com/",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://finki.ukim.mk/",
    "https://www.booking.com/searchresults.html?dest_id=8767025&aid=304142",
    "https://rnicrosoft.com/",
    "http://secure-login-verification-example.com/login",
    "http://192.168.1.25/login",
    "https://paypal-login.com/verify/account?session=" + "a1b2c3" * 30,
]


class LogitsOnly(torch.nn.Module):
    # The exported graph returns only the logits tensor.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="urlbert_combined_v2",
        help="Model directory name in research/models/.",
    )

    args = parser.parse_args()

    source_dir = MODELS_DIR / args.name
    target_dir = BACKEND_DIR / "models" / args.name
    target_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(source_dir),
        local_files_only=True,
    )

    # The default attention implementation optimises away the
    # padding mask during export, eager attention keeps it.
    model = AutoModelForSequenceClassification.from_pretrained(
        str(source_dir),
        local_files_only=True,
        attn_implementation="eager",
    )

    model.eval()

    max_length = int(model.config.max_position_embeddings)

    # The example must contain padding, otherwise the exported graph
    # may treat the attention mask as constant.
    example_ids = torch.ones((2, max_length), dtype=torch.long)
    example_mask = torch.ones((2, max_length), dtype=torch.long)
    example_mask[1, max_length // 2:] = 0

    onnx_path = target_dir / "model.onnx"

    torch.onnx.export(
        LogitsOnly(model).eval(),
        (example_ids, example_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "chunks"},
            "attention_mask": {0: "chunks"},
            "logits": {0: "chunks"},
        },
        opset_version=17,
        dynamo=False,
    )

    shutil.copy(source_dir / "tokenizer.json", target_dir / "tokenizer.json")

    preprocessing_path = source_dir / "preprocessing.json"

    preprocessing = (
        json.loads(preprocessing_path.read_text(encoding="utf-8"))
        if preprocessing_path.exists()
        else {}
    )

    preprocessing.setdefault("max_length", max_length)

    (target_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2),
        encoding="utf-8",
    )

    print(f"Exported to {target_dir}")

    # torch.onnx.export restores the training mode of the wrapper,
    # so make sure dropout is off for the comparison.
    model.eval()

    # Compare both versions on the same chunks.
    from app import predictor

    session = onnxruntime.InferenceSession(str(onnx_path))
    largest_difference = 0.0

    for url in CHECK_URLS:
        token_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(predictor.preprocess_url(url, preprocessing))
        )

        input_ids, attention_mask = predictor.create_chunks(
            token_ids,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id or 0,
            max_length=max_length,
        )

        with torch.no_grad():
            torch_logits = model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
            ).logits.numpy()

        onnx_logits = session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]

        largest_difference = max(
            largest_difference,
            float(np.abs(torch_logits - onnx_logits).max()),
        )

    print(f"Largest logit difference PyTorch vs ONNX: {largest_difference:.2e}")

    if largest_difference > 1e-3:
        raise SystemExit("The ONNX model does not match the PyTorch model.")


if __name__ == "__main__":
    main()
