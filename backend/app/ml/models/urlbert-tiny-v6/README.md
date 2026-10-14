---
license: apache-2.0
datasets:
- JPxxx/url-benchmark-dataset
- ealvaradob/phishing-dataset
- cybersectony/PhishingEmailDetectionv2.0
- kmack/Phishing_urls
- snats/url-classifications
- phishdestroy/destroylist
- WebOrganizer/TopicAnnotations-Llama-3.1-8B
pipeline_tag: feature-extraction
tags:
- transformers
- pytorch
- feature-extraction
- text-classification
- url-classification
- malicious-url
- malicious-url-detection
- phishing
- phishing-detection
- malware
- cybersecurity
- security
- tiny
- lightweight
- embeddings
---


# CrabInHoney/urlbert-tiny-v6

**CrabInHoney/urlbert-tiny-v6** is a lightweight URL encoder based on the ModernBERT architecture, designed for malicious URL detection, phishing classification, and URL feature extraction.

With only 2.03M parameters and a hidden dimension of 128, the model executes very fast simply due to its small footprint and compact embedding size, while remaining competitive on detection tasks.

The base model was trained via knowledge distillation of hidden representations from an ensemble of multiple teacher models, which were intentionally trained specifically for this purpose. Please note that this model is designed to process and analyze URL strings exclusively.

The repository contains the base encoder alongside 10 pre-trained classification heads trained on various public cybersecurity datasets.

---

### Key Specifications

| Parameter | Value |
| :--- | :--- |
| **Architecture** | ModernBERT (`ModernBertModel`) |
| **Parameters** | 2,033,408 (2.03M total; 0.98M backbone, 1.05M embeddings) |
| **Hidden Dimension** | 128 |
| **Layers / Heads** | 6 hidden layers / 4 attention heads |
| **Context Window** | 256 tokens |
| **Vocabulary Size** | 8,193 |
| **Model Size** | ~7.76 MB (FP32) / 3.88 MB (FP16) |
| **Average Tokens / URL** | 24.8 (Truncation rate: 0.093% at `max_len=256`) |

---

### Benchmark Comparison

The models were evaluated on the [JPxxx/url-benchmark-dataset](https://huggingface.co/datasets/JPxxx/url-benchmark-dataset)

The benchmark contains 3M URLs with a strict 9:1 benign-to-malicious ratio. Evaluation was conducted using site-aware 5-fold cross-validation, ensuring that no second-level domain or IP address appears in more than one fold to eliminate data leakage.

| Rank | Model | Base Dim | AUPRC ↑ | AUROC ↑ | F1 ↑ | TPR @ 0.1% FPR ↑ | TPR @ 1% FPR ↑ | FPR @ 95% TPR ↓ |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [CrabInHoney/urlbert-tiny-v6](https://huggingface.co/CrabInHoney/urlbert-tiny-v6) | **128** | **0.9732** | **0.9928** | **0.9302** | **0.8339** | **0.9337** | **0.0173** |
| 2 | [r3ddkahili/final-complete-malicious-url-model](https://huggingface.co/r3ddkahili/final-complete-malicious-url-model) | 768 | 0.9627 | 0.9901 | 0.9120 | 0.7855 | 0.9076 | 0.0335 |
| 3 | [kmack/malicious-url-detection](https://huggingface.co/kmack/malicious-url-detection) | 768 | 0.9626 | 0.9903 | 0.9137 | 0.7491 | 0.9123 | 0.0294 |
| 4 | [ealvaradob/bert-finetuned-phishing](https://huggingface.co/ealvaradob/bert-finetuned-phishing) | 1024 | 0.9451 | 0.9858 | 0.8843 | 0.6446 | 0.8658 | 0.0544 |
| 5 | [cybersectony/phishing-email-detection-distilbert_v2.4.1](https://huggingface.co/cybersectony/phishing-email-detection-distilbert_v2.4.1) | 768 | 0.9426 | 0.9855 | 0.8803 | 0.6670 | 0.8600 | 0.0645 |
| 6 | [CrabInHoney/urlbert-tiny-base-v4](https://huggingface.co/CrabInHoney/urlbert-tiny-base-v4) | 192 | 0.9339 | 0.9826 | 0.8716 | 0.6523 | 0.8416 | 0.0844 |
| 7 | [CrabInHoney/urlbert-tiny-base-v3](https://huggingface.co/CrabInHoney/urlbert-tiny-base-v3) | 192 | 0.8871 | 0.9720 | 0.8096 | 0.4817 | 0.7372 | 0.1531 |
| 8 | [CrabInHoney/urlbert-tiny-base-v2](https://huggingface.co/CrabInHoney/urlbert-tiny-base-v2) | 192 | 0.8223 | 0.9567 | 0.7381 | 0.3657 | 0.6123 | 0.2218 |

---

### Feature Extraction (Embeddings)

```python
from transformers import AutoModel, AutoTokenizer
import torch

REPO = "CrabInHoney/urlbert-tiny-v6"

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModel.from_pretrained(REPO)

text = "http://example.com/login-verify-account"
inputs = tok(text, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs)

print(out.last_hidden_state.shape)
```

**Output:**
```text
torch.Size([1, 15, 128])
```
**Important Note on Pooling:** For optimal downstream performance, it is highly recommended to use a combination of **CLS + MEAN pooling** (concatenating the `[CLS]` token representation with the mean representation of all tokens). This strategy yields the most robust and useful signal for feature extraction.

---

### Pretrained Classification Heads

Trained classification heads are stored in the `heads/` directory:

| Head Subfolder | Source Dataset | Classes | Accuracy | Test Loss |
| :--- | :--- | :---: | :---: | :---: |
| `phiusiil_phishing_url` | [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) | 2 | **99.86%** | 0.0833 |
| `cybersectony_phishing_email_v2` | [Cybersectony Phishing v2.0](https://huggingface.co/datasets/cybersectony/PhishingEmailDetectionv2.0) | 2 | **99.83%** | 0.0841 |
| `ealvaradob_phishing_dataset` | [EA Phishing Dataset](https://huggingface.co/datasets/ealvaradob/phishing-dataset) | 2 | **99.40%** | 0.0965 |
| `malicious_urls_4class` | [Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) | 4 | **99.26%** | 0.1556 |
| `url_65lakh` | [65 Lakh+ Labeled URLs](https://www.kaggle.com/datasets/somanshumahajan/65-lakh-urls-labelled-as-1-and-0) | 2 | **99.21%** | 0.1015 |
| `phishbd_2026` | [PhishBD_2026](https://data.mendeley.com/datasets/8d6zsfwc7z) | 2 | **97.27%** | 0.1364 |
| `phishdestroy_destroylist` | [PhishDestroy Destroylist](https://huggingface.co/datasets/phishdestroy/destroylist) | 2 | **95.68%** | 0.1823 |
| `kmack_phishing_urls` | [KMack Phishing URLs](https://huggingface.co/datasets/kmack/Phishing_urls) | 2 | **90.89%** | 0.2514 |
| `snats_url_classifications_clean` | [Snats URL Classifications](https://huggingface.co/datasets/snats/url-classifications) | 17 | **58.91%** | 1.4642 |
| `weborganizer_topic_annotations` | [WebOrganizer Topic Annotations](https://huggingface.co/datasets/WebOrganizer/TopicAnnotations-Llama-3.1-8B) | 24 | **52.32%** | 1.7157 |

---

### Multi-Head Classification Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

REPO = "CrabInHoney/urlbert-tiny-v6"

HEADS = [
    "cybersectony_phishing_email_v2",
    "ealvaradob_phishing_dataset",
    "kmack_phishing_urls",
    "malicious_urls_4class",
    "phishbd_2026",
    "phishdestroy_destroylist",
    "phiusiil_phishing_url",
    "snats_url_classifications_clean",
    "url_65lakh",
    "weborganizer_topic_annotations",
]

tok = AutoTokenizer.from_pretrained(REPO)
text = "http://paypal-secure-login.verify-account.com"

for head in HEADS:
    model = AutoModelForSequenceClassification.from_pretrained(
        REPO, subfolder=f"heads/{head}", trust_remote_code=True
    )
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=model.config.max_length)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred = probs.argmax().item()
    print(f"{head:35s} -> {model.config.id2label[pred]:15s} ({probs[pred]:.3f})")
```

**Output:**
```text
cybersectony_phishing_email_v2      -> phishing_url    (0.984)
ealvaradob_phishing_dataset         -> phishing        (0.985)
kmack_phishing_urls                 -> phishing        (0.984)
malicious_urls_4class               -> phishing        (0.979)
phishbd_2026                        -> phishing        (0.985)
phishdestroy_destroylist            -> malicious       (0.984)
phiusiil_phishing_url               -> phishing        (0.985)
snats_url_classifications_clean     -> legal           (0.352)
url_65lakh                          -> malicious       (0.985)
weborganizer_topic_annotations      -> Finance & Business (0.485)
```