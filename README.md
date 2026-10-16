# mlmaba_Doctors_Guide

A research chat interface for chest X-ray report generation. You upload a chest X-ray, and the stages stream into the chat one after another. The result is a generated report, with similar training X-rays, matching reports and CheXbert labels beside it. Sessions are stored, can be replayed, and can be exported as JSON or Markdown.

The models are hybrid Mamba / Mamba-3 and mLSTM language models that read image prefix tokens from a chest X-ray image encoder.

> **Research prototype, not for clinical use.**

## What is in this repository

| Path | What it holds |
|---|---|
| `app/` | The FastAPI server, the page (plain ES modules in `app/static/`, no build step), the SSH tunnel and the runbook `app/README.md` |
| `hybrid_xmamba/` | The model package: Mamba, Mamba-3 (SSD), mLSTM, sLSTM and attention layers, the language model, the image prefix mapper and the training loop |
| `scripts/` | Training, evaluation and test code, the retrieval gallery tools, the cluster job wrappers (`*_h100.sh`) and the checks |
| `configs/` | Hydra configs: model, dataset, trainer |
| `tests/` | Python tests, the page's node tests (`tests/frontend/`) and the end-to-end tests in Chrome (`tests/e2e/`) |
| `docs/chat_ui/evidence/` | Screenshots of the page, taken from the tiny test model |

**No data and no weights.** This repository holds code only. It contains no MIMIC-CXR data, no model weights and no outputs. To train or serve the real models you need credentialed PhysioNet access to MIMIC-CXR and MIMIC-CXR-JPG under their data use agreement. The data, the weights and the session store stay on your own machine or cluster, outside the repository. A pre-commit hook (`bash scripts/install_hooks.sh`) refuses data and weight files in a commit.

## Quick start (laptop, tiny test model)

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt -r requirements_eval.txt -r app/requirements.txt -r requirements-e2e.txt
venv/bin/pip install -e .

# The page and server with a tiny random model and a synthetic gallery: no data and no weights needed.
venv/bin/python -c "import uvicorn; from app.server import create_app; uvicorn.run(create_app(home='/tmp/cxrchat_dev', tiny_step_delay_s=0.05, tiny_gallery=True), host='127.0.0.1', port=8000, log_level='warning')"
```

Then open http://127.0.0.1:8000. The tiny model's text is never a real report: its cards say so, and the page shows a "test model" badge.

## Checks

```bash
bash scripts/validate.sh                                    # configs, the Python tests, the page's node tests, a model smoke test
venv/bin/python -m pytest tests/e2e -m e2e -q               # the page in a real Chrome
venv/bin/python scripts/chat_ui_layout_check.py             # layout at phone and desktop widths
venv/bin/python scripts/chat_ui_browser_check.py --no-evidence
venv/bin/python scripts/chat_ui_features_check.py --no-evidence
```

## Training and evaluation

- **Training:**
  - `scripts/train.py`: language-model pre-training (Hydra);
  - `scripts/train_contrastive.py`: the text and image encoders;
  - `scripts/train_report_generation.py`: the report generator.
- **Evaluation:**
  - `scripts/evaluate_report_generation.py`: ROUGE-L, BLEU and CheXbert F1;
  - `scripts/score_chexbert_standalone.py`;
  - `scripts/bootstrap_compare.py`: paired bootstrap confidence intervals.
- **Retrieval gallery:**
  - `scripts/build_retrieval_gallery.py`;
  - `scripts/label_gallery_reports.py`.
- **On a SLURM cluster:** the `scripts/*_h100.sh` wrappers submit these as jobs.

## Serving on a cluster

`app/README.md` is the runbook: start the server as a CPU job, reach it through the SSH tunnel, and use, stop, requeue or export. Cluster-specific settings live in `scripts/chat_cluster.env`, which git ignores: copy `scripts/chat_cluster.env.example` and fill it in. That includes the SLURM site settings (partition, account, QOS, excluded nodes): the wrappers carry placeholders for them, and `bash scripts/chat_remote.sh sync` fills them in on the cluster copy.

## Licence

All rights reserved; see `LICENSE`. Third-party notices are in `NOTICE`.
