# GCTAF SupCon Time-Series Prediction

This repository contains deep learning experiments for solar active-region time-series data. The main workflow is a **SWAN/GCTAF supervised contrastive classifier** for binary solar flare prediction, with an additional experimental transformer forecaster for multivariate time-series regression.

The project uses PyTorch and is managed with [`uv`](https://docs.astral.sh/uv/), so setup is reproducible from `pyproject.toml` and `uv.lock`.

## Project Overview

### Classification

The primary experiment is in `classification/swan_supcon_classifier.py`.

It trains a missing-value-aware SWAN-style encoder with:

- standardized time-series values,
- an observation mask,
- a time-since-last-observed delta tensor,
- learnable temporal embeddings,
- relative-position self-attention,
- GCTAF global-token fusion blocks,
- supervised contrastive pretraining,
- a frozen encoder plus linear classifier for binary flare prediction.

The classifier groups flare subtypes as:

| Raw subtype | Binary class |
| --- | --- |
| `F`, `FQ`, `Q`, `B`, `C` | `0` non-flare / weak flare |
| `M`, `X` | `1` major flare |

### Regression

`multivariate_regression/main.py` contains a transformer-based forecaster for predicting future multivariate solar features. It supports 60-step and 120-step inputs, raw or delta prediction, optional log scaling for selected physical features, and comparison against a persistence baseline.

This script currently keeps Colab-style paths such as `/content/partition1_grouped.npz`, so update those paths before running locally.

## Repository Layout

```text
.
|-- classification/
|   |-- main.py                         # Entry point for the main classifier
|   |-- swan_supcon_classifier.py       # Main SWAN + SupCon experiment
|   `-- swan_supcon_classifier_exp.py   # Standalone/experimental variant
|-- multivariate_regression/
|   `-- main.py                         # Experimental forecasting script
|-- notebooks/
|   `-- multivariate_timeseries_regression.ipynb
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Requirements

- macOS, Linux, or Windows
- Python `>=3.13`
- `uv`
- PyTorch-compatible hardware:
  - Apple Silicon uses `mps` when available.
  - NVIDIA GPUs use `cuda` when available.
  - Otherwise the code falls back to CPU.

## Setup With uv

Install `uv` if it is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then check:

```bash
uv --version
```

Clone or enter the project directory:

```bash
cd gctaf-supcon-time-series-prediction
```

Create/sync the virtual environment from the lockfile:

```bash
uv sync
```

Run a quick import check:

```bash
uv run python -c "import torch, numpy, pandas, sklearn; print(torch.__version__)"
```

## Data Setup

The classification scripts expect a local `LLM_TS/` directory at the repository root:

```text
LLM_TS/
|-- partition1_grouped.npz
`-- partition2_grouped.npz
```

Each `.npz` file should contain:

| Key | Shape | Description |
| --- | --- | --- |
| `features` | `(N, T, F)` | Multivariate active-region time series |
| `flare_type` | `(N,)` | Raw flare subtype labels such as `FQ`, `B`, `C`, `M`, `X` |

Missing feature values may be represented as `NaN`. The classifier standardizes observed values only, builds an observation mask, computes time-since-last-observed deltas, and fills missing standardized values with zero before feeding tensors to the model.

The data directory is intentionally not committed because these files are usually large.

## Run the Classifier

Run the main classification experiment:

```bash
uv run python classification/main.py
```

This calls `run_script()` from `classification/swan_supcon_classifier.py`, which performs three repeated runs using seeds `42`, `43`, and `44`.

Each run has two phases:

1. **Supervised contrastive pretraining**
   - seventy epochs
   - positives are samples with the same grouped binary label
   - cosine learning-rate schedule with warmup

2. **Frozen-encoder linear classification**
   - forty epochs
   - weighted cross-entropy
   - best checkpoint selected by TSS

The script reports:

- macro F1
- class-1 F1
- ROC-AUC
- PR-AUC
- TSS
- HSS
- confusion matrix
- classification report

It saves these artifacts in the repository root:

```text
supcon_swan_best.pt
confusion_matrix.png
training_curves.png
```

## Run the Experimental Variant

The experimental classifier file can also be run directly:

```bash
uv run python classification/swan_supcon_classifier_exp.py
```

It uses the same expected `LLM_TS/partition1_grouped.npz` and `LLM_TS/partition2_grouped.npz` inputs.

## Run the Regression Experiment

Before running locally, edit the `file_path` values in `multivariate_regression/main.py`. They currently point to:

```python
"/content/partition1_grouped.npz"
```

After updating paths, run:

```bash
uv run python multivariate_regression/main.py
```

The regression `.npz` data is expected to include:

| Key | Shape | Description |
| --- | --- | --- |
| `features` | `(N, 60, 8)` | Input feature windows |
| `timestamps` | `(N, 60)` | Timestamps for each window |
| `ar_region` | `(N,)` | Active-region identifier |
| `feature_names` | `(8,)` | Feature names |

The script builds temporal train/validation/test splits by active region, trains transformer forecasting models, and compares predictions with a persistence baseline using MAE, RMSE, per-feature R2, and per-horizon MAE.

## Dependencies

The project dependencies are declared in `pyproject.toml`:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Add new dependencies with:

```bash
uv add package-name
```

Then commit both `pyproject.toml` and `uv.lock`.

## Notes

- The active classifier automatically chooses `mps`, `cuda`, or `cpu`.
- The classifier uses `num_workers=0`, which is a stable default for notebooks and macOS.
- Generated artifacts such as model checkpoints and plots can be safely deleted and regenerated.
- For reproducibility, keep `uv.lock` under version control.
