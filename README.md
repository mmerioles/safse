# Species-Aware Fish Length Estimation from Images

This repository contains code, data configuration, and exploratory notebooks for estimating fish length from images with species-aware modeling. The project combines image-based species classification, length-estimation experiments, and species-level biological priors.

## Project Structure

- `src/fish_classifier/`: Reusable Python package for the species classification component.
  - `data.py` loads fold metadata, prepares train/validation/test splits, normalizes image tensors, and encodes species labels.
  - `model.py` defines a compact TensorFlow/Keras convolutional classifier with residual blocks.
  - `train.py` provides a training entry point for the classifier.
  - `predict.py` converts model outputs back into species labels.
- `notebooks/`: Research notebooks for classification, length estimation, and combined species-aware models.
- `configs/`: Species length prior files in YAML format. These provide expected, minimum, and maximum lengths by species or dataset code.
- `data/`: Pickled fold metadata and test data used by the notebooks and package code. These files are managed with Git LFS.
- `tests/`: Basic pytest test suite for repository validation.

## Workflow

The codebase is organized around a typical experimental workflow:

1. Load folded image metadata from `data/`.
2. Train or load a species classifier using `src/fish_classifier/`.
3. Run length-estimation experiments in the notebooks.
4. Incorporate species priors from `configs/` to constrain or evaluate predictions.
5. Compare model behavior using validation and test folds.

The notebooks contain the main experimental work, while the package under `src/` holds reusable components that can be imported across experiments.

## Setup

This project uses `uv` for Python environment and dependency management.

Install `uv` if needed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create and sync the environment from the project root:

```bash
uv sync
uv pip install -e .
```

## Data and Model Files

Large data and model artifacts are tracked with Git LFS. After cloning the repository, install Git LFS and pull the tracked files:

```bash
git lfs install
git lfs pull
```

Some Linux systems may require Git LFS to be installed through the system package manager first.

## Testing

Run the test suite from the project root:

```bash
uv run pytest
```
