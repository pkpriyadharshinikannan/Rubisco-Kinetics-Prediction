# Rubisco Kinetics Prediction

Predicting Rubisco kinetic properties — specificity factor (Sc/o), CO2 catalytic turnover rate (kcat), and the Michaelis constant for CO2 (Km) — from RbcL (Rubisco large subunit) protein sequences, using embeddings from pretrained protein language models (ESM) and Gaussian Process / Random Forest regression.

This work is part of a Ph.D. thesis (*Machine Learning-Driven Protein Engineering: Predictive Insights into Cyanobacteriochrome Fluorescence and Rubisco Kinetics*, KIST/Gangneung-Wonju National University, 2025) and an associated manuscript in preparation.

## Overview

- **Two model scopes**: a *general* model trained across 9 phyla, and a *plant-specific* model trained on Streptophyta (land plants) only.
- **Two model types**: Gaussian Process (GPflow, Matern52 kernel) and Random Forest (scikit-learn), compared against each other via layer-wise ESM embedding analysis.
- **Input representations**: mean-pooled embeddings from every layer of six ESM checkpoints (`esm2_t6_8M` through `esm2_t36_3B`, plus `esm1b_t33_650M`), so that both the default final layer and the best-performing intermediate layer can be identified per target property.

## Repository structure

```
data/
├── processed/                    # curated kinetics + sequence CSVs (model input)
└── README.md                     # data provenance, known caveats, QC notes

src/
├── utils.py                      # shared metrics, seeding, geometric mean
├── data_prep.py                  # preprocessing + train/test split logic (general vs. plants)
├── models/
│   ├── gp_model.py                # GPflow GPR wrapper + training/prediction
│   └── rf_model.py                # Random Forest training/prediction
└── pipeline.py                   # hyperparameter search + evaluation, shared by GP and RF

run_experiments.py                 # main entry point (see Usage below)
```

## Embeddings

Sequence embeddings are generated using **[Facebook Research's ESM](https://github.com/facebookresearch/esm)** (Evolutionary Scale Modeling) library — specifically its `extract.py` script with `--include mean`, which produces **mean-pooled representations across every layer** of each ESM checkpoint. This lets the modeling pipeline compare the standard final-layer embedding against other layers to find which one best predicts each kinetic property

This repository does not include the embedding extraction code or raw `.pt` output — instead:

1. Generate embeddings for your sequences using ESM's own `extract.py` (see their repo for setup and usage).
2. Convert the per-sequence `.pt` outputs into one CSV per (ESM checkpoint, layer) combination, with columns `Name, D1, D2, ..., Dn` (`Name` matching the `Name` column in `data/processed/*.csv`).
3. Place these CSVs at `data/embeddings/{target}_embeddings/{model}_{layer}.csv` (e.g. `data/embeddings/Specificity_embeddings/esm2_t33_650M_UR50D_33.csv`).

**Both model training and prediction expect this CSV format as input** — not raw `.pt` files — so the pipeline in `src/` and `run_experiments.py` never touches ESM or PyTorch directly.

## Usage

```bash
pip install -r requirements.txt

# Train a single combination
python run_experiments.py --model gp --scope general
python run_experiments.py --model rf --scope plants

# Train all four combinations (GP/RF x general/plants)
python run_experiments.py --all
```

Each run produces, per target (`Specificity`, `kcatC`, `KC`):
- `{target}/best_params_cv_scores.csv` — best hyperparameters and CV metrics per (ESM checkpoint, layer)
- `{target}/cv_AP/`, `{target}/train_AP/`, `{target}/test_AP/` — actual-vs-predicted values for plotting
- `{target}_cvtestscores.csv` — merged CV + test scores with a combined Score (geometric mean of CV and test correlation)
- `all_targets_metrics.csv` — default-layer vs. best-layer summary across all three targets