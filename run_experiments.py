"""
Training Rubisco kinetics prediction models.

Usage
-----
    python run_experiments.py --model gp --scope general
    python run_experiments.py --model rf --scope plants
    python run_experiments.py --all

Expected input layout (see data/README.md and main README.md):
    data/processed/{target}.csv                          -- kinetics + sequence data
    data/embeddings/{target}_embeddings/{model}_{layer}.csv  -- ESM mean-representation
                                                               embeddings (not included in
                                                               this repo; see embeddings
                                                               README for how to generate)
"""

import argparse
import os
import time

import gpflow
import pandas as pd

from src.data_prep import train_test_split_general, train_test_split_plants
from src.models.gp_model import set_seed_gp
from src.pipeline import (
    add_streptophyta_metrics,
    run_randomsearchcv_gp,
    run_randomsearchcv_rf,
    train_and_test_all_models_gp,
    train_and_test_all_models_rf,
)
from src.utils import geometric_mean, set_seed, time_taken

TARGETS = ["Specificity", "kcatC", "KC"]

ESM_MODELS = {
    'esm2_t6_8M_UR50D': 6,
    'esm2_t12_35M_UR50D': 12,
    'esm2_t30_150M_UR50D': 30,
    'esm2_t33_650M_UR50D': 33,
    'esm2_t36_3B_UR50D': 36,
    'esm1b_t33_650M_UR50S': 33,
}

DATA_DIR = "data/processed"
EMBEDDINGS_DIR = "data/embeddings"


def _make_results_dirs(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'cv_AP'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'train_AP'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'test_AP'), exist_ok=True)


def _finalize_target_run(best_params_df, train_test_results_df, target, results_dir,
                          compute_streptophyta_metrics):
    """Shared post-processing: merge CV + train/test results, compute Score,
    select default layer (esm2_t33_650M_UR50D / layer 33) and best layer
    (highest Score), optionally add Streptophyta-specific metrics.
    """
    best_params_df = best_params_df.copy()
    train_test_results_df = train_test_results_df.copy()
    best_params_df['Params'] = best_params_df['Params'].apply(str)
    train_test_results_df['Params'] = train_test_results_df['Params'].apply(str)

    final_df = pd.merge(best_params_df, train_test_results_df, on=['Model', 'Layer', 'Params'],
                         suffixes=('_cv', '_test'))
    final_df['Score'] = final_df.apply(
        lambda row: geometric_mean(row['CV Correlation'], row['Test Correlation']), axis=1
    )
    final_df = final_df.round(3)
    final_df.to_csv(f"{target}_cvtestscores.csv", index=False)

    default_layer = final_df[
        (final_df['Model'] == 'esm2_t33_650M_UR50D') & (final_df['Layer'] == 33)
    ].copy()
    best_layer = final_df.loc[final_df['Score'].idxmax()].to_frame().T

    default_layer['Target'] = target
    default_layer['ESM type'] = 'default'
    best_layer['Target'] = target
    best_layer['ESM type'] = 'best'

    if compute_streptophyta_metrics:
        add_streptophyta_metrics(
            default_layer,
            f"{target}/cv_AP/{target}_esm2_t33_650M_UR50D_33_cv_actual_vs_predicted.csv",
            f"{target}/test_AP/esm2_t33_650M_UR50D_33_test_actual_vs_predicted.csv",
        )
        best_model = best_layer.iloc[0]['Model']
        best_layer_idx = int(best_layer.iloc[0]['Layer'])
        add_streptophyta_metrics(
            best_layer,
            f"{target}/cv_AP/{target}_{best_model}_{best_layer_idx}_cv_actual_vs_predicted.csv",
            f"{target}/test_AP/{best_model}_{best_layer_idx}_test_actual_vs_predicted.csv",
        )

    return default_layer, best_layer


def run_gp(scope):
    """scope: 'general' or 'plants'"""
    assert scope in ("general", "plants")
    set_seed_gp(seed=42, cuda_visible_devices="1" if scope == "general" else "0")
    split_fn = train_test_split_general if scope == "general" else train_test_split_plants

    kernel = gpflow.kernels.Matern52()  # single shared instance -- see gp_model.py docstring
    start_time = time.time()
    all_targets = []

    for target in TARGETS:
        results_dir = target
        _make_results_dirs(results_dir)

        input_df = pd.read_csv(f"{DATA_DIR}/{target}.csv").set_index("Name")
        print(f"[GP/{scope}] Processing target: {target}")

        best_params_cv_scores, best_params_df = run_randomsearchcv_gp(
            target=target, models=ESM_MODELS, input_df=input_df, results_dir=results_dir,
            embeddings_dir=EMBEDDINGS_DIR, kernel=kernel, split_fn=split_fn,
        )
        train_test_results_df = train_and_test_all_models_gp(
            models=ESM_MODELS, input_df=input_df, target=target, kernel=kernel,
            best_params_cv_scores=best_params_cv_scores, results_dir=results_dir,
            embeddings_dir=EMBEDDINGS_DIR, split_fn=split_fn,
        )
        train_test_results_df.to_csv(f'{results_dir}/{target}_train_test_results.csv', index=False)

        default_layer, best_layer = _finalize_target_run(
            best_params_df, train_test_results_df, target, results_dir,
            compute_streptophyta_metrics=(scope == "general"),
        )
        all_targets.extend([default_layer, best_layer])
        print(f"Results saved for {target}.")

    pd.concat(all_targets, ignore_index=True).to_csv("all_targets_metrics.csv", index=False)
    h, m, s = time_taken(start_time, time.time())
    print(f"\n[GP/{scope}] Time taken: {h}h {m}m {s}s")


def run_rf(scope):
    """scope: 'general' or 'plants'"""
    assert scope in ("general", "plants")
    set_seed(22)  # matches original scripts; see rf_model.py note on this being vestigial
    split_fn = train_test_split_general if scope == "general" else train_test_split_plants

    start_time = time.time()
    all_targets = []

    for target in TARGETS:
        results_dir = target
        _make_results_dirs(results_dir)

        input_df = pd.read_csv(f"{DATA_DIR}/{target}.csv").set_index("Name")
        print(f"[RF/{scope}] Processing target: {target}")

        best_params_cv_scores, best_params_df = run_randomsearchcv_rf(
            target=target, models=ESM_MODELS, input_df=input_df, results_dir=results_dir,
            embeddings_dir=EMBEDDINGS_DIR, split_fn=split_fn,
        )
        train_test_results_df = train_and_test_all_models_rf(
            models=ESM_MODELS, input_df=input_df, target=target,
            best_params_cv_scores=best_params_cv_scores, results_dir=results_dir,
            embeddings_dir=EMBEDDINGS_DIR, split_fn=split_fn,
        )
        train_test_results_df.to_csv(f'{results_dir}/{target}_train_test_results.csv', index=False)

        default_layer, best_layer = _finalize_target_run(
            best_params_df, train_test_results_df, target, results_dir,
            compute_streptophyta_metrics=(scope == "general"),
        )
        all_targets.extend([default_layer, best_layer])
        print(f"Results saved for {target}.")

    pd.concat(all_targets, ignore_index=True).to_csv("all_targets_metrics.csv", index=False)
    h, m, s = time_taken(start_time, time.time())
    print(f"\n[RF/{scope}] Time taken: {h}h {m}m {s}s")


RUNNERS = {
    ("gp", "general"): lambda: run_gp("general"),
    ("gp", "plants"): lambda: run_gp("plants"),
    ("rf", "general"): lambda: run_rf("general"),
    ("rf", "plants"): lambda: run_rf("plants"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gp", "rf"], help="Which model type to train")
    parser.add_argument("--scope", choices=["general", "plants"], help="Which data scope to train on")
    parser.add_argument("--all", action="store_true", help="Run all four combinations")
    args = parser.parse_args()

    if args.all:
        for (model, scope), runner in RUNNERS.items():
            runner()
    elif args.model and args.scope:
        RUNNERS[(args.model, args.scope)]()
    else:
        parser.error("Specify --model and --scope, or use --all")
