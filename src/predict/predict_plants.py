"""
Batch prediction using the finalized *plant-specific* models (trained on
Streptophyta only; Random Forest, per thesis Table 3).

REDESIGNED from the original plants_prediction_parallel.py -- see
predict_general.py's module docstring for the full reasoning (CSV
embeddings instead of per-sequence .pt files; single vectorized batch
prediction instead of a per-sequence joblib.Parallel loop). It applies here
too, and even more directly: RandomForestRegressor.predict() is already
vectorized, so the original script was calling it once per sequence when a
single call over the whole batch does the same work.

Note: the original script's `predict_for_sequence` accepted an
`intermediate_save_path` parameter that was never actually used inside the
function body (dead parameter) -- not carried over here.
"""

import argparse
import os

import joblib
import pandas as pd

from src.predict.predict_utils import align_sequences, load_embeddings_for_target

PLANTS_MODELS_INFO = [
    {"Target": "Specificity", "ESMmodel": "esm1b_t33_650M_UR50S", "Layer": 6},
    {"Target": "kcatC", "ESMmodel": "esm2_t36_3B_UR50D", "Layer": 11},
    {"Target": "KC", "ESMmodel": "esm1b_t33_650M_UR50S", "Layer": 4},
]


def predict_plants(sequences_csv, embeddings_dir, saved_models_dir, output_csv,
                    models_info=PLANTS_MODELS_INFO):
    """Predict Sc/o, kcat, and Km for new sequences using the saved plant-specific models.

    Parameters
    ----------
    sequences_csv : str
        CSV with a 'Name' column identifying which sequences to predict for.
    embeddings_dir : str
        Base directory containing `{target}_embeddings/{esm_model}_{layer}.csv`.
    saved_models_dir : str
        Base directory containing `plants/{target}/{target}.pkl` RF models.
    output_csv : str
        Where to write the combined predictions.
    """
    new_sequences = pd.read_csv(sequences_csv)
    names = new_sequences['Name'].tolist()

    results_df = pd.DataFrame({'Name': names}).set_index('Name')

    for info in models_info:
        target, esm_model, layer = info["Target"], info["ESMmodel"], info["Layer"]
        print(f"Predicting {target} using {esm_model} layer {layer}...")

        embedding_df = load_embeddings_for_target(embeddings_dir, target, esm_model, layer, names=names)
        X = align_sequences(embedding_df, names).values

        model_path = os.path.join(saved_models_dir, "plants", target, f"{target}.pkl")
        model = joblib.load(model_path)

        results_df[target] = model.predict(X)

    results_df = results_df.reset_index()
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, help="CSV with a 'Name' column of sequences to predict")
    parser.add_argument("--embeddings-dir", default="data/embeddings")
    parser.add_argument("--saved-models-dir", default="models/saved_models")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    predict_plants(args.sequences, args.embeddings_dir, args.saved_models_dir, args.output)
