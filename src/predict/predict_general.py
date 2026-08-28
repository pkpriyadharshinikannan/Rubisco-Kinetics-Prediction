"""
Batch prediction using the finalized *general* models (trained across all 9
phyla; Gaussian Process, per thesis Table 3).

REDESIGNED from the original general_prediction_parallel.py -- not a
preserved-as-is port. What changed and why:

1. Embeddings are now read from CSV (matching the training pipeline's
   convention, per project decision that both training and prediction take
   CSV input), rather than per-sequence .pt files via torch.load.
2. Because embeddings are loaded as a batch (see
   src/predict/predict_utils.load_embeddings_for_target) and both GPflow and
   scikit-learn support vectorized prediction over many rows in a single
   call, this replaces the original per-sequence joblib.Parallel loop
   entirely. There is no longer a need to reload the model once per
   sequence -- it's loaded once per target and predicts the whole batch at
   once.
3. The original script reloaded the TensorFlow SavedModel *inside* the
   per-sequence function specifically because joblib's default 'loky'
   backend runs workers as separate processes, and GPflow/TF model objects
   generally don't pickle cleanly across process boundaries. Removing the
   per-sequence parallel loop removes the need for this workaround too.

Known limitation carried over from predict_utils.align_sequences: any
requested sequence missing from the embeddings CSV will produce a NaN
prediction row, with a printed warning -- not silently dropped, not yet
auto-handled.
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import absl.logging

from src.models.gp_model import set_seed_gp
from src.predict.predict_utils import align_sequences, load_embeddings_for_target

absl.logging.set_verbosity('error')

GENERAL_MODELS_INFO = [
    {"Target": "Specificity", "ESMmodel": "esm2_t33_650M_UR50D", "Layer": 33},
    {"Target": "kcatC", "ESMmodel": "esm2_t12_35M_UR50D", "Layer": 7},
    {"Target": "KC", "ESMmodel": "esm2_t30_150M_UR50D", "Layer": 26},
]


def predict_with_gp_model(model, X):
    """Predict mean and variance from a GPflow model exported as a TF SavedModel."""
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float64)
    predict_fn = model.signatures["serving_default"]
    result = predict_fn(Xnew=X_tensor)
    mean = result["output_0"].numpy()
    variance = result["output_1"].numpy()
    return mean.flatten(), variance.flatten()


def predict_general(sequences_csv, embeddings_dir, saved_models_dir, output_csv,
                     models_info=GENERAL_MODELS_INFO):
    """Predict Sc/o, kcat, and Km for new sequences using the saved general models.

    Parameters
    ----------
    sequences_csv : str
        CSV with a 'Name' column identifying which sequences to predict for.
    embeddings_dir : str
        Base directory containing `{target}_embeddings/{esm_model}_{layer}.csv`
        (same convention as training; see data/README.md).
    saved_models_dir : str
        Base directory containing `general/{target}/` GPflow SavedModel dirs.
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

        model_path = os.path.join(saved_models_dir, "general", target)
        model = tf.saved_model.load(model_path)

        mean, variance = predict_with_gp_model(model, X)
        results_df[target] = mean
        results_df[f"{target}_uncertainty"] = variance

    results_df = results_df.reset_index()
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")
    return results_df


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU-only inference, matching original script
    set_seed_gp(seed=42, cuda_visible_devices="-1")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, help="CSV with a 'Name' column of sequences to predict")
    parser.add_argument("--embeddings-dir", default="data/embeddings")
    parser.add_argument("--saved-models-dir", default="models/saved_models")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    predict_general(args.sequences, args.embeddings_dir, args.saved_models_dir, args.output)
