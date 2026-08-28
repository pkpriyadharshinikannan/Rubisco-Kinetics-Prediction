"""
Shared utilities for loading ESM embedding CSVs at prediction time.

Uses the same CSV convention as the training pipeline (see src/pipeline.py,
data/README.md): one file per (target, ESM checkpoint, layer), located at
`{embeddings_dir}/{target}_embeddings/{esm_model}_{layer}.csv`, indexed by
the `Name` column.

This replaces the original scripts' per-sequence .pt file loading
(`torch.load(f"{pt_filepath}/{name}.pt")`), which is no longer used here --
see predict_general.py / predict_plants.py module docstrings for why.
"""

import os

import pandas as pd


def load_embeddings_for_target(embeddings_dir, target, esm_model, layer, names=None, chunksize=5000):
    """Load an ESM embedding CSV for one (target, ESM checkpoint, layer) combination.

    Parameters
    ----------
    names : list-like, optional
        If provided, the CSV is read in chunks and filtered to only these
        sequence names as it goes, so peak memory stays proportional to the
        number of *requested* sequences rather than the full embedding
        file. This matters for large mutagenesis scans: e.g. a double-mutant
        set can be 300,000+ rows, and at higher-dimensional ESM checkpoints
        (esm2_t36_3B_UR50D, 2560-dim) loading the full CSV at once could
        require several GB of memory. If `names` is None, the full file is
        loaded (fine for smaller sequence sets).
    chunksize : int
        Rows per chunk when `names` is provided.
    """
    file_path = os.path.join(embeddings_dir, f"{target}_embeddings", f"{esm_model}_{layer}.csv")

    if names is None:
        return pd.read_csv(file_path).set_index('Name')

    names_set = set(names)
    matched_chunks = [
        chunk[chunk['Name'].isin(names_set)]
        for chunk in pd.read_csv(file_path, chunksize=chunksize)
    ]
    matched = pd.concat(matched_chunks) if matched_chunks else pd.DataFrame(columns=['Name'])
    return matched.set_index('Name')


def align_sequences(embedding_df, sequence_names):
    """Reindex an embedding DataFrame to a specific list of sequence names.

    Reports (but does not silently drop) any sequences missing from the
    embedding file -- same "warn, don't act" pattern used during training
    (see src/pipeline.py `_check_for_missing_embeddings`), preserved here
    for consistency rather than fixed differently in two places.
    """
    aligned = embedding_df.reindex(sequence_names)
    missing = aligned[aligned.isnull().any(axis=1)].index.tolist()
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        print(f"Warning: {len(missing)} sequence(s) not found in embeddings: {preview}{suffix}")
    return aligned
