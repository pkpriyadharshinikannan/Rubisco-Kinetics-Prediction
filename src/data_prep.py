"""
Data preprocessing and train/test splitting logic.

- train_test_split_general: used for models trained across all 9 phyla.
  Streptophyta and non-Streptophyta rows are split separately; minor
  non-Streptophyta phyla are grouped into an "Other" for stratified
  sampling.
- train_test_split_plants: used for plant-specific models. Trains/evaluates
  on Streptophyta rows only (this is simply Streptophyta filtered from the
  same general CSVs in data/processed/ -- see data/README.md).

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(embedding_df, input_df, target):
    """Merge an embedding DataFrame with kinetics data for a given target.

    Returns
    -------
    X : np.ndarray, embedding feature matrix
    y : np.ndarray, target kinetic values
    phylum : np.ndarray, phylum label per row (used for stratified splitting)
    """
    merged_df = (
        pd.merge(embedding_df, input_df[[target, 'Phylum']], left_index=True, right_index=True)
        .dropna(subset=[target])
    )
    X = merged_df.drop([target, 'Phylum'], axis=1).values
    y = merged_df[target].values
    phylum = merged_df['Phylum'].values
    return X, y, phylum


def train_test_split_general(X, y, phylum, test_size=0.1, random_state=42, minor_phylum_threshold=4):
    """Train/test split for the general (all-phyla) models.

    Streptophyta rows are split independently from non-Streptophyta rows.
    Non-Streptophyta rows are stratified by phylum, with phyla below
    `minor_phylum_threshold` occurrences grouped into a single "Other"
    bucket so that sklearn's stratify requirement (>=2 members per class)
    is more easily satisfied.
    """
    streptophyta_mask = (phylum == 'Streptophyta')

    # Streptophyta split
    X_strep, y_strep, phylum_strep = X[streptophyta_mask], y[streptophyta_mask], phylum[streptophyta_mask]
    (X_train_strep, X_test_strep,
     y_train_strep, y_test_strep,
     phylum_train_strep, phylum_test_strep) = train_test_split(
        X_strep, y_strep, phylum_strep, test_size=test_size, random_state=random_state
    )

    # Non-Streptophyta split, stratified with minor phyla grouped into "Other"
    X_non_strep, y_non_strep, phylum_non_strep = (
        X[~streptophyta_mask], y[~streptophyta_mask], phylum[~streptophyta_mask]
    )
    phylum_counts = pd.Series(phylum_non_strep).value_counts()
    phylum_grouped = np.where(
        phylum_counts[phylum_non_strep].values > minor_phylum_threshold, phylum_non_strep, 'Other'
    )
    (X_train_non_strep, X_test_non_strep,
     y_train_non_strep, y_test_non_strep,
     phylum_train_non_strep, phylum_test_non_strep) = train_test_split(
        X_non_strep, y_non_strep, phylum_non_strep,
        test_size=test_size, random_state=random_state, stratify=phylum_grouped
    )

    X_train = np.concatenate([X_train_strep, X_train_non_strep])
    X_test = np.concatenate([X_test_strep, X_test_non_strep])
    y_train = np.concatenate([y_train_strep, y_train_non_strep])
    y_test = np.concatenate([y_test_strep, y_test_non_strep])
    phylum_train = np.concatenate([phylum_train_strep, phylum_train_non_strep])
    phylum_test = np.concatenate([phylum_test_strep, phylum_test_non_strep])

    return X_train, X_test, y_train, y_test, phylum_train, phylum_test


def train_test_split_plants(X, y, phylum, test_size=0.1, random_state=42):
    """Train/test split for plant-specific models: Streptophyta rows only."""
    streptophyta_mask = (phylum == 'Streptophyta')
    X_strep, y_strep, phylum_strep = X[streptophyta_mask], y[streptophyta_mask], phylum[streptophyta_mask]
    return train_test_split(
        X_strep, y_strep, phylum_strep, test_size=test_size, random_state=random_state
    )
