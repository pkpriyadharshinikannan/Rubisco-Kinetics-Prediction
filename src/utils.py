"""
Shared utility functions used by both the Gaussian Process and Random Forest
training pipelines (src/models/gp_model.py, src/models/rf_model.py,
src/pipeline.py).
"""

import os
import random

import numpy as np
from scipy.stats import pearsonr, gmean
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed=42):
    """Set Python/NumPy/hash seeds for reproducibility.

    Note: TensorFlow/GPflow-specific seeding (tf.random.set_seed) and GPU
    device selection (CUDA_VISIBLE_DEVICES) are handled separately in
    src/models/gp_model.py, since the Random Forest pipeline has no
    TensorFlow dependency at all.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def time_taken(start_time, end_time):
    """Return (hours, minutes, seconds) elapsed between two time.time() calls."""
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def calculate_metrics(y_true, y_pred):
    """Compute MAE, MSE, RMSE, R2, and Pearson correlation (each rounded to 3 dp)."""
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 3),
        'MSE': round(mean_squared_error(y_true, y_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        'R2': round(r2_score(y_true, y_pred), 3),
        'Correlation': round(pearsonr(y_true, y_pred)[0], 3),
    }


def correlation_scorer(y_true, y_pred):
    """Pearson correlation coefficient; used as a custom scorer in RandomizedSearchCV."""
    return pearsonr(y_true, y_pred)[0]


def geometric_mean(a, b):
    """Geometric mean of two values (used to combine CV and test correlation into one Score)."""
    return round(gmean([a, b]), 3)
