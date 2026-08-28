"""
Gaussian Process model (GPflow) wrapper, training, and prediction functions.
"""

import os

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import to_default_float
from sklearn.base import BaseEstimator, RegressorMixin

gpflow.config.set_default_float(np.float64)


def set_seed_gp(seed=42, cuda_visible_devices=None):
    """Set TensorFlow seed and (optionally) restrict visible GPU devices."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    tf.random.set_seed(seed)


class GPFlowModel(BaseEstimator, RegressorMixin):

    def __init__(self, kernel, lengthscale=1.0, variance=1.0):
        self.kernel = kernel
        if hasattr(self.kernel, 'lengthscales'):
            self.kernel.lengthscales = lengthscale
        if hasattr(self.kernel, 'variance'):
            self.kernel.variance = variance
        self.model = None

    def fit(self, X, y):
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        self.model = gpflow.models.GPR(
            data=(to_default_float(X), to_default_float(y[:, None])),
            kernel=self.kernel,
        )
        return self

    def predict(self, X):
        X = X.astype(np.float64)
        y_pred, _ = self.model.predict_f(X)
        return y_pred.numpy().flatten()

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)

            if param == "lengthscale" and hasattr(self.kernel, 'lengthscales'):
                if isinstance(self.kernel.lengthscales, tf.Tensor):
                    self.kernel.lengthscales.assign(value)
                else:
                    self.kernel.lengthscales = value

            if param == "variance" and hasattr(self.kernel, 'variance'):
                if isinstance(self.kernel.variance, tf.Tensor):
                    self.kernel.variance.assign(value)
                else:
                    self.kernel.variance = value

        return self

    def get_params(self, deep=True):
        params = {"kernel": self.kernel, "lengthscale": None, "variance": None}

        if hasattr(self.kernel, 'lengthscales'):
            if isinstance(self.kernel.lengthscales, tf.Tensor):
                params["lengthscale"] = self.kernel.lengthscales.numpy()
            else:
                params["lengthscale"] = self.kernel.lengthscales

        if hasattr(self.kernel, 'variance'):
            if isinstance(self.kernel.variance, tf.Tensor):
                params["variance"] = self.kernel.variance.numpy()
            else:
                params["variance"] = self.kernel.variance

        return params


def train_gp_model(X_train, Y_train, kernel, lengthscale=None, variance=None):
    """Train a GPR model directly (used for final train/test evaluation
    after the best hyperparameters have been selected via search)."""
    if lengthscale and variance:
        if hasattr(kernel, 'lengthscales'):
            kernel.lengthscales = lengthscale
        if hasattr(kernel, 'variance'):
            kernel.variance = variance

    model = gpflow.models.GPR(
        data=(to_default_float(X_train), to_default_float(Y_train[:, None])),
        kernel=kernel,
    )
    return model


def predict_gp_model(model, X):
    """Predict mean and standard deviation from a trained GPR model."""
    Y_pred, Y_var = model.predict_f(X)
    return Y_pred.numpy().flatten(), np.sqrt(Y_var.numpy().flatten())
