"""
Random Forest model training, prediction, and hyperparameter search.

"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
import numpy as np

from src.utils import correlation_scorer


def random_search_cv(X_train, y_train, random_state=42):
    """Randomized hyperparameter search for RandomForestRegressor.

    """
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    }

    rf = RandomForestRegressor(random_state=random_state)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        'R2': make_scorer(r2_score),
        'MAE': make_scorer(mean_absolute_error),
        'MSE': make_scorer(mean_squared_error),
        'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
        'Correlation': make_scorer(correlation_scorer, greater_is_better=True),
    }

    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=100,
        cv=cv,
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring,
        refit='Correlation',
        return_train_score=True,
    )
    rf_random.fit(X_train, y_train)
    return rf_random


def train_rf_model(X_train, Y_train, **params):
    """Train a RandomForestRegressor with the given hyperparameters."""
    rf = RandomForestRegressor(**params, random_state=42)
    rf.fit(X_train, Y_train)
    return rf


def evaluate_rf_model(model, X_test, Y_test):
    """Predict and compute metrics for a trained RF model."""
    from src.utils import calculate_metrics
    Y_pred = model.predict(X_test)
    return calculate_metrics(Y_test, Y_pred), Y_pred
