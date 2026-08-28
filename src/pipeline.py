"""
Training/evaluation pipeline: hyperparameter search, train/test evaluation,
and Streptophyta-specific metric reporting.

Ported from the four original scripts (gpmodel_general.py, gpmodel_plants.py,
rfmodel_general.py, rfmodel_plants.py). Two structural changes were made
during this refactor, both purely organizational (no computation changed):

1. `target`, `results_dir`, and `embeddings_dir` were implicit module-level
   globals in the original scripts (referenced inside functions via Python
   closures over the enclosing for-loop, rather than passed as arguments).
   They are now explicit function parameters, which is necessary for these
   functions to work correctly when imported as a reusable module rather
   than run top-to-bottom as a single script.
2. GP and RF versions of run_randomsearchcv / train_and_test_all_models are
   kept as separate, clearly named functions (rather than one function
   branching on model type), since their internals genuinely differ
   (GPFlowModel vs. RandomForestRegressor) and forcing a shared interface
   would add complexity without a real benefit.

KNOWN LIMITATION (flagged, not yet fixed): after `embedding_df.reindex(input_df.index)`,
any species missing from an embedding CSV becomes a row of NaNs. The original
scripts detect and print this ("NaN values in embedding_df: ...") but do not
drop or impute those rows before calling .fit(). Random Forest will raise an
error on NaN input; GPflow will silently propagate NaNs. This has been
preserved as a documented open item pending a decision, not silently fixed.
"""

import os

import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
import numpy as np

from src.data_prep import preprocess
from src.utils import correlation_scorer, calculate_metrics, geometric_mean
from src.models.gp_model import GPFlowModel
from src.models.rf_model import train_rf_model, evaluate_rf_model


def _check_for_missing_embeddings(embedding_df, input_df):
    """Print (but do not act on) NaN/missing-index warnings after reindexing.

    See module docstring "KNOWN LIMITATION" note above.
    """
    null_counts = embedding_df.isnull().sum()
    if null_counts.sum() > 0:
        print("NaN values in embedding_df:", null_counts[null_counts > 0])

    missing_index = input_df.index[~input_df.index.isin(embedding_df.index)]
    if not missing_index.empty:
        print("Missing indices in embedding_df:", missing_index)


# ---------------------------------------------------------------------------
# Gaussian Process pipeline
# ---------------------------------------------------------------------------

def run_randomsearchcv_gp(target, models, input_df, results_dir, embeddings_dir, kernel, split_fn):
    """Run RandomizedSearchCV over ESM checkpoints/layers for a GP model.

    Parameters
    ----------
    split_fn : callable
        Either data_prep.train_test_split_general or
        data_prep.train_test_split_plants, selected by the caller.
    """
    best_params_cv_scores = {model: {} for model in models.keys()}

    for model, n_layers in models.items():
        print(f"Processing model: {model}")

        for layer in range(0, n_layers + 1):
            print(layer, end=" ")

            file_path = os.path.join(embeddings_dir, f"{target}_embeddings", f"{model}_{layer}.csv")
            embedding_df = pd.read_csv(file_path).set_index('Name')
            embedding_df = embedding_df.reindex(input_df.index)
            _check_for_missing_embeddings(embedding_df, input_df)

            X, y, phylum = preprocess(embedding_df, input_df, target)
            X_train, X_test, y_train, y_test, phylum_train, phylum_test = split_fn(X, y, phylum)

            scoring = {
                'R2': make_scorer(r2_score),
                'MAE': make_scorer(mean_absolute_error),
                'MSE': make_scorer(mean_squared_error),
                'RMSE': make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))),
                'Correlation': make_scorer(correlation_scorer, greater_is_better=True),
            }
            cv = KFold(n_splits=10, shuffle=True, random_state=42)

            model_gpflow = GPFlowModel(kernel=kernel)
            param_grid = {
                'lengthscale': np.logspace(-2, 2, 15),
                'variance': np.logspace(-2, 2, 15),
            }

            rfsearch = RandomizedSearchCV(
                model_gpflow, param_distributions=param_grid, n_iter=100, cv=cv,
                scoring=scoring, refit='Correlation', return_train_score=True,
            )
            rfsearch.fit(X_train, y_train)

            results_df = pd.DataFrame(rfsearch.cv_results_)
            results_df.to_csv(f'{results_dir}/{target}_{model}_{layer}_randomsearch_results.csv', index=False)

            best_params = rfsearch.best_params_
            best_params_cv_scores[model][layer] = {
                'params': best_params,
                'MAE': rfsearch.cv_results_['mean_test_MAE'][rfsearch.best_index_],
                'MSE': rfsearch.cv_results_['mean_test_MSE'][rfsearch.best_index_],
                'RMSE': rfsearch.cv_results_['mean_test_RMSE'][rfsearch.best_index_],
                'R2': rfsearch.cv_results_['mean_test_R2'][rfsearch.best_index_],
                'Correlation': rfsearch.cv_results_['mean_test_Correlation'][rfsearch.best_index_],
            }

            # Cross-validation actual vs predicted, with Phylum, for plotting
            y_cv_actual, y_cv_predicted, phylum_cv = [], [], []
            for train_index, test_index in cv.split(X_train):
                X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
                y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]
                phylum_cv_test = phylum_train[test_index]

                model_gpflow.set_params(**best_params).fit(X_cv_train, y_cv_train)
                y_pred_cv = model_gpflow.predict(X_cv_test)

                y_cv_actual.extend(y_cv_test)
                y_cv_predicted.extend(y_pred_cv)
                phylum_cv.extend(phylum_cv_test)

            cv_pred_df = pd.DataFrame({'Actual': y_cv_actual, 'Predicted': y_cv_predicted, 'Phylum': phylum_cv})
            cv_pred_df.to_csv(f'{results_dir}/cv_AP/{target}_{model}_{layer}_cv_actual_vs_predicted.csv', index=False)

    best_params_df = pd.DataFrame({
        'Model': [m for m in best_params_cv_scores for _ in best_params_cv_scores[m]],
        'Layer': [l for m in best_params_cv_scores for l in best_params_cv_scores[m]],
        'Params': [best_params_cv_scores[m][l]['params'] for m in best_params_cv_scores for l in best_params_cv_scores[m]],
        'CV MAE': [best_params_cv_scores[m][l]['MAE'] for m in best_params_cv_scores for l in best_params_cv_scores[m]],
        'CV MSE': [best_params_cv_scores[m][l]['MSE'] for m in best_params_cv_scores for l in best_params_cv_scores[m]],
        'CV RMSE': [best_params_cv_scores[m][l]['RMSE'] for m in best_params_cv_scores for l in best_params_cv_scores[m]],
        'CV R2': [best_params_cv_scores[m][l]['R2'] for m in best_params_cv_scores for l in best_params_cv_scores[m]],
        'CV Correlation': [best_params_cv_scores[m][l]['Correlation'] for m in best_params_cv_scores for l in best_params_cv_scores[m]],
    })
    best_params_df.to_csv(f"{results_dir}/best_params_cv_scores.csv", index=False)

    return best_params_cv_scores, best_params_df


def train_and_test_all_models_gp(models, input_df, target, kernel, best_params_cv_scores,
                                  results_dir, embeddings_dir, split_fn):
    """Train final GP models with best params and evaluate on held-out train/test split."""
    from src.models.gp_model import train_gp_model, predict_gp_model

    all_train_test_results = []

    for model, layers in best_params_cv_scores.items():
        for layer, details in layers.items():
            best_params = details['params']

            file_path = os.path.join(embeddings_dir, f"{target}_embeddings", f"{model}_{layer}.csv")
            embedding_df = pd.read_csv(file_path).set_index('Name')
            embedding_df = embedding_df.reindex(input_df.index)

            X, y, phylum = preprocess(embedding_df, input_df, target)
            X_train, X_test, Y_train, Y_test, phylum_train, phylum_test = split_fn(X, y, phylum)

            model_gp = train_gp_model(X_train, Y_train, kernel, best_params.get('lengthscale'), best_params.get('variance'))

            y_train_pred, _ = predict_gp_model(model_gp, X_train)
            y_test_pred, _ = predict_gp_model(model_gp, X_test)

            train_metrics = calculate_metrics(Y_train, y_train_pred)
            test_metrics = calculate_metrics(Y_test, y_test_pred)

            all_train_test_results.append({
                "Model": model, "Layer": layer, "Params": best_params,
                "Train MAE": train_metrics['MAE'], "Train MSE": train_metrics['MSE'],
                "Train RMSE": train_metrics['RMSE'], "Train R2": train_metrics['R2'],
                "Train Correlation": train_metrics['Correlation'],
                "Test MAE": test_metrics['MAE'], "Test MSE": test_metrics['MSE'],
                "Test RMSE": test_metrics['RMSE'], "Test R2": test_metrics['R2'],
                "Test Correlation": test_metrics['Correlation'],
            })

            pd.DataFrame({'Actual': Y_train, 'Predicted': y_train_pred, 'Phylum': phylum_train}) \
                .to_csv(f"{results_dir}/train_AP/{model}_{layer}_train_actual_vs_predicted.csv", index=False)
            pd.DataFrame({'Actual': Y_test, 'Predicted': y_test_pred, 'Phylum': phylum_test}) \
                .to_csv(f"{results_dir}/test_AP/{model}_{layer}_test_actual_vs_predicted.csv", index=False)

    return pd.DataFrame(all_train_test_results)


# ---------------------------------------------------------------------------
# Random Forest pipeline
# ---------------------------------------------------------------------------

def run_randomsearchcv_rf(target, models, input_df, results_dir, embeddings_dir, split_fn):
    from src.models.rf_model import random_search_cv

    best_params_cv_scores = {model: {} for model in models.keys()}

    for model, n_layers in models.items():
        print(f"Processing model: {model}")
        for layer in range(0, n_layers + 1):
            print(layer, end=" ")

            file_path = os.path.join(embeddings_dir, f"{target}_embeddings", f"{model}_{layer}.csv")
            embedding_df = pd.read_csv(file_path).set_index('Name')
            embedding_df = embedding_df.reindex(input_df.index)

            X, y, phylum = preprocess(embedding_df, input_df, target)
            X_train, X_test, y_train, y_test, phylum_train, phylum_test = split_fn(X, y, phylum)

            rfsearch = random_search_cv(X_train, y_train)
            best_params = rfsearch.best_params_

            best_params_cv_scores[model][layer] = {
                'Params': best_params,
                'MAE': rfsearch.cv_results_['mean_test_MAE'][rfsearch.best_index_],
                'MSE': rfsearch.cv_results_['mean_test_MSE'][rfsearch.best_index_],
                'RMSE': rfsearch.cv_results_['mean_test_RMSE'][rfsearch.best_index_],
                'R2': rfsearch.cv_results_['mean_test_R2'][rfsearch.best_index_],
                'Correlation': rfsearch.cv_results_['mean_test_Correlation'][rfsearch.best_index_],
            }

            cv = KFold(n_splits=10, shuffle=True, random_state=42)
            y_cv_actual, y_cv_predicted, phylum_cv = [], [], []
            for train_index, test_index in cv.split(X_train):
                X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
                y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]
                phylum_cv_test = phylum_train[test_index]

                model_rf = train_rf_model(X_cv_train, y_cv_train, **best_params)
                y_pred_cv = model_rf.predict(X_cv_test)

                y_cv_actual.extend(y_cv_test)
                y_cv_predicted.extend(y_pred_cv)
                phylum_cv.extend(phylum_cv_test)

            pd.DataFrame({'Actual': y_cv_actual, 'Predicted': y_cv_predicted, 'Phylum': phylum_cv}) \
                .to_csv(f'{results_dir}/cv_AP/{target}_{model}_{layer}_cv_actual_vs_predicted.csv', index=False)

    all_best_params = [
        {
            'Model': model, 'Layer': layer, 'Params': details['Params'],
            'CV MAE': details['MAE'], 'CV MSE': details['MSE'], 'CV RMSE': details['RMSE'],
            'CV R2': details['R2'], 'CV Correlation': details['Correlation'],
        }
        for model, layers in best_params_cv_scores.items()
        for layer, details in layers.items()
    ]
    best_params_df = pd.DataFrame(all_best_params)
    best_params_df.to_csv(f"{results_dir}/best_params_cv_scores.csv", index=False)

    return best_params_cv_scores, best_params_df


def train_and_test_all_models_rf(models, input_df, target, best_params_cv_scores, results_dir, embeddings_dir, split_fn):
    all_train_test_results = []

    for model, layers in best_params_cv_scores.items():
        for layer, details in layers.items():
            print(f"Training and evaluating model {model} at layer {layer}")

            file_path = os.path.join(embeddings_dir, f"{target}_embeddings", f"{model}_{layer}.csv")
            embedding_df = pd.read_csv(file_path).set_index('Name')
            embedding_df = embedding_df.reindex(input_df.index)

            X, y, phylum = preprocess(embedding_df, input_df, target)
            X_train, X_test, Y_train, Y_test, phylum_train, phylum_test = split_fn(X, y, phylum)

            model_rf = train_rf_model(X_train, Y_train, **details['Params'])
            train_metrics, train_pred = evaluate_rf_model(model_rf, X_train, Y_train)
            test_metrics, test_pred = evaluate_rf_model(model_rf, X_test, Y_test)

            all_train_test_results.append({
                "Model": model, "Layer": layer, "Params": details['Params'],
                "Train MAE": train_metrics['MAE'], "Train MSE": train_metrics['MSE'],
                "Train RMSE": train_metrics['RMSE'], "Train R2": train_metrics['R2'],
                "Train Correlation": train_metrics['Correlation'],
                "Test MAE": test_metrics['MAE'], "Test MSE": test_metrics['MSE'],
                "Test RMSE": test_metrics['RMSE'], "Test R2": test_metrics['R2'],
                "Test Correlation": test_metrics['Correlation'],
            })

            pd.DataFrame({'Actual': Y_train, 'Predicted': train_pred, 'Phylum': phylum_train}) \
                .to_csv(f"{results_dir}/train_AP/{model}_{layer}_train_actual_vs_predicted.csv", index=False)
            pd.DataFrame({'Actual': Y_test, 'Predicted': test_pred, 'Phylum': phylum_test}) \
                .to_csv(f"{results_dir}/test_AP/{model}_{layer}_test_actual_vs_predicted.csv", index=False)

    return pd.DataFrame(all_train_test_results)


# ---------------------------------------------------------------------------
# Shared: Streptophyta-specific metrics (general-model runs only)
# ---------------------------------------------------------------------------

def add_streptophyta_metrics(layer_df, cv_file_path, test_file_path):
    """Add Streptophyta-only MAE/MSE/RMSE/R2/Correlation columns to a results row.

    Only meaningful for the *general* model runs, where the training set
    spans multiple phyla and it's informative to see performance on the
    Streptophyta subset specifically. Not called for plant-specific model
    runs, since those are already trained and evaluated on Streptophyta
    exclusively -- computing a "Streptophyta-specific" metric there would
    just reproduce the overall metric.
    """
    def calculate_streptophyta_metrics(file_path):
        df = pd.read_csv(file_path)
        strep_df = df[df['Phylum'] == 'Streptophyta']
        return calculate_metrics(strep_df['Actual'].values, strep_df['Predicted'].values)

    cv_metrics = calculate_streptophyta_metrics(cv_file_path)
    test_metrics = calculate_streptophyta_metrics(test_file_path)

    for metric, value in cv_metrics.items():
        layer_df[f'CV Strep {metric}'] = value
    for metric, value in test_metrics.items():
        layer_df[f'Test Strep {metric}'] = value

    layer_df['Strep Score'] = geometric_mean(cv_metrics['Correlation'], test_metrics['Correlation'])
