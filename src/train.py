"""
Train and compare multiple churn models, logging every run to MLflow.

Evaluation protocol (no test-set leakage):
  1. Every candidate is scored with stratified K-fold cross-validation on the
     TRAIN split only. SMOTE lives inside the CV pipeline, so it is fit on the
     training folds only.
  2. The best candidate is picked by a CV metric (default: F1, see --select-by).
  3. That one candidate is refit on the full train split and evaluated ONCE on
     the held-out test split. Those are the numbers to report.

Run:
    mlflow ui  &            # optional, to watch runs live at localhost:5000
    python src/train.py
    python src/train.py --select-by recall
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

PROCESSED_DIR = Path("data/processed")
METRICS_PATH = Path("metrics.json")
EXPERIMENT_NAME = "churn-prediction"
CV_FOLDS = 5
SEED = 42
SCORING = ["accuracy", "f1", "precision", "recall", "roc_auc"]


def load_splits():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]
    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def build_candidates(y_train) -> dict:
    """name -> (unfitted estimator, params to log). Uses only training labels."""
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb_kwargs = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                      eval_metric="logloss", random_state=SEED)
    return {
        "logreg_baseline": (
            LogisticRegression(max_iter=1000),
            {"model": "LogisticRegression", "imbalance_strategy": "none"},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED),
            {"model": "RandomForest", "n_estimators": 300, "max_depth": 8,
             "imbalance_strategy": "none"},
        ),
        "xgboost_class_weight": (
            xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **xgb_kwargs),
            {"model": "XGBoost", "imbalance_strategy": "class_weight"},
        ),
        # SMOTE inside a pipeline => resampling happens per CV training fold only.
        "xgboost_smote": (
            ImbPipeline([
                ("smote", SMOTE(random_state=SEED)),
                ("model", xgb.XGBClassifier(**xgb_kwargs)),
            ]),
            {"model": "XGBoost", "imbalance_strategy": "SMOTE"},
        ),
        "lightgbm_class_weight": (
            lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                               class_weight="balanced", random_state=SEED, verbose=-1),
            {"model": "LightGBM", "imbalance_strategy": "class_weight"},
        ),
    }


def cross_validate_candidate(estimator, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_validate(clone(estimator), X_train, y_train, cv=cv, scoring=SCORING)
    return {m: float(np.mean(scores[f"test_{m}"])) for m in SCORING}


def select_best(cv_results: dict, select_by: str) -> str:
    """Name of the candidate with the highest mean CV score on `select_by`."""
    return max(cv_results, key=lambda name: cv_results[name][select_by])


def log_model(model):
    if isinstance(model, xgb.XGBClassifier):
        mlflow.xgboost.log_model(model, name="model")
    elif isinstance(model, lgb.LGBMClassifier):
        mlflow.lightgbm.log_model(model, name="model")
    else:  # sklearn estimators and imblearn pipelines
        mlflow.sklearn.log_model(model, name="model")


def main(select_by: str = "f1"):
    X_train, y_train, X_test, y_test = load_splits()
    mlflow.set_experiment(EXPERIMENT_NAME)
    candidates = build_candidates(y_train)

    # --- 1. Model selection: cross-validation on TRAIN only ---------------
    cv_results = {}
    for name, (estimator, params) in candidates.items():
        with mlflow.start_run(run_name=f"cv_{name}"):
            mlflow.set_tag("stage", "cross_validation")
            summary = cross_validate_candidate(estimator, X_train, y_train)
            mlflow.log_params({**params, "cv_folds": CV_FOLDS})
            mlflow.log_metrics({f"cv_{k}": v for k, v in summary.items()})
        cv_results[name] = summary
        print(f"{name:25s} | CV " + " | ".join(f"{k}={v:.3f}" for k, v in summary.items()))

    best = select_best(cv_results, select_by)
    print(f"\nSelected by CV {select_by}: {best}")

    # --- 2. Final evaluation: refit once, touch the test set once ---------
    estimator, params = candidates[best]
    final_model = clone(estimator).fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, y_pred, y_proba)

    with mlflow.start_run(run_name=f"final_{best}"):
        mlflow.set_tag("stage", "final_test_evaluation")
        mlflow.log_params({**params, "selected_by": select_by})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        log_model(final_model)

    METRICS_PATH.write_text(json.dumps({
        "selected_model": best,
        "selection_metric": select_by,
        "cv": cv_results,
        "test": test_metrics,
    }, indent=2))

    print("\n=== Held-out test metrics (selected model, evaluated once) ===")
    print(" | ".join(f"{k}={v:.3f}" for k, v in test_metrics.items()))
    print(f"Wrote {METRICS_PATH}")
    print(
        "\nNext step: open the MLflow UI (`mlflow ui`), check the final_* run, "
        "and promote its model to the Model Registry as 'production'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--select-by", default="f1", choices=SCORING,
                        help="CV metric used to pick the model (default: f1)")
    main(parser.parse_args().select_by)
