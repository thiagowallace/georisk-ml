"""Fit preliminary baselines. Random holdout does not control spatial autocorrelation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.evaluate_baselines import (
    EVALUATION_NOTICE, OUTPUT_DIR, THRESHOLD, evaluate_predictions, save_evaluation_figures,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/santa_tereza_ml_dataset.csv"
CONTINUOUS = ["elevation", "slope", "aspect_sin", "aspect_cos", "plan_curvature",
              "profile_curvature", "ndvi_pre_event"]
CATEGORICAL = ["land_cover"]
FEATURES = CONTINUOUS + CATEGORICAL
FORBIDDEN = ["aspect", "x", "y", "row", "col", "sample_id", "cell_id", "landslide_count",
             "landslide_ids", "sample_type", "target"]
SEED = 42
TEST_SIZE = 0.25
LOGISTIC_PARAMS = {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 5000,
                   "random_state": SEED, "class_weight": None}
FOREST_PARAMS = {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 5,
                 "min_samples_split": 10, "max_features": "sqrt", "bootstrap": True,
                 "random_state": SEED, "n_jobs": 1, "class_weight": None}
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "average_precision"]


def select_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = sorted(set(FEATURES + ["target"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty or not frame.index.is_unique:
        raise ValueError("Dataset must be nonempty and have a unique row index.")
    X = frame[FEATURES].apply(pd.to_numeric, errors="raise").copy()
    y = pd.to_numeric(frame.target, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Predictors and target must be finite and non-null.")
    if set(y.unique()) != {0, 1} or y.value_counts().min() < 2:
        raise ValueError("Target requires both classes 0 and 1, with at least two samples each.")
    if X[CONTINUOUS].eq(-9999).any().any():
        raise ValueError("Continuous features contain NoData.")
    if X.land_cover.le(0).any() or not np.equal(X.land_cover, np.floor(X.land_cover)).all():
        raise ValueError("land_cover must preserve positive integer categorical codes.")
    X["land_cover"] = X.land_cover.astype(np.int64)
    if (not X.ndvi_pre_event.between(-1, 1).all() or not X.slope.between(0, 90).all()
            or not X.aspect_sin.between(-1, 1).all() or not X.aspect_cos.between(-1, 1).all()):
        raise ValueError("Features outside their physical bounds.")
    if not np.allclose(X.aspect_sin**2 + X.aspect_cos**2, 1, atol=1e-8, rtol=0):
        raise ValueError("Inconsistent circular aspect components.")
    return X, y.astype(np.int64)


def split_dataset(X: pd.DataFrame, y: pd.Series):
    """Preliminary only: no control of spatial autocorrelation; replaced in Stage 4."""
    if list(X.columns) != FEATURES or not X.index.equals(y.index):
        raise ValueError("Split requires the exact feature contract and aligned label indices.")
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def make_pipelines() -> dict[str, Pipeline]:
    """Return independent, unfitted transformers; all learning happens on training rows."""
    pipelines = {}
    for name in ("logistic", "random_forest"):
        preprocessor = ColumnTransformer(
            [("continuous", StandardScaler() if name == "logistic" else "passthrough", CONTINUOUS),
             ("land_cover", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
            remainder="drop", verbose_feature_names_out=True,
        )
        classifier = (LogisticRegression(**LOGISTIC_PARAMS) if name == "logistic"
                      else RandomForestClassifier(**FOREST_PARAMS))
        pipelines[name] = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    return pipelines


def fit_baselines(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, dict]:
    if list(X_train.columns) != FEATURES or not X_train.index.equals(y_train.index):
        raise ValueError("Training requires only approved predictors and aligned labels.")
    pipelines, diagnostics = make_pipelines(), {}
    for name, pipeline in pipelines.items():
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipeline.fit(X_train, y_train)
        diagnostics[name] = {
            "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in captured],
            "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in captured),
        }
        if name == "logistic":
            diagnostics[name]["iterations"] = pipeline.named_steps["classifier"].n_iter_.tolist()
    return pipelines, diagnostics


def positive_probabilities(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if list(X.columns) != FEATURES:
        raise ValueError("Prediction requires exactly the approved predictors.")
    classes = pipeline.named_steps["classifier"].classes_
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("Model has no unique positive class 1.")
    probabilities = pipeline.predict_proba(X)[:, positive[0]]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid predicted probabilities.")
    return probabilities


def interpretation_tables(pipelines: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    logistic, forest = pipelines["logistic"], pipelines["random_forest"]
    coefficients = logistic.named_steps["classifier"].coef_[0]
    logistic_table = pd.DataFrame({
        "feature": logistic.named_steps["preprocessor"].get_feature_names_out(),
        "coefficient": coefficients, "absolute_coefficient": np.abs(coefficients),
    }).sort_values(["absolute_coefficient", "feature"], ascending=[False, True]).reset_index(drop=True)
    forest_table = pd.DataFrame({
        "feature": forest.named_steps["preprocessor"].get_feature_names_out(),
        "importance": forest.named_steps["classifier"].feature_importances_,
    }).sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    return logistic_table, forest_table


def train_and_evaluate(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y = select_features(frame)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    pipelines, diagnostics = fit_baselines(X_train, y_train)
    predictions = pd.DataFrame({"source_row": X_test.index, "target": y_test.to_numpy()})
    membership = pd.DataFrame({"source_row": frame.index, "target": y.to_numpy(), "split": "train"})
    membership.loc[membership.source_row.isin(X_test.index), "split"] = "test"
    for column in ("sample_id", "cell_id"):
        if column in frame:
            predictions[column] = frame.loc[X_test.index, column].to_numpy()
            membership[column] = frame[column].to_numpy()
    report = {
        "evaluation_notice": EVALUATION_NOTICE,
        "features": FEATURES, "continuous_features": CONTINUOUS, "categorical_features": CATEGORICAL,
        "excluded_predictors": FORBIDDEN, "target": "target", "positive_class": 1,
        "threshold": THRESHOLD,
        "split": {"method": "train_test_split", "test_size": TEST_SIZE, "stratify": "target",
                  "random_state": SEED, "n_train": len(y_train), "n_test": len(y_test),
                  "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
                  "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()}},
        "metric_definitions": {"pr_auc": "Trapezoidal auc(recall, precision) from precision_recall_curve",
                               "average_precision": "Non-interpolated average_precision_score; distinct from PR-AUC",
                               "precision_recall_f1": "Binary, positive class=1, zero_division=0",
                               "confusion_matrix": "Rows=true labels [0,1]; columns=predicted labels [0,1]"},
        "parameters": {"logistic": LOGISTIC_PARAMS, "random_forest": FOREST_PARAMS},
        "preprocessing": {"fit_scope": "training only", "logistic_continuous": "StandardScaler",
                          "random_forest_continuous": "passthrough",
                          "land_cover": "OneHotEncoder(handle_unknown='ignore'); no category grouping"},
        "models": {}, "diagnostics": diagnostics,
        "interpretation_notice": "Coefficients and impurity importances describe fitted models, not causality.",
    }
    for name, pipeline in pipelines.items():
        p_train, p_test = positive_probabilities(pipeline, X_train), positive_probabilities(pipeline, X_test)
        train_metrics, test_metrics = evaluate_predictions(y_train, p_train), evaluate_predictions(y_test, p_test)
        report["models"][name] = {
            "train": train_metrics, "test": test_metrics,
            "train_minus_test": {metric: train_metrics[metric] - test_metrics[metric] for metric in METRICS},
        }
        predictions[f"{name}_probability"] = p_test
        predictions[f"{name}_prediction"] = (p_test >= THRESHOLD).astype(int)
        encoder = pipeline.named_steps["preprocessor"].named_transformers_["land_cover"]
        known = [int(code) for code in encoder.categories_[0]]
        report["diagnostics"][name]["training_land_cover_categories"] = known
        report["diagnostics"][name]["test_unknown_land_cover_categories"] = sorted(set(X_test.land_cover) - set(known))
        report["diagnostics"][name]["test_unknown_land_cover_count"] = int((~X_test.land_cover.isin(known)).sum())
    report["diagnostics"]["logistic"]["intercept"] = float(pipelines["logistic"].named_steps["classifier"].intercept_[0])
    coefficients, importances = interpretation_tables(pipelines)
    return report, predictions, coefficients, importances, membership


def run(input_path: Path = INPUT, output_dir: Path = OUTPUT_DIR) -> dict:
    original = input_path.read_bytes()
    frame = pd.read_csv(input_path, float_precision="round_trip")
    if len(frame) != 540 or frame.target.value_counts().to_dict() != {0: 270, 1: 270}:
        raise ValueError("Official baseline expects 540 samples, 270 per class.")
    if frame.cell_id.isna().any() or not frame.cell_id.is_unique:
        raise ValueError("Official dataset must have unique, non-null cell_id values.")
    report, predictions, coefficients, importances, membership = train_and_evaluate(frame)
    if input_path.read_bytes() != original:
        raise RuntimeError("Input changed during training; no results exported.")
    report.update({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "source": "data/processed/santa_tereza_ml_dataset.csv",
                   "source_sha256": hashlib.sha256(original).hexdigest(),
                   "runtime": {"python": platform.python_version(), **{
                       package: importlib.metadata.version(package)
                       for package in ("scikit-learn", "numpy", "pandas", "scipy", "matplotlib")}}})
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [("test_predictions", predictions), ("logistic_coefficients", coefficients),
                        ("random_forest_importance", importances), ("split_membership", membership)]:
        table.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.17g", encoding="utf-8")
    save_evaluation_figures(predictions, output_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(EVALUATION_NOTICE)
    print(json.dumps({"split": report["split"], "models": report["models"],
                      "diagnostics": report["diagnostics"]}, indent=2))
    return report


if __name__ == "__main__":
    run()
