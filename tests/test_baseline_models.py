"""Feature contract, leakage prevention and deterministic baseline evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from src.models import train_baselines as training
from src.models.evaluate_baselines import evaluate_predictions, save_evaluation_figures


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(17)
    n = 160
    y = np.tile([0, 1], n // 2)
    angle = rng.uniform(0, 2 * np.pi, n)
    return pd.DataFrame({
        "elevation": rng.uniform(80, 550, n), "slope": np.clip(15 + 8 * y + rng.normal(0, 7, n), 0, 60),
        "aspect_sin": np.sin(angle), "aspect_cos": np.cos(angle),
        "plan_curvature": rng.normal(0, 0.01, n), "profile_curvature": rng.normal(0, 0.005, n),
        "ndvi_pre_event": rng.uniform(0.4, 0.95, n), "land_cover": rng.choice([3, 15, 21], n),
        "target": y, "sample_id": [f"sample_{i}" for i in range(n)], "cell_id": np.arange(n),
        "aspect": np.degrees(angle), "x": np.arange(n) * 30, "y": np.arange(n) * 30,
        "row": np.arange(n), "col": 0, "landslide_count": y,
        "landslide_ids": "[]", "sample_type": np.where(y == 1, "landslide", "background"),
    })


@pytest.fixture(scope="module")
def fitted(dataset):
    X, y = training.select_features(dataset)
    X_train, X_test, y_train, y_test = training.split_dataset(X, y)
    models, diagnostics = training.fit_baselines(X_train, y_train)
    return models, diagnostics, X_train, X_test, y_train, y_test


def test_feature_contract_excludes_every_forbidden_column(dataset):
    X, y = training.select_features(dataset)
    assert list(X.columns) == training.FEATURES
    assert len(X.columns) == 8
    assert not set(X.columns).intersection(training.FORBIDDEN)
    assert y.name == "target"
    pd.testing.assert_frame_equal(dataset[training.FEATURES], X)


def test_missing_feature_rejected(dataset):
    with pytest.raises(ValueError, match="Missing required"):
        training.select_features(dataset.drop(columns="land_cover"))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -9999])
def test_invalid_feature_values_rejected(dataset, value):
    altered = dataset.copy()
    altered.loc[0, "elevation"] = value
    with pytest.raises(ValueError):
        training.select_features(altered)


def test_split_is_reproducible_stratified_and_disjoint(dataset):
    X, y = training.select_features(dataset)
    first = training.split_dataset(X, y)
    second = training.split_dataset(X, y)
    for left, right in zip(first, second):
        if isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right)
        else:
            pd.testing.assert_series_equal(left, right)
    X_train, X_test, y_train, y_test = first
    assert len(X_train) == 120 and len(X_test) == 40
    assert y_train.value_counts().to_dict() == {0: 60, 1: 60}
    assert y_test.value_counts().to_dict() == {0: 20, 1: 20}
    assert not set(X_train.index).intersection(X_test.index)
    assert set(X_train.index).union(X_test.index) == set(X.index)


def test_split_and_fit_refuse_unapproved_predictors(dataset):
    X, y = training.select_features(dataset)
    X["cell_id"] = dataset.cell_id
    with pytest.raises(ValueError, match="feature contract"):
        training.split_dataset(X, y)
    with pytest.raises(ValueError, match="approved predictors"):
        training.fit_baselines(X, y)


def test_transformers_start_unfitted_and_are_independent():
    pipelines = training.make_pipelines()
    first = pipelines["logistic"].named_steps["preprocessor"]
    second = pipelines["random_forest"].named_steps["preprocessor"]
    assert first is not second
    assert first.transformers[1][1] is not second.transformers[1][1]
    with pytest.raises(NotFittedError):
        first.get_feature_names_out()


def test_standard_scaler_is_fitted_only_on_training_rows(fitted):
    models, _, X_train, X_test, _, _ = fitted
    scaler = models["logistic"].named_steps["preprocessor"].named_transformers_["continuous"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.n_samples_seen_ == len(X_train)
    np.testing.assert_allclose(scaler.mean_, X_train[training.CONTINUOUS].mean(), rtol=1e-14)
    np.testing.assert_allclose(scaler.var_, X_train[training.CONTINUOUS].var(ddof=0), rtol=1e-14)
    assert not np.allclose(scaler.mean_, pd.concat([X_train, X_test])[training.CONTINUOUS].mean())


def test_test_only_category_and_extremes_do_not_leak_into_preprocessing(dataset):
    X, y = training.select_features(dataset)
    X_train, X_test, y_train, _ = training.split_dataset(X, y)
    X_train = X_train.copy(); X_train["land_cover"] = 3
    X_test = X_test.copy(); X_test["land_cover"] = 999; X_test["elevation"] = 1e9
    for name, pipeline in training.make_pipelines().items():
        preprocessor = pipeline.named_steps["preprocessor"]
        preprocessor.fit(X_train, y_train)
        encoder = preprocessor.named_transformers_["land_cover"]
        np.testing.assert_array_equal(encoder.categories_[0], [3])
        transformed = preprocessor.transform(X_test)
        assert transformed.shape[1] == 8
        np.testing.assert_array_equal(transformed[:, -1], 0)
        np.testing.assert_array_equal(encoder.categories_[0], [3])
        if name == "logistic":
            scaler = preprocessor.named_transformers_["continuous"]
            np.testing.assert_allclose(scaler.mean_, X_train[training.CONTINUOUS].mean())
            assert scaler.n_samples_seen_ == len(X_train)
        else:
            np.testing.assert_array_equal(transformed[:, :7], X_test[training.CONTINUOUS].to_numpy())


def test_random_forest_continuous_values_are_not_scaled(fitted):
    models, _, X_train, _, _, _ = fitted
    transformed = models["random_forest"].named_steps["preprocessor"].transform(X_train)
    np.testing.assert_array_equal(transformed[:, :7], X_train[training.CONTINUOUS].to_numpy())


def test_encoder_categories_come_only_from_training(fitted):
    models, _, X_train, _, _, _ = fitted
    for pipeline in models.values():
        encoder = pipeline.named_steps["preprocessor"].named_transformers_["land_cover"]
        assert encoder.handle_unknown == "ignore"
        np.testing.assert_array_equal(encoder.categories_[0], np.sort(X_train.land_cover.unique()))


def test_probabilities_and_metrics_have_expected_ranges(fitted):
    models, _, _, X_test, _, y_test = fitted
    for pipeline in models.values():
        scores = training.positive_probabilities(pipeline, X_test)
        assert np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()
        metrics = evaluate_predictions(y_test, scores)
        assert all(0 <= metrics[name] <= 1 for name in training.METRICS)
        assert np.asarray(metrics["confusion_matrix"]).sum() == len(y_test)


def test_metrics_match_hand_calculated_example():
    result = evaluate_predictions([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        assert result[name] == 0.5
    assert result["roc_auc"] == 0.75
    assert result["pr_auc"] == pytest.approx(19 / 24)
    assert result["average_precision"] == pytest.approx(5 / 6)


@pytest.mark.parametrize("scores", [[-0.1, 0.8], [0.1, 1.1], [np.nan, 0.8], [0.1, np.inf]])
def test_invalid_probabilities_are_rejected(scores):
    with pytest.raises(ValueError, match="finite and in"):
        evaluate_predictions([0, 1], scores)


def test_zero_predicted_positives_has_defined_metrics():
    result = evaluate_predictions([0, 1], [0.1, 0.2])
    assert result["precision"] == result["recall"] == result["f1"] == 0


def test_training_reproduces_with_same_seed(fitted):
    first, _, X_train, X_test, y_train, _ = fitted
    second, diagnostics = training.fit_baselines(X_train, y_train)
    for name in first:
        np.testing.assert_array_equal(training.positive_probabilities(first[name], X_test),
                                      training.positive_probabilities(second[name], X_test))
    for left, right in zip(training.interpretation_tables(first), training.interpretation_tables(second)):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert not diagnostics["logistic"]["convergence_warning"]


def test_interpretation_names_preserve_categories_and_are_sorted(fitted):
    models, _, X_train, _, _, _ = fitted
    coefficients, importances = training.interpretation_tables(models)
    expected_names = {f"land_cover__land_cover_{code}" for code in X_train.land_cover.unique()}
    for table in (coefficients, importances):
        assert expected_names.issubset(set(table.feature))
        assert table.feature.is_unique
        assert len(table) == 7 + X_train.land_cover.nunique()
    assert coefficients.absolute_coefficient.is_monotonic_decreasing
    assert importances.importance.is_monotonic_decreasing
    assert importances.importance.sum() == pytest.approx(1)


def test_report_and_test_artifacts(tmp_path, dataset):
    report, predictions, coefficients, importances, membership = training.train_and_evaluate(dataset)
    assert report["split"]["n_train"] == 120
    assert report["split"]["n_test"] == 40
    assert predictions.source_row.is_unique
    assert set(predictions.source_row) == set(membership.loc[membership.split.eq("test"), "source_row"])
    for name in ("logistic", "random_forest"):
        assert report["models"][name]["test"] == evaluate_predictions(predictions.target, predictions[f"{name}_probability"])
        np.testing.assert_array_equal(predictions[f"{name}_prediction"], predictions[f"{name}_probability"].ge(0.5).astype(int))
    save_evaluation_figures(predictions, tmp_path)
    expected = {"confusion_matrix_logistic.png", "confusion_matrix_random_forest.png",
                "roc_curves.png", "precision_recall_curves.png"}
    assert {file.name for file in tmp_path.iterdir()} == expected
    assert all((tmp_path / name).stat().st_size > 1000 for name in expected)


@pytest.mark.skipif(not training.INPUT.exists(), reason="Local dataset not available")
def test_official_split_sizes_and_class_stratification():
    frame = pd.read_csv(training.INPUT, float_precision="round_trip")
    X, y = training.select_features(frame)
    X_train, X_test, y_train, y_test = training.split_dataset(X, y)
    assert len(X_train) == 405 and len(X_test) == 135
    assert sorted(y_train.value_counts().tolist()) == [202, 203]
    assert sorted(y_test.value_counts().tolist()) == [67, 68]
