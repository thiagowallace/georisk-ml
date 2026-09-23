"""Spatial isolation, fold-local learning and auditable Stage 4 evaluation."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models import evaluate_spatial_validation as evaluation
from src.models.spatial_validation import SpatialConfig, build_spatial_folds, iter_spatial_splits
from src.models.train_baselines import (
    INPUT, FEATURES, FORBIDDEN, CONTINUOUS, METRICS, make_pipelines, select_features,
)


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(INPUT, float_precision="round_trip")


@pytest.fixture(scope="module")
def evaluated(dataset):
    return evaluation.evaluate_spatial(dataset)


def test_all_540_cells_assigned_once_and_spatially_disjoint(dataset):
    assignments, metadata = build_spatial_folds(dataset)
    assert len(assignments) == 540
    assert assignments.sample_id.is_unique and assignments.cell_id.is_unique
    assert assignments.groupby("block_id").fold.nunique().eq(1).all()
    seen = []
    for _, train, valid in iter_spatial_splits(assignments, 5):
        seen.extend(valid)
        assert not set(train) & set(valid)
        assert len(train) + len(valid) == 540
        assert not set(assignments.iloc[train].block_id) & set(assignments.iloc[valid].block_id)
        for index in (train, valid):
            assert set(dataset.iloc[index].target) == {0, 1}
    assert sorted(seen) == list(range(540))
    assert metadata["cells_per_block_side"] == 50
    # Entire 30 m cell footprint lies inside its assigned half-open block.
    for axis in ("x", "y"):
        lower = metadata[f"origin_{axis}"] + assignments[f"block_{axis}"] * 1500
        assert (assignments[axis] - 15 >= lower - 1e-8).all()
        assert (assignments[axis] + 15 <= lower + 1500 + 1e-8).all()


@pytest.mark.parametrize("n_splits", [3, 5])
def test_reproducible_and_configurable(dataset, n_splits):
    a, m = build_spatial_folds(dataset, SpatialConfig(n_splits=n_splits))
    b, n = build_spatial_folds(dataset, SpatialConfig(n_splits=n_splits))
    pd.testing.assert_frame_equal(a, b)
    assert m == n and a.fold.nunique() == n_splits


@pytest.mark.parametrize("config", [SpatialConfig(n_splits=1), SpatialConfig(n_splits=2.5),
                                       SpatialConfig(block_size_m=0), SpatialConfig(block_size_m=np.nan),
                                       SpatialConfig(block_size_m=31), SpatialConfig(block_size_m=30000)])
def test_invalid_config_rejected(dataset, config):
    with pytest.raises(ValueError):
        build_spatial_folds(dataset, config)


def test_duplicate_cells_and_invalid_coordinates_rejected(dataset):
    for column, value in [("x", np.nan), ("cell_id", dataset.cell_id.iloc[1]),
                          ("sample_id", dataset.sample_id.iloc[1]), ("x", dataset.x.iloc[0] + 1)]:
        altered = dataset.copy()
        altered.loc[0, column] = value
        with pytest.raises(ValueError):
            build_spatial_folds(altered)


def test_corrupted_assignments_and_single_class_fold_rejected(dataset):
    assignments, _ = build_spatial_folds(dataset)
    block = assignments.block_id.value_counts().idxmax()
    altered = assignments.copy()
    index = altered.index[altered.block_id.eq(block)][0]
    altered.loc[index, "fold"] = altered.loc[index, "fold"] % 5 + 1
    with pytest.raises(ValueError, match="multiple folds"):
        list(iter_spatial_splits(altered, 5))
    altered = assignments.copy()
    altered.loc[altered.fold.eq(1), "target"] = 0
    with pytest.raises(ValueError, match="both classes"):
        list(iter_spatial_splits(altered, 5))


def test_pipeline_fit_receives_only_training_rows(dataset, monkeypatch):
    assignments, _ = build_spatial_folds(dataset)
    expected = list(iter_spatial_splits(assignments, 5))
    original = evaluation.fit_baselines
    calls = []
    def checked_fit(X, y):
        _, train, valid = expected[len(calls)]
        assert X.index.tolist() == train.tolist()
        assert not set(X.index) & set(valid)
        assert X.columns.tolist() == FEATURES
        models, diagnostics = original(X, y)
        for name, pipeline in models.items():
            prep = pipeline.named_steps["preprocessor"]
            assert list(prep.feature_names_in_) == FEATURES
            assert not set(prep.feature_names_in_) & set(FORBIDDEN)
            encoder = prep.named_transformers_["land_cover"]
            assert set(encoder.categories_[0]) == set(X.land_cover)
            if name == "logistic":
                scaler = prep.named_transformers_["continuous"]
                assert scaler.n_samples_seen_ == len(train)
                np.testing.assert_allclose(scaler.mean_, X[CONTINUOUS].mean())
        calls.append(train)
        return models, diagnostics
    monkeypatch.setattr(evaluation, "fit_baselines", checked_fit)
    evaluation.evaluate_spatial(dataset)
    assert len(calls) == 5


def test_validation_only_category_does_not_enter_encoder(dataset):
    X, y = select_features(dataset)
    X = X.copy()
    X.loc[X.index[:10], "land_cover"] = 999
    for pipeline in make_pipelines().values():
        prep = pipeline.named_steps["preprocessor"]
        prep.fit(X.iloc[10:], y.iloc[10:])
        assert 999 not in prep.named_transformers_["land_cover"].categories_[0]
        transformed = prep.transform(X.iloc[:10])
        assert np.isfinite(transformed).all()
        assert (transformed[:, len(CONTINUOUS):] == 0).all()


def test_probabilities_metrics_summaries_and_confusion(evaluated):
    report, assignments, metrics, matrices, predictions = evaluated
    assert len(metrics) == 10 and len(matrices) == 10
    assert np.isfinite(metrics[METRICS]).all().all()
    assert metrics[METRICS].ge(0).all().all() and metrics[METRICS].le(1).all().all()
    for name, group in metrics.groupby("model"):
        probabilities = predictions[f"{name}_probability"]
        assert probabilities.notna().all() and probabilities.between(0, 1).all()
        for metric in METRICS:
            values = report["models"][name][metric]
            assert values["mean"] == pytest.approx(np.mean(group[metric]))
            assert values["std"] == pytest.approx(np.std(group[metric], ddof=1))
            assert values["min"] == group[metric].min()
            assert values["max"] == group[metric].max()
        for row in group.itertuples():
            subset = predictions[predictions.fold.eq(row.fold)]
            p = subset[f"{name}_probability"].to_numpy()
            y = subset.target.to_numpy()
            predicted = p >= .5
            assert row.accuracy == pytest.approx(np.mean(predicted == y))
            assert row.recall == pytest.approx(predicted[y == 1].mean())
            assert row.roc_auc == pytest.approx(roc_auc_score(y, p))
            assert row.average_precision == pytest.approx(average_precision_score(y, p))
            cm = matrices[matrices.model.eq(name) & matrices.fold.eq(row.fold)].iloc[0]
            assert cm.tn == np.sum((y == 0) & ~predicted)
            assert cm.fp == np.sum((y == 0) & predicted)
            assert cm.fn == np.sum((y == 1) & ~predicted)
            assert cm.tp == np.sum((y == 1) & predicted)
            assert cm.tn + cm.fp + cm.fn + cm.tp == len(subset)


def test_saved_baseline_comparison(evaluated):
    baseline = json.loads(evaluation.BASELINE.read_text(encoding="utf-8"))
    table = evaluation.compare_baseline(evaluated[0], baseline)
    assert len(table) == 16
    np.testing.assert_allclose(table.spatial_minus_baseline, table.spatial_mean - table.baseline_random_test)
    for row in table.itertuples():
        assert row.baseline_random_test == baseline["models"][row.model]["test"][row.metric]


def test_export_artifacts_and_source_immutability(tmp_path, evaluated, monkeypatch):
    before = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    monkeypatch.setattr(evaluation, "evaluate_spatial", lambda *args: evaluated)
    report = evaluation.run(output_dir=tmp_path)
    expected = ["fold_assignments.csv", "fold_metrics.csv", "summary_metrics.json", "confusion_matrices.csv",
                "baseline_vs_spatial.csv", "spatial_folds.png", "metric_comparison.png", "roc_auc_by_fold.png",
                "recall_by_fold.png", "oof_predictions.csv", "fold_class_counts.csv"]
    assert all((tmp_path / file).stat().st_size > 0 for file in expected)
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == before == report["source_sha256"]
    assert len(pd.read_csv(tmp_path / "fold_assignments.csv")) == 540
