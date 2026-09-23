"""Fixed-buffer sensitivity: exact validation preservation and center separation."""

import json

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate_spatial_buffer as sensitivity
from src.models import evaluate_spatial_validation as evaluation
from src.models.spatial_validation import SpatialConfig, buffer_training_rows, iter_spatial_splits
from src.models.train_baselines import INPUT, FEATURES, CONTINUOUS, METRICS


@pytest.fixture(scope="module")
def frozen():
    frame = pd.read_csv(INPUT, float_precision="round_trip")
    assignments = pd.read_csv(sensitivity.PRINCIPAL / "fold_assignments.csv", float_precision="round_trip")
    principal = json.loads((sensitivity.PRINCIPAL / "summary_metrics.json").read_text())
    return frame, assignments, principal


def test_every_remaining_training_center_at_least_300m_and_validation_unchanged(frozen):
    frame, assignments, _ = frozen
    original_assignments = assignments.copy(deep=True)
    validated = []
    for _, original, valid in iter_spatial_splits(assignments, 5):
        valid_before = valid.copy()
        train, removed, nearest = buffer_training_rows(frame, original, valid)
        np.testing.assert_array_equal(valid, valid_before)
        validated.extend(valid)
        # Independent distance computation, without the implementation's cdist.
        delta = frame.iloc[original][["x", "y"]].to_numpy()[:, None, :] - frame.iloc[valid][["x", "y"]].to_numpy()[None, :, :]
        distances = np.sqrt((delta ** 2).sum(axis=2)).min(axis=1)
        np.testing.assert_allclose(nearest, distances)
        np.testing.assert_array_equal(train, original[distances >= 300])
        np.testing.assert_array_equal(removed, original[distances < 300])
        assert set(train) | set(removed) == set(original)
        assert not set(train) & set(removed)
        assert not set(train) & set(valid)
        assert not set(assignments.iloc[train].block_id) & set(assignments.iloc[valid].block_id)
        assert set(frame.iloc[train].target) == set(frame.iloc[valid].target) == {0, 1}
    assert sorted(validated) == list(range(540))
    pd.testing.assert_frame_equal(assignments, original_assignments)


def test_strict_boundary_keeps_exactly_300m_and_excludes_any_validation_neighbor():
    frame = pd.DataFrame({"x": [0, 0, 299.999, 300, 300, 299.999, 301],
                          "y": [0, 1000, 0, 0, 1000, 1000, 1000],
                          "target": [0, 1, 0, 0, 1, 1, 1]})
    kept, removed, _ = buffer_training_rows(frame, np.arange(2, 7), np.array([0, 1]))
    assert kept.tolist() == [3, 4, 6]
    assert removed.tolist() == [2, 5]


@pytest.mark.parametrize("radius", [-1, np.nan, np.inf])
def test_invalid_buffer_rejected(frozen, radius):
    frame, assignments, _ = frozen
    _, train, valid = next(iter_spatial_splits(assignments, 5))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        buffer_training_rows(frame, train, valid, radius)


def test_missing_class_after_buffer_fails_before_model_fit(frozen, monkeypatch):
    frame, assignments, principal = frozen
    def forbidden_fit(*args):
        pytest.fail("Invalid folds must be rejected before model fitting.")
    monkeypatch.setattr(evaluation, "fit_baselines", forbidden_fit)
    with pytest.raises(ValueError, match="both classes"):
        evaluation.evaluate_spatial(frame, frozen_assignments=assignments,
                                    frozen_metadata=principal["spatial"], buffer_m=1e6)


def test_buffer_removing_only_one_class_is_rejected():
    frame = pd.DataFrame({"x": [0, 0, 100, 400], "y": [0, 1000, 0, 0], "target": [0, 1, 1, 0]})
    with pytest.raises(ValueError, match="both classes"):
        buffer_training_rows(frame, np.array([2, 3]), np.array([0, 1]))


def test_fit_receives_only_remaining_training_rows_and_preserved_contract(frozen, monkeypatch):
    frame, assignments, principal = frozen
    expected = list(iter_spatial_splits(assignments, 5))
    original_fit = evaluation.fit_baselines
    calls = []
    def checked_fit(X, y):
        _, original, valid = expected[len(calls)]
        kept, removed, _ = buffer_training_rows(frame, original, valid)
        assert X.index.tolist() == kept.tolist()
        assert not set(X.index) & (set(removed) | set(valid))
        assert list(X.columns) == FEATURES
        models, diagnostic = original_fit(X, y)
        for name, model in models.items():
            for key, value in principal["parameters"][name].items():
                assert model.named_steps["classifier"].get_params()[key] == value
            prep = model.named_steps["preprocessor"]
            assert list(prep.feature_names_in_) == FEATURES
            encoder = prep.named_transformers_["land_cover"]
            assert set(encoder.categories_[0]) == set(X.land_cover)
            if name == "logistic":
                scaler = prep.named_transformers_["continuous"]
                assert scaler.n_samples_seen_ == len(kept)
                np.testing.assert_allclose(scaler.mean_, X[CONTINUOUS].mean())
        calls.append(kept)
        return models, diagnostic
    monkeypatch.setattr(evaluation, "fit_baselines", checked_fit)
    result = evaluation.evaluate_spatial(frame, SpatialConfig(), frozen_assignments=assignments,
                                         frozen_metadata=principal["spatial"], buffer_m=300)
    assert len(calls) == 5
    pd.testing.assert_frame_equal(result[1], assignments)
    assert principal["spatial"]["buffer_m"] == 0
    assert result[0]["spatial"]["buffer_m"] == 300


@pytest.mark.parametrize("relative", [".", "nested", ".."])
def test_protected_output_directory_rejected(relative):
    with pytest.raises(ValueError, match="protected"):
        sensitivity.validate_output_path(sensitivity.PRINCIPAL / relative, sensitivity.PRINCIPAL)


def test_frozen_folds_cannot_be_reassigned(frozen):
    frame, assignments, principal = frozen
    altered = assignments.copy()
    altered["fold"] = altered.fold % 5 + 1
    with pytest.raises(AssertionError):
        evaluation.evaluate_spatial(frame, frozen_assignments=altered,
                                    frozen_metadata=principal["spatial"], buffer_m=300)


def test_export_preserves_all_official_artifacts_and_valid_metrics(tmp_path, frozen):
    frame, assignments, principal = frozen
    paths = [INPUT, *[p for p in sensitivity.PRINCIPAL.rglob("*") if p.is_file()]]
    before = sensitivity.snapshot(paths)
    report = sensitivity.run(output_dir=tmp_path)
    assert sensitivity.snapshot(paths) == before
    assert report["protected_sources_unchanged"]
    assert (tmp_path / "fold_assignments.csv").read_bytes() == (sensitivity.PRINCIPAL / "fold_assignments.csv").read_bytes()
    metrics = pd.read_csv(tmp_path / "fold_metrics.csv")
    predictions = pd.read_csv(tmp_path / "oof_predictions.csv")
    membership = pd.read_csv(tmp_path / "buffer_membership.csv")
    comparison = pd.read_csv(tmp_path / "principal_vs_buffer300.csv")
    paired = pd.read_csv(tmp_path / "fold_comparison.csv")
    assert len(predictions) == 540 and len(membership) == 2700 and len(paired) == 10
    assert len(comparison) == 16
    assert np.isfinite(metrics[METRICS]).all().all()
    assert metrics[METRICS].ge(0).all().all() and metrics[METRICS].le(1).all().all()
    for count in report["fold_counts"]:
        part = membership[membership.evaluation_fold.eq(count["fold"])]
        assert part.role.eq("train").sum() == count["n_train"]
        assert part.role.eq("removed_by_buffer").sum() == count["n_removed"]
        assert part.role.eq("validation").sum() == count["n_validation"]
        assert part[part.role.eq("train")].distance_to_validation_m.ge(300).all()
        assert part[part.role.eq("removed_by_buffer")].distance_to_validation_m.lt(300).all()
        assert count["min_train_validation_distance_m"] >= 300
    for name, group in metrics.groupby("model"):
        assert predictions[f"{name}_probability"].between(0, 1).all()
        for metric in METRICS:
            values = report["models"][name][metric]
            assert values["mean"] == pytest.approx(group[metric].mean())
            assert values["std"] == pytest.approx(group[metric].std(ddof=1))
        for row in group.itertuples():
            subset = predictions[predictions.fold.eq(row.fold)]
            predicted = subset[f"{name}_probability"].ge(.5)
            assert row.accuracy == pytest.approx(predicted.eq(subset.target).mean())
            assert row.recall == pytest.approx(predicted[subset.target.eq(1)].mean())
    for row in comparison.itertuples():
        assert row.principal_mean == pytest.approx(principal["models"][row.model][row.metric]["mean"])
        assert row.buffer300_minus_principal == pytest.approx(row.buffer300_mean - row.principal_mean)
    for metric in METRICS:
        np.testing.assert_allclose(paired[f"{metric}_delta"], paired[f"{metric}_buffer300"] - paired[f"{metric}_principal"], atol=1e-15)
