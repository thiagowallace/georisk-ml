"""Deterministic statistical checks independent of fitting any model."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.features import explore_ml_dataset as eda


@pytest.fixture
def dataset():
    count = 540
    index = np.arange(count)
    angle = index % 360
    return pd.DataFrame({
        "sample_id": [f"sample_{i:06d}" for i in index], "cell_id": index,
        "row": index // 30, "col": index % 30, "x": index * 30 + 15,
        "y": 6000 - index * 30, "landslide_ids": [f"[{i}]" if i < 270 else "[]" for i in index],
        "landslide_count": (index < 270).astype(int), "target": (index < 270).astype(int),
        "sample_type": np.where(index < 270, "landslide", "background"),
        "elevation": 100.0 + index, "slope": 10.0 + index % 40,
        "aspect": angle.astype(float), "aspect_sin": np.sin(np.deg2rad(angle)),
        "aspect_cos": np.cos(np.deg2rad(angle)), "plan_curvature": (index % 13 - 6) / 1000,
        "profile_curvature": (index % 17 - 8) / 2000,
        "ndvi_pre_event": 0.4 + (index % 10) / 20, "land_cover": np.where(index % 3, 3, 21),
    })


def test_expected_schema_count_balance_and_finite_features(dataset):
    qa = eda.validate_dataset(dataset)
    assert qa["rows"] == 540
    assert sum(qa["missing_by_column"].values()) == 0
    assert qa["infinite_values"] == 0
    assert qa["duplicates"] == {"rows": 0, "cell_id": 0, "sample_id": 0}
    assert set(eda.EXPECTED).issubset(dataset.columns)
    assert dataset.target.value_counts().to_dict() == {0: 270, 1: 270}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -9999])
def test_nonfinite_and_nodata_rejected(dataset, value):
    dataset.loc[0, "elevation"] = value
    with pytest.raises(ValueError):
        eda.validate_dataset(dataset)


def test_wrong_row_count_rejected(dataset):
    with pytest.raises(ValueError, match="540 rows"):
        eda.validate_dataset(dataset.iloc[:-1])


def test_missing_column_rejected(dataset):
    with pytest.raises(ValueError, match="Missing columns"):
        eda.validate_dataset(dataset.drop(columns="slope"))


def test_imbalanced_classes_rejected(dataset):
    dataset.loc[0, ["sample_type", "target"]] = ["background", 0]
    with pytest.raises(ValueError, match="balanced"):
        eda.validate_dataset(dataset)


def test_target_mismatch_rejected(dataset):
    dataset.loc[0, "target"] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        eda.validate_dataset(dataset)


@pytest.mark.parametrize("key", ["sample_id", "cell_id"])
def test_duplicate_identifiers_rejected(dataset, key):
    dataset.loc[1, key] = dataset.loc[0, key]
    with pytest.raises(ValueError, match="duplicates"):
        eda.validate_dataset(dataset)


def test_descriptive_statistics_match_known_sequence(dataset):
    tables, _ = eda.analyze(dataset)
    summary = tables["descriptive_by_class"].set_index(["sample_type", "variable"])
    values = summary.loc[("landslide", "elevation")]
    assert values["count"] == 270
    assert values["min"] == 100
    assert values["max"] == 369
    assert values["mean"] == values["median"] == 234.5
    assert values["std"] == pytest.approx(np.std(np.arange(100, 370), ddof=1))
    comparison = tables["continuous_comparison"].set_index("variable").loc["elevation"]
    assert comparison["mean_difference"] == -270
    assert comparison["rank_biserial"] == -1


def test_circular_mean_handles_north_seam_and_undefined_direction():
    result = eda.circular_statistics(np.array([359, 1]))
    assert result["mean_direction_deg"] == pytest.approx(0, abs=1e-12)
    assert result["resultant_length"] > 0.99
    assert eda.circular_statistics(np.array([0, 90, 180, 270]))["mean_direction_deg"] is None


def test_iqr_flags_are_not_automatic_removals():
    values = pd.Series([0.0, 0.0, 0.0, 0.0, 5.0])
    flags, low, high = eda.iqr_flags(values)
    assert flags.tolist() == [False, False, False, False, True]
    assert low == high == 0
    assert len(values) == 5


def test_correlations_use_only_appropriate_variables(dataset):
    tables, _ = eda.analyze(dataset)
    matrix = tables["correlation_pearson_pooled"].set_index("variable")
    assert list(matrix.columns) == list(eda.CONTINUOUS)
    assert not set(eda.TRACE + ("target", "land_cover", "aspect")).intersection(matrix.columns)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 1)


def test_constant_features_produce_undefined_not_fabricated_correlation(dataset):
    dataset["elevation"] = 100.0
    tables, _ = eda.analyze(dataset)
    matrix = tables["correlation_pearson_pooled"].set_index("variable")
    assert matrix.loc["elevation"].isna().all()
    comparison = tables["continuous_comparison"].set_index("variable")
    assert pd.isna(comparison.loc["elevation", "standardized_mean_difference"])


def test_land_cover_class_counts_and_denominators(dataset):
    tables, _ = eda.analyze(dataset)
    cover = tables["land_cover_by_class"]
    assert set(cover.land_cover) == {3, 21}
    assert cover.groupby("sample_type")["count"].sum().eq(270).all()
    np.testing.assert_allclose(cover.groupby("sample_type").percent_within_class.sum(), [100, 100])


def test_analysis_is_reproducible_under_input_reordering(dataset):
    first, qa1 = eda.analyze(dataset)
    second, qa2 = eda.analyze(dataset.sample(frac=1, random_state=13))
    assert qa1 == qa2
    assert first.keys() == second.keys()
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key], check_exact=True)


def test_saved_analytical_outputs_are_reproducible(tmp_path, dataset, monkeypatch):
    source = tmp_path / "dataset.csv"
    dataset.to_csv(source, index=False, float_format="%.17g")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(eda, "render_figures", lambda *args: None)
    eda.run(source, tmp_path / "first")
    eda.run(source, tmp_path / "second")
    for file in (tmp_path / "first").iterdir():
        assert file.read_bytes() == (tmp_path / "second" / file.name).read_bytes()
    summary = json.loads((tmp_path / "first/statistical_summary.json").read_text(encoding="utf-8"))
    assert summary["qa"]["rows"] == 540
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_figures_are_created(tmp_path, dataset):
    tables, _ = eda.analyze(dataset)
    eda.render_figures(dataset, tables, tmp_path)
    expected = {"histograms.png", "boxplots.png", "correlations.png", "land_cover.png",
                "aspect_circular.png", "class_distribution.png"}
    assert {file.name for file in tmp_path.iterdir()} == expected
    assert all((tmp_path / name).stat().st_size > 1000 for name in expected)


@pytest.mark.skipif(not eda.INPUT.exists(), reason="Local processed dataset not available")
def test_local_dataset_contract():
    frame = pd.read_csv(eda.INPUT, float_precision="round_trip")
    qa = eda.validate_dataset(frame)
    assert qa["rows"] == 540
    assert frame.sample_type.value_counts().to_dict() == {"landslide": 270, "background": 270}
