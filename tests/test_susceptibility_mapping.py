"""Stage 5 contracts, adversarial spatial QA and real-product reproducibility."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.features.build_ml_dataset import Grid, assert_grid_matches
from src.models.train_baselines import (
    FEATURES, FORBIDDEN, CONTINUOUS, fit_baselines, select_features,
    LOGISTIC_PARAMS, FOREST_PARAMS,
)
from src.models.train_final_models import OUTPUT, load_training_frame, verify_preservation, sha256, INPUT
from src.models.build_inference_grid import (
    load_environment, build_grid, verify_training_cells, TRACE,
)
from src.models.generate_susceptibility import (
    predict_scores, score_array, write_rasters, SCORES, NODATA, raster_path,
)
from src.models.evaluate_susceptibility import validate_rasters, compare_surfaces, distribution


@pytest.fixture(scope="module")
def real_products():
    if not (OUTPUT / "susceptibility_summary.json").exists():
        pytest.skip("Run Stage 5 to validate the real spatial artifacts.")
    grid, features, mask, _, _ = load_environment()
    frame = build_grid(grid, features, mask)
    scores = pd.read_csv(OUTPUT / "cell_scores.csv", float_precision="round_trip")
    models = {name: joblib.load(OUTPUT / f"{name}.joblib") for name in ("logistic", "random_forest")}
    return grid, mask, frame, scores, models


def test_feature_contract_and_training_preprocessing(real_products):
    _, _, frame, _, models = real_products
    training = load_training_frame()
    X, y = select_features(training)
    assert len(X) == 540 and y.value_counts().to_dict() == {0: 270, 1: 270}
    assert list(frame.columns) == TRACE + FEATURES
    assert not set(FORBIDDEN) & set(frame[FEATURES].columns)
    verify_training_cells(frame, training)
    for name, model in models.items():
        assert list(model.feature_names_in_) == FEATURES
        params = LOGISTIC_PARAMS if name == "logistic" else FOREST_PARAMS
        assert all(model.named_steps["classifier"].get_params()[k] == v for k, v in params.items())
        prep = model.named_steps["preprocessor"]
        transformed = prep.transform(X)
        continuous = X[CONTINUOUS].to_numpy()
        if name == "logistic":
            scaler = prep.named_transformers_["continuous"]
            assert scaler.n_samples_seen_ == 540
            np.testing.assert_allclose(scaler.mean_, continuous.mean(axis=0))
            np.testing.assert_allclose(scaler.var_, continuous.var(axis=0))
            np.testing.assert_allclose(transformed[:, :7], (continuous-scaler.mean_)/scaler.scale_)
        else:
            np.testing.assert_array_equal(transformed[:, :7], continuous)
        encoder = prep.named_transformers_["land_cover"]
        np.testing.assert_array_equal(encoder.categories_[0], np.sort(X.land_cover.unique()))
        unknown = X.iloc[:1].copy()
        unknown["land_cover"] = 999999
        assert np.all(prep.transform(unknown)[:, 7:] == 0)


def test_real_rasters_and_coverage(real_products):
    grid, mask, frame, scores, _ = real_products
    assert len(scores) == len(frame) == mask.sum() == 80931
    assert grid.shape == (570, 408)
    assert np.isfinite(frame.to_numpy()).all()
    pd.testing.assert_frame_equal(frame[TRACE], scores[TRACE], check_exact=True)
    assert (scores[list(SCORES.values())].ge(0) & scores[list(SCORES.values())].le(1)).all().all()
    assert np.array_equal(scores.ensemble_mean_score, (scores.logistic_score+scores.random_forest_score)/2)
    qa = validate_rasters(scores, grid, mask, OUTPUT)
    assert all(all(item["checks"].values()) for item in qa.values())


def test_reproducible_full_fit_inference_and_rasters(real_products, tmp_path):
    grid, mask, frame, scores, saved = real_products
    X, y = select_features(load_training_frame())
    fresh, diagnostics = fit_baselines(X, y)
    assert not any(item["warnings"] for item in diagnostics.values())
    repeated = predict_scores(fresh, frame)
    pd.testing.assert_frame_equal(repeated, scores, check_exact=True)
    pd.testing.assert_frame_equal(predict_scores(saved, frame), scores, check_exact=True)
    write_rasters(repeated, grid, mask, tmp_path)
    validate_rasters(repeated, grid, mask, tmp_path)
    for name in SCORES:
        assert sha256(raster_path(tmp_path, name)) == sha256(raster_path(OUTPUT, name))


def test_original_dataset_and_artifacts_preserved(real_products):
    result = verify_preservation()
    assert result["passed"] and result["checked_files"] >= 125
    assert sha256(INPUT) == "334fd5bcd190ffc4ea7ad4d34debecce8d1c4af6910e663411eab6dd5b379162"


def test_statistics_reconcile(real_products):
    _, _, _, scores, _ = real_products
    summary = json.loads((OUTPUT / "susceptibility_summary.json").read_text(encoding="utf-8"))
    for name, col in SCORES.items():
        assert distribution(scores[col]) == summary["scores"][name]
    assert compare_surfaces(scores.logistic_score, scores.random_forest_score) == summary["surface_comparison"]
    sampled = load_training_frame()[["row", "col", "target"]].merge(scores, on=["row", "col"], validate="one_to_one")
    for group in summary["positive_background"]:
        target = 1 if group["group"] == "positive" else 0
        expected = distribution(sampled.loc[sampled.target.eq(target), SCORES[group["model"]]])
        assert expected["count"] == 270
        assert all(group[key] == val for key, val in expected.items())


@pytest.fixture
def small_grid():
    grid = Grid(rasterio.crs.CRS.from_epsg(31982), from_origin(400000, 6800000, 30, 30), 3, 2)
    mask = np.array([[True, False, True], [False, True, False]])
    frame = pd.DataFrame({"row": [0, 0, 1], "col": [0, 2, 1]})
    return grid, mask, frame


@pytest.mark.parametrize("values", [[-0.1, .5, .8], [.2, 1.01, .8], [np.nan, .5, .8], [.2, np.inf, .8]])
def test_reject_invalid_scores(small_grid, values):
    grid, mask, frame = small_grid
    with pytest.raises(ValueError, match="scores"):
        score_array(values, frame, grid, mask)


@pytest.mark.parametrize("fault", ["missing", "outside", "duplicate", "negative", "fractional"])
def test_reject_invalid_mask_coverage(small_grid, fault):
    grid, mask, frame = small_grid
    if fault == "missing":
        frame = frame.iloc[:2]
    elif fault == "outside":
        frame.loc[0, "col"] = 1
    elif fault == "duplicate":
        frame.loc[1, ["row", "col"]] = [0, 0]
    elif fault == "negative":
        frame.loc[0, "row"] = -1
    else:
        frame = frame.astype(float)
        frame.loc[0, "row"] = .5
    with pytest.raises(ValueError):
        score_array(np.full(len(frame), .5), frame, grid, mask)


@pytest.mark.parametrize("fault", ["crs", "transform", "dimensions"])
def test_reject_misaligned_environment(small_grid, tmp_path, fault):
    grid, _, _ = small_grid
    path = tmp_path / "mismatch.tif"
    with rasterio.open(path, "w", driver="GTiff", width=4 if fault == "dimensions" else 3,
            height=2, count=1, dtype="float32", crs="EPSG:4326" if fault == "crs" else grid.crs,
            transform=from_origin(400015, 6800000, 30, 30) if fault == "transform" else grid.transform) as dst:
        with pytest.raises(ValueError, match="reference grid"):
            assert_grid_matches(dst, grid)


def test_nodata_and_float32_roundtrip(small_grid, tmp_path):
    grid, mask, frame = small_grid
    for column in SCORES.values():
        frame[column] = [0, .5, 1]
    write_rasters(frame, grid, mask, tmp_path)
    validate_rasters(frame, grid, mask, tmp_path)
    with rasterio.open(raster_path(tmp_path, "logistic")) as src:
        assert (src.read(1)[~mask] == NODATA).all()
    # A superficially aligned file with a score outside the mask must fail QA.
    with rasterio.open(raster_path(tmp_path, "logistic"), "r+") as dst:
        values = dst.read(1)
        values[0, 1] = .5
        dst.write(values, 1)
    with pytest.raises(ValueError, match="QA failed"):
        validate_rasters(frame, grid, mask, tmp_path)


def test_manual_surface_comparison():
    result = compare_surfaces([0, .5, 1], [0, .25, .75])
    assert result["mean_absolute_difference"] == pytest.approx(1/6)
    assert result["rmse"] == pytest.approx(np.sqrt(.125/3))
    assert result["mean_signed_difference_lr_minus_rf"] == pytest.approx(1/6)
