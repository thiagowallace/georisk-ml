"""Generate unclassified landslide susceptibility scores on the environmental grid."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import rasterio

from src.models.train_baselines import FEATURES, positive_probabilities
from src.models.train_final_models import (
    OUTPUT, NOTICE, run as train_final, verify_preservation, sha256, write_json,
)
from src.models.build_inference_grid import run as build_inference, TRACE

NODATA = -9999.0
SCORES = {"logistic": "logistic_score", "random_forest": "random_forest_score",
          "ensemble_mean": "ensemble_mean_score"}


def raster_path(output_dir, name):
    return output_dir / f"santa_tereza_susceptibility_{name}.tif"


def predict_scores(models, frame):
    X = frame[FEATURES]
    result = frame[TRACE].copy()
    for name in ("logistic", "random_forest"):
        if list(models[name].feature_names_in_) != FEATURES:
            raise ValueError("Serialized model feature contract differs.")
        result[SCORES[name]] = positive_probabilities(models[name], X)
    result["ensemble_mean_score"] = (result.logistic_score + result.random_forest_score) / 2
    return result


def score_array(values, frame, grid, mask):
    values = np.asarray(values)
    if len(values) != len(frame) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid susceptibility scores.")
    rows, cols = frame.row.to_numpy(), frame.col.to_numpy()
    if (not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(cols.dtype, np.integer)
            or ((rows < 0) | (rows >= grid.height) | (cols < 0) | (cols >= grid.width)).any()
            or frame.duplicated(["row", "col"]).any()):
        raise ValueError("Invalid or duplicate spatial cell indices.")
    covered = np.zeros(grid.shape, dtype=bool)
    covered[rows, cols] = True
    if not np.array_equal(covered, mask):
        raise ValueError("Scores must cover exactly the environmental mask.")
    result = np.full(grid.shape, NODATA, dtype=np.float32)
    result[rows, cols] = values.astype(np.float32)
    return result


def write_rasters(scores, grid, mask, output_dir):
    for name, column in SCORES.items():
        values = score_array(scores[column], scores, grid, mask)
        with rasterio.open(raster_path(output_dir, name), "w", driver="GTiff",
                width=grid.width, height=grid.height, count=1, dtype="float32",
                crs=grid.crs, transform=grid.transform, nodata=NODATA,
                compress="deflate", predictor=3) as dst:
            dst.write(values, 1)
            dst.set_band_description(1, column)
            dst.update_tags(score_semantics=NOTICE, product=column,
                            ensemble_status="exploratory; not separately validated" if name == "ensemble_mean" else "single model")


def run(output_dir: Path = OUTPUT, *, retrain: bool = True):
    if retrain:
        train_final(output_dir)
    metadata = json.loads((output_dir / "final_model_metadata.json").read_text(encoding="utf-8"))
    models = {}
    for name in ("logistic", "random_forest"):
        path = output_dir / f"{name}.joblib"
        if sha256(path) != metadata["model_sha256"][path.name]:
            raise ValueError("Serialized model hash differs from final training metadata.")
        models[name] = joblib.load(path)
    frame, grid, mask = build_inference(output_dir)
    scores = predict_scores(models, frame)
    scores.to_csv(output_dir / "cell_scores.csv", index=False, float_format="%.17g")
    write_rasters(scores, grid, mask, output_dir)
    unknown = {}
    for name, model in models.items():
        known = model.named_steps["preprocessor"].named_transformers_["land_cover"].categories_[0]
        unknown[name] = {"codes": sorted(int(v) for v in set(frame.land_cover) - set(known)),
                         "cells": int((~frame.land_cover.isin(known)).sum())}
    write_json(output_dir / "prediction_metadata.json", {
        "notice": NOTICE, "unknown_land_cover": unknown,
        "score_precision": "CSV float64; GeoTIFF float32; ensemble averaged before float32 conversion",
        "preservation": verify_preservation(output_dir),
    })
    from src.models.evaluate_susceptibility import run as evaluate
    return evaluate(output_dir)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
