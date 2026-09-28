"""Reconstruct the exact environmental intersection used by Stage 3."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from shapely import contains_xy

from src.features.build_ml_dataset import (
    Grid, Inputs, FEATURE_PATHS, VECTOR_PATHS, assert_grid_matches,
    cell_centers, joint_valid_mask, extract_features,
)
from src.models.train_baselines import ROOT, FEATURES
from src.models.train_final_models import OUTPUT, sha256, write_json, load_training_frame

TRACE = ["row", "col", "x", "y"]


def load_environment(root: Path = ROOT):
    """Read only environmental sources; occurrence inventories do not define inference eligibility."""
    with rasterio.open(root / FEATURE_PATHS["elevation"]) as reference:
        grid = Grid(reference.crs, reference.transform, reference.width, reference.height)
    t = grid.transform
    if (grid.crs is None or grid.crs.to_epsg() != 31982 or (t.a, t.e, t.b, t.d) != (30, -30, 0, 0)):
        raise ValueError("Expected north-up EPSG:31982, 30 m reference grid.")
    features, validity, sources = {}, {}, {}
    for name, relative in FEATURE_PATHS.items():
        path = root / relative
        with rasterio.open(path) as src:
            assert_grid_matches(src, grid)
            band = src.read(1, masked=True)
            data = np.asarray(band.data)
            valid = ~np.ma.getmaskarray(band) & np.isfinite(data)
            if src.nodata is not None:
                valid &= data != src.nodata
            features[name], validity[name] = data, valid
            sources[name] = {"path": relative, "sha256": sha256(path),
                             "nodata": src.nodata, "aligned": True}
    area_path = root / VECTOR_PATHS["study_area"]
    area = gpd.read_file(area_path)
    if (len(area) != 1 or area.crs != grid.crs or area.geometry.isna().any()
            or area.geometry.is_empty.any() or not area.geometry.is_valid.all()
            or not area.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Expected one valid municipality polygon in the reference CRS.")
    xy = cell_centers(np.arange(grid.width * grid.height), grid)
    municipality = contains_xy(area.geometry.union_all(), xy[:, 0], xy[:, 1]).reshape(grid.shape)
    mask = joint_valid_mask(municipality, features, validity)
    if not mask.any():
        raise ValueError("Empty environmental intersection.")
    sources["study_area"] = {"path": VECTOR_PATHS["study_area"], "sha256": sha256(area_path)}
    return grid, features, mask, municipality, sources


def build_grid(grid, features, mask) -> pd.DataFrame:
    ids = np.flatnonzero(mask)
    # Reuse the original float64 circular transformation and original class codes.
    inputs = Inputs(grid, features, mask, None, np.empty((0, 2)), mask)
    values = extract_features(ids, inputs)[FEATURES]
    rows, cols = np.divmod(ids, grid.width)
    xy = cell_centers(ids, grid)
    frame = pd.DataFrame({"row": rows, "col": cols, "x": xy[:, 0], "y": xy[:, 1]})
    frame[FEATURES] = values
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Inference grid contains NaN/Inf.")
    if (frame.land_cover.le(0).any() or not frame.slope.between(0, 90).all()
            or frame[FEATURES].eq(-9999).any().any()):
        raise ValueError("Invalid environmental predictor values.")
    return frame


def verify_training_cells(frame: pd.DataFrame, supervised: pd.DataFrame) -> None:
    recovered = supervised[TRACE + FEATURES].merge(frame, on=["row", "col"], how="left",
        suffixes=("_train", "_grid"), validate="one_to_one")
    for name in ["x", "y"] + FEATURES:
        if not np.allclose(recovered[f"{name}_train"], recovered[f"{name}_grid"], rtol=0, atol=1e-10):
            raise ValueError(f"Supervised cell differs from environmental grid: {name}")


def grid_metadata(grid) -> dict:
    return {"crs": grid.crs.to_string(), "resolution": [grid.transform.a, -grid.transform.e],
            "width": grid.width, "height": grid.height, "transform": list(grid.transform)[:6],
            "bounds": list(array_bounds(grid.height, grid.width, grid.transform))}


def run(output_dir: Path = OUTPUT):
    grid, features, mask, municipality, sources = load_environment()
    frame = build_grid(grid, features, mask)
    verify_training_cells(frame, load_training_frame())
    frozen = json.loads((ROOT / "data/processed/santa_tereza_ml_dataset_metadata.json").read_text())
    if int(mask.sum()) != frozen["counts"]["joint_valid_cells"]:
        raise ValueError("Joint mask count differs from Stage 3.")
    for name, item in sources.items():
        if name in frozen["sources"] and item["sha256"] != frozen["sources"][name]["sha256"]:
            raise ValueError(f"Environmental source changed since Stage 3: {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "inference_grid.csv", index=False, float_format="%.17g")
    write_json(output_dir / "inference_grid_metadata.json", {
        "grid": grid_metadata(grid), "valid_cells": len(frame),
        "municipality_cells": int(municipality.sum()), "features": FEATURES,
        "trace_only": TRACE, "sources": sources, "finite": True,
        "supervised_features_match": True,
        "mask_rule": "Municipal cell center strictly inside polygon AND all seven environmental rasters valid",
    })
    return frame, grid, mask


if __name__ == "__main__":
    frame, _, _ = run()
    print(f"Inference cells: {len(frame)}")
