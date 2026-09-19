"""Audit the saved cell-based ML dataset and its provenance, without writing files."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.build_ml_dataset import (
    COLUMNS, FEATURE_PATHS, LAYER_NAME, OUTPUT_NAMES, PREDICTOR_COLUMNS,
    PROJECT_ROOT, STRATEGY, TRACE_COLUMNS, Inputs, assemble_dataset,
    cell_centers, grid_metadata, load_inputs, positive_cells, source_manifest,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compare_tables(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(f"{label}: {exc}") from exc


def minimum_distances(centers: np.ndarray, occurrences: np.ndarray) -> np.ndarray:
    """Independent audit of the tree-based sampler using pairwise distances."""
    require(len(occurrences) > 0, "Exclusion inventory is empty.")
    result = np.empty(len(centers), dtype=np.float64)
    # Bound memory even if future inventories contain many more points.
    for start in range(0, len(centers), 64):
        batch = centers[start:start + 64]
        nearest = np.full(len(batch), np.inf)
        for offset in range(0, len(occurrences), 4096):
            distances = cdist(batch, occurrences[offset:offset + 4096], metric="euclidean")
            nearest = np.minimum(nearest, distances.min(axis=1))
        result[start:start + len(batch)] = nearest
    return result


def audit_table(table: pd.DataFrame, inputs: Inputs, radius: float) -> dict:
    """Check persisted values against source cells, labels and exclusion geometry."""
    require(list(table.columns) == COLUMNS, "Unexpected dataset columns or column ordering.")
    require(not table.empty and not table.isna().any().any(), "Dataset is empty or contains missing values.")
    for name in ("cell_id", "row", "col", "landslide_count", "target", "land_cover"):
        require(pd.api.types.is_integer_dtype(table[name]), f"{name} must be integer-valued.")
    require(table.cell_id.is_unique, "Duplicate cell_id.")
    require(table.sample_id.is_unique, "Duplicate sample_id.")
    require(set(table.target) == {0, 1}, "Both landslide and background classes are required.")
    require(table.sample_type.eq(table.target.map({1: "landslide", 0: "background"})).all(),
            "sample_type does not match target.")
    grid = inputs.grid
    require(table.row.between(0, grid.height - 1).all()
            and table.col.between(0, grid.width - 1).all(), "Row/column outside reference grid.")
    require(table.cell_id.eq(table.row * grid.width + table.col).all(), "Incorrect cell_id.")
    cell_ids = table.cell_id.to_numpy(dtype=np.int64)
    require(table.sample_id.tolist() == [f"sample_{cell_id:06d}" for cell_id in cell_ids],
            "Incorrect sample_id.")
    rows, cols = table.row.to_numpy(), table.col.to_numpy()
    require(inputs.valid_mask[rows, cols].all(), "Samples outside the joint valid municipal mask.")
    centers = cell_centers(cell_ids, grid)
    require(np.allclose(table[["x", "y"]].to_numpy(), centers, rtol=0, atol=1e-8),
            "Coordinates do not match cell centers.")

    for name in [*FEATURE_PATHS, "aspect_sin", "aspect_cos"]:
        values = table[name].to_numpy(dtype=np.float64)
        require(np.isfinite(values).all(), f"Nonfinite feature values: {name}.")
        if name in FEATURE_PATHS:
            require(np.allclose(values, inputs.features[name][rows, cols], rtol=1e-12, atol=1e-12),
                    f"Feature values differ from source raster: {name}.")
    require(np.array_equal(table.land_cover.to_numpy(), inputs.features["land_cover"][rows, cols]),
            "Original MapBiomas codes were not preserved.")
    require(table.aspect.between(0, 360, inclusive="left").all(), "Invalid aspect degrees.")
    require(table.ndvi_pre_event.between(-1, 1).all(), "NDVI outside [-1, 1].")
    radians = np.deg2rad(table.aspect.to_numpy(dtype=float))
    for name, expected in [("aspect_sin", np.sin(radians)), ("aspect_cos", np.cos(radians))]:
        require(table[name].between(-1, 1).all(), f"{name} outside [-1, 1].")
        require(np.allclose(table[name], expected, rtol=0, atol=1e-12), f"Incorrect {name}.")

    positive = table.loc[table.target.eq(1)]
    background = table.loc[table.target.eq(0)]
    require(not set(positive.cell_id).intersection(background.cell_id),
            "A background cell coincides with a positive cell.")
    require(background.landslide_count.eq(0).all() and background.landslide_ids.eq("[]").all(),
            "Background must have count 0 and an empty landslide_ids list.")
    expected_positive = positive_cells(inputs.occurrences, grid, inputs.valid_mask)
    compare_tables(
        positive[["cell_id", "landslide_count", "landslide_ids"]].sort_values("cell_id"),
        expected_positive, "Positive occurrence traceability differs from the source inventory",
    )
    # Independently parse the lists: the count and global occurrence IDs must agree.
    ids = []
    for record in positive.itertuples(index=False):
        parsed = json.loads(record.landslide_ids)
        require(isinstance(parsed, list) and len(parsed) == record.landslide_count,
                "landslide_count differs from landslide_ids.")
        require(all(type(value) is int for value in parsed), "landslide_ids must contain integers.")
        ids.extend(parsed)
    require(sorted(ids) == sorted(inputs.occurrences.landslide_id.tolist()),
            "Original occurrence IDs were lost or duplicated.")
    require(np.isfinite(radius) and radius >= 0, "Invalid exclusion radius.")
    distances = minimum_distances(
        cell_centers(background.cell_id.to_numpy(), grid), inputs.exclusion_xy,
    )
    require((distances >= radius).all(), "Background center lies inside the exclusion radius.")
    return {
        "total_samples": len(table),
        "class_distribution": {"landslide": len(positive), "background": len(background)},
        "original_occurrences": len(ids), "unique_cell_ids": int(table.cell_id.nunique()),
        "minimum_background_distance_m": float(distances.min()),
        "joint_mask_and_feature_validity": True, "positive_traceability": True,
        "no_background_positive_overlap": True, "exclusion_radius_respected": True,
        "aspect_components_valid": True, "original_land_cover_codes_preserved": True,
    }


def audit_geometry(spatial: gpd.GeoDataFrame, table: pd.DataFrame, inputs: Inputs) -> None:
    require(spatial.crs == inputs.grid.crs, "GPKG CRS differs from the reference CRS.")
    require(len(spatial) == len(table), "GPKG record count differs from CSV.")
    require(not spatial.geometry.isna().any() and not spatial.geometry.is_empty.any()
            and spatial.geometry.is_valid.all() and spatial.geom_type.eq("Point").all(),
            "GPKG must contain valid, nonempty Point geometries.")
    require(spatial.geometry.to_wkb().is_unique, "GPKG has duplicate point geometries.")
    compare_tables(pd.DataFrame(spatial.drop(columns=spatial.geometry.name)), table,
                   "GPKG attributes differ from CSV")
    require(np.allclose(np.column_stack((spatial.geometry.x, spatial.geometry.y)),
                        table[["x", "y"]].to_numpy(), rtol=0, atol=1e-8),
            "GPKG geometries differ from the recorded cell centers.")


def audit_dataset(project_root: Path = PROJECT_ROOT) -> dict:
    output_dir = project_root / "data/processed"
    metadata = json.loads((output_dir / OUTPUT_NAMES["metadata"]).read_text(encoding="utf-8"))
    require(metadata.get("schema_version") == 1, "Unsupported metadata schema.")
    timestamp = datetime.fromisoformat(metadata["generated_at_utc"])
    require(timestamp.utcoffset() is not None and timestamp.utcoffset().total_seconds() == 0,
            "Generation timestamp must specify UTC.")
    sources = source_manifest(project_root)
    require(metadata["sources"] == sources, "Source paths/hashes differ from generation inputs.")
    require(metadata["features"] == [{"name": n, "path": p} for n, p in FEATURE_PATHS.items()],
            "Incorrect feature manifest.")
    require(metadata["sampling_strategy"] == STRATEGY, "Unexpected sampling strategy.")
    require(metadata["columns"] == COLUMNS and metadata["traceability_columns"] == TRACE_COLUMNS
            and metadata["predictor_columns"] == PREDICTOR_COLUMNS, "Incorrect column roles.")
    require(metadata["outputs"] == OUTPUT_NAMES and metadata["gpkg_layer"] == LAYER_NAME,
            "Incorrect output manifest.")
    inputs = load_inputs(project_root)
    require(metadata["grid"] == grid_metadata(inputs.grid), "Incorrect reference grid metadata.")
    table = pd.read_csv(output_dir / OUTPUT_NAMES["csv"],
                        dtype={"sample_id": str, "sample_type": str, "landslide_ids": str},
                        float_precision="round_trip")
    radius, seed = metadata["exclusion_radius_m"], metadata["random_seed"]
    report = audit_table(table, inputs, radius)
    background_count = int(table.target.eq(0).sum())
    expected, counts = assemble_dataset(inputs, radius, seed, background_count)
    repeated, repeated_counts = assemble_dataset(inputs, radius, seed, background_count)
    compare_tables(repeated, expected, "Same-seed regeneration is not reproducible")
    compare_tables(table, expected, "Saved dataset differs from same-seed regeneration")
    require(counts == repeated_counts and metadata["counts"] == counts, "Incorrect sample/mask counts.")
    layers = gpd.list_layers(output_dir / OUTPUT_NAMES["gpkg"])
    require(layers["name"].tolist() == [LAYER_NAME], "Unexpected GPKG layers.")
    spatial = gpd.read_file(output_dir / OUTPUT_NAMES["gpkg"], layer=LAYER_NAME)
    audit_geometry(spatial, table, inputs)
    require(source_manifest(project_root) == sources, "Inputs changed during the audit.")
    report.update({"same_seed_reproducible": True, "metadata_and_source_hashes_valid": True,
                   "gpkg_geometry_crs_and_attributes_valid": True, "status": "passed"})
    return report


def main() -> None:
    print(json.dumps(audit_dataset(), indent=2, allow_nan=False))
    print("Dataset audit passed. No files were written.")


if __name__ == "__main__":
    main()
