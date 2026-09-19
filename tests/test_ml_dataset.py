"""Deterministic unit checks plus a small synthetic file round-trip."""

from dataclasses import replace
import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from src.features.build_ml_dataset import (
    FEATURE_PATHS, OUTPUT_NAMES, PREDICTOR_COLUMNS, TRACE_COLUMNS, VECTOR_PATHS,
    Grid, Inputs, assemble_dataset, assert_grid_matches, background_candidates,
    extract_features, joint_valid_mask, load_inputs, make_metadata,
    positive_cells, sample_background, source_manifest, write_products,
)
from src.transform.audit_ml_dataset import audit_dataset, audit_geometry, audit_table


@pytest.fixture
def inputs():
    grid = Grid(rasterio.crs.CRS.from_epsg(31982), from_origin(0, 150, 30, 30), 6, 5)
    features = {
        "elevation": np.full(grid.shape, 100, dtype=np.float32),
        "slope": np.full(grid.shape, 20, dtype=np.float32),
        "aspect": (np.arange(30).reshape(grid.shape) % 4 * 90).astype(np.float32),
        "plan_curvature": np.full(grid.shape, 0.01, dtype=np.float32),
        "profile_curvature": np.full(grid.shape, -0.02, dtype=np.float32),
        "land_cover": np.where(np.indices(grid.shape)[1] % 2, 9, 3).astype(np.uint8),
        "ndvi_pre_event": np.full(grid.shape, 0.7, dtype=np.float32),
    }
    occurrences = gpd.GeoDataFrame(
        {"landslide_id": [9, 2, 7]}, geometry=[Point(14, 136), Point(16, 134), Point(75, 75)],
        crs=grid.crs,
    )
    valid = np.ones(grid.shape, dtype=bool)
    valid[4, 5] = False
    exclusions = np.array([[14, 136], [16, 134], [75, 75], [-1, 135]], dtype=float)
    return Inputs(grid, features, valid, occurrences, exclusions, np.ones(grid.shape, dtype=bool))


def test_occurrences_aggregate_by_cell_and_retain_all_ids(inputs):
    result = positive_cells(inputs.occurrences, inputs.grid, inputs.valid_mask)
    assert result.to_dict("records") == [
        {"cell_id": 0, "landslide_count": 2, "landslide_ids": "[2,9]"},
        {"cell_id": 14, "landslide_count": 1, "landslide_ids": "[7]"},
    ]
    reordered = positive_cells(inputs.occurrences.iloc[::-1], inputs.grid, inputs.valid_mask)
    pd.testing.assert_frame_equal(result, reordered)


def test_positive_invalid_cell_fails_without_dropping_occurrences(inputs):
    mask = inputs.valid_mask.copy()
    mask[0, 0] = False
    with pytest.raises(ValueError, match="no silent exclusion"):
        positive_cells(inputs.occurrences, inputs.grid, mask)


def test_positive_outside_grid_fails(inputs):
    points = inputs.occurrences.copy()
    points.loc[0, "geometry"] = Point(-1, 150)
    with pytest.raises(ValueError, match="outside the reference grid"):
        positive_cells(points, inputs.grid, inputs.valid_mask)


def test_duplicate_occurrence_id_fails(inputs):
    points = inputs.occurrences.copy()
    points["landslide_id"] = [9, 9, 7]
    with pytest.raises(ValueError, match="unique, non-null integers"):
        positive_cells(points, inputs.grid, inputs.valid_mask)


def test_joint_mask_requires_all_features_and_inside_center(inputs):
    features = {name: value.copy() for name, value in inputs.features.items()}
    validity = {name: np.ones(inputs.grid.shape, dtype=bool) for name in FEATURE_PATHS}
    municipality = np.ones(inputs.grid.shape, dtype=bool)
    municipality[0, 0] = False
    features["ndvi_pre_event"][0, 1] = np.nan
    features["slope"][0, 2] = np.inf
    features["elevation"][0, 3] = -9999
    validity["elevation"][0, 3] = False
    features["land_cover"][0, 4] = 0
    validity["land_cover"][0, 4] = False
    mask = joint_valid_mask(municipality, features, validity)
    assert not mask[0, :5].any()
    assert mask.sum() == 25
    del features["aspect"]
    with pytest.raises(ValueError, match="seven environmental features"):
        joint_valid_mask(municipality, features, validity)


def test_exact_radius_is_eligible_and_positive_cells_always_excluded(inputs):
    grid = replace(inputs.grid, width=3, height=1, transform=from_origin(0, 30, 30, 30))
    mask = np.ones(grid.shape, dtype=bool)
    points = np.array([[15.0, 15.0]])
    np.testing.assert_array_equal(background_candidates(mask, np.array([0]), grid, points, 30), [1, 2])
    np.testing.assert_array_equal(background_candidates(mask, np.array([0]), grid, points, 30.01), [2])
    np.testing.assert_array_equal(background_candidates(mask, np.array([0]), grid, points, 0), [1, 2])


def test_external_occurrence_excludes_cells_near_municipal_border(inputs):
    grid = replace(inputs.grid, width=3, height=1, transform=from_origin(0, 30, 30, 30))
    candidates = background_candidates(np.ones(grid.shape, bool), np.array([2]), grid,
                                       np.array([[-1.0, 15.0]]), 50)
    assert candidates.size == 0  # Distances are 16 and 46 m; the other cell is positive.


@pytest.mark.parametrize("radius", [-1, np.nan, np.inf])
def test_invalid_radius_rejected(inputs, radius):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        background_candidates(inputs.valid_mask, np.array([0]), inputs.grid, inputs.exclusion_xy, radius)


def test_sampling_is_reproducible_without_replacement_and_order_independent():
    candidates = np.arange(100)
    first = sample_background(candidates, 30, 42)
    np.testing.assert_array_equal(first, sample_background(candidates[::-1], 30, 42))
    assert len(np.unique(first)) == 30
    assert not np.array_equal(first, sample_background(candidates, 30, 43))
    with pytest.raises(ValueError, match="only 100 available"):
        sample_background(candidates, 101, 42)


@pytest.mark.parametrize("count,seed", [(0, 42), (-1, 42), (1.5, 42), (1, -1), (1, 2.5)])
def test_invalid_sampling_parameters_rejected(count, seed):
    with pytest.raises(ValueError):
        sample_background(np.arange(10), count, seed)


def test_aspect_components_and_original_categorical_codes(inputs):
    table = extract_features(np.array([0, 1, 2, 3]), inputs)
    np.testing.assert_allclose(table.aspect_sin, [0, 1, 0, -1], atol=1e-15)
    np.testing.assert_allclose(table.aspect_cos, [1, 0, -1, 0], atol=1e-15)
    assert table.land_cover.tolist() == [3, 9, 3, 9]
    assert not set(PREDICTOR_COLUMNS).intersection(TRACE_COLUMNS)
    assert "aspect" not in PREDICTOR_COLUMNS  # Raw degrees remain in the dataset.


def test_assembly_default_balance_and_traceability(inputs):
    table, counts = assemble_dataset(inputs, radius=30)
    assert counts["original_occurrences"] == 3
    assert counts["positive_cells"] == counts["background_cells"] == 2
    assert counts["total_samples"] == 4
    assert table.cell_id.is_unique
    assert table.landslide_count.sum() == 3
    assert table.loc[table.target.eq(0), "landslide_ids"].eq("[]").all()
    report = audit_table(table, inputs, radius=30)
    assert report["minimum_background_distance_m"] >= 30


def test_custom_background_count(inputs):
    table, counts = assemble_dataset(inputs, radius=0, seed=7, background_count=5)
    assert counts["background_cells"] == 5
    assert len(table) == 7


@pytest.mark.parametrize("column,value", [
    ("elevation", np.nan), ("elevation", -9999), ("land_cover", 99),
    ("aspect_sin", 2.0), ("aspect_cos", 0.4), ("x", -1), ("landslide_ids", "[]"),
])
def test_audit_detects_corrupted_attributes(inputs, column, value):
    table, _ = assemble_dataset(inputs, radius=30)
    table.loc[0, column] = value
    with pytest.raises(ValueError):
        audit_table(table, inputs, radius=30)


def test_audit_detects_duplicate_cell(inputs):
    table, _ = assemble_dataset(inputs, radius=30)
    table.loc[2, "cell_id"] = table.loc[0, "cell_id"]
    with pytest.raises(ValueError, match="Duplicate cell_id"):
        audit_table(table, inputs, radius=30)


def test_audit_independently_detects_background_inside_radius(inputs):
    table, _ = assemble_dataset(inputs, radius=30)
    center = table.loc[table.target.eq(0), ["x", "y"]].iloc[0].to_numpy()
    changed = replace(inputs, exclusion_xy=np.vstack([inputs.exclusion_xy, center]))
    with pytest.raises(ValueError, match="inside the exclusion radius"):
        audit_table(table, changed, radius=30)


def test_geometry_audit_rejects_displaced_points_and_wrong_crs(inputs):
    table, _ = assemble_dataset(inputs, radius=30)
    spatial = gpd.GeoDataFrame(table.copy(), geometry=gpd.points_from_xy(table.x, table.y), crs=inputs.grid.crs)
    audit_geometry(spatial, table, inputs)
    wrong_crs = spatial.set_crs(4326, allow_override=True)
    with pytest.raises(ValueError, match="CRS"):
        audit_geometry(wrong_crs, table, inputs)
    spatial.loc[0, "geometry"] = Point(0, 0)
    with pytest.raises(ValueError, match="cell centers"):
        audit_geometry(spatial, table, inputs)


def test_grid_alignment_rejects_half_pixel_shift(inputs):
    grid = inputs.grid
    source = SimpleNamespace(count=1, crs=grid.crs, transform=grid.transform,
                             width=grid.width, height=grid.height, name="synthetic")
    assert_grid_matches(source, grid)
    source.transform = from_origin(15, 150, 30, 30)
    with pytest.raises(ValueError, match="exactly match"):
        assert_grid_matches(source, grid)


def write_synthetic_inputs(root, inputs):
    """Create a tiny isolated fixture only under pytest's temporary directory."""
    for name, path in FEATURE_PATHS.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        values = inputs.features[name].copy()
        nodata = 0 if name == "land_cover" else -9999
        values[~inputs.valid_mask] = nodata
        with rasterio.open(destination, "w", driver="GTiff", count=1, dtype=values.dtype,
                           crs=inputs.grid.crs, transform=inputs.grid.transform,
                           width=inputs.grid.width, height=inputs.grid.height, nodata=nodata) as dst:
            dst.write(values, 1)
    vectors = {
        "study_area": gpd.GeoDataFrame(geometry=[box(0, 0, 180, 150)], crs=inputs.grid.crs),
        "occurrences": inputs.occurrences,
        "exclusion_inventory": gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in inputs.exclusion_xy], crs=inputs.grid.crs),
    }
    for name, frame in vectors.items():
        path = root / VECTOR_PATHS[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_file(path, layer=name, driver="GPKG", index=False)


def test_synthetic_saved_products_and_audit(tmp_path, inputs):
    write_synthetic_inputs(tmp_path, inputs)
    loaded = load_inputs(tmp_path)
    table, counts = assemble_dataset(loaded, radius=30)
    metadata = make_metadata(loaded, counts, 30, 42, source_manifest(tmp_path))
    write_products(table, metadata, loaded.grid, project_root=tmp_path)
    report = audit_dataset(tmp_path)
    assert report["status"] == "passed"
    assert report["same_seed_reproducible"]
    assert report["gpkg_geometry_crs_and_attributes_valid"]
    with pytest.raises(FileExistsError):
        write_products(table, metadata, loaded.grid, project_root=tmp_path)
    metadata_path = tmp_path / "data/processed" / OUTPUT_NAMES["metadata"]
    corrupted = json.loads(metadata_path.read_text(encoding="utf-8"))
    corrupted["counts"]["positive_cells"] += 1
    metadata_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="Incorrect sample/mask counts"):
        audit_dataset(tmp_path)


def test_load_inputs_excludes_center_on_polygon_boundary(tmp_path, inputs):
    write_synthetic_inputs(tmp_path, inputs)
    path = tmp_path / VECTOR_PATHS["study_area"]
    # The leftmost cell centers have x=15 and lie exactly on this boundary.
    area = gpd.GeoDataFrame(geometry=[box(15, 0, 180, 150)], crs=inputs.grid.crs)
    path.unlink()
    area.to_file(path, layer="study_area", driver="GPKG", index=False)
    # Keep the input points inside this smaller municipality for this mask check.
    points = inputs.occurrences.copy()
    points.loc[0, "geometry"] = Point(16, 136)
    point_path = tmp_path / VECTOR_PATHS["occurrences"]
    point_path.unlink()
    points.to_file(point_path, layer="occurrences", driver="GPKG", index=False)
    loaded = load_inputs(tmp_path)
    assert not loaded.municipality_mask[:, 0].any()
    with pytest.raises(ValueError, match="no silent exclusion"):
        assemble_dataset(loaded, radius=0)
