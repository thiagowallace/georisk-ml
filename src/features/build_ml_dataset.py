"""Build a reproducible, cell-based landslide/background sample dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy.spatial import cKDTree
from shapely import contains_xy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATHS = {
    "elevation": "data/interim/dem/santa_tereza_dem_30m.tif",
    "slope": "data/interim/terrain_features/santa_tereza_slope.tif",
    "aspect": "data/interim/terrain_features/santa_tereza_aspect.tif",
    "plan_curvature": "data/interim/terrain_features/santa_tereza_plan_curvature.tif",
    "profile_curvature": "data/interim/terrain_features/santa_tereza_profile_curvature.tif",
    "land_cover": "data/interim/land_cover/santa_tereza_land_cover_2023.tif",
    "ndvi_pre_event": "data/interim/ndvi/santa_tereza_ndvi_pre_event_2024.tif",
}
VECTOR_PATHS = {
    "study_area": "data/interim/study_area/santa_tereza.gpkg",
    "occurrences": "data/interim/study_area/landslide_initiation_santa_tereza.gpkg",
    "exclusion_inventory": "data/interim/landslides_rs_2024/landslide_initiation.gpkg",
}
TRACE_COLUMNS = [
    "sample_id", "cell_id", "row", "col", "x", "y",
    "landslide_count", "landslide_ids",
]
PREDICTOR_COLUMNS = [
    "elevation", "slope", "aspect_sin", "aspect_cos", "plan_curvature",
    "profile_curvature", "land_cover", "ndvi_pre_event",
]
COLUMNS = TRACE_COLUMNS + ["target", "sample_type"] + list(FEATURE_PATHS) + [
    "aspect_sin", "aspect_cos",
]
OUTPUT_NAMES = {
    "csv": "santa_tereza_ml_dataset.csv",
    "gpkg": "santa_tereza_ml_samples.gpkg",
    "metadata": "santa_tereza_ml_dataset_metadata.json",
}
LAYER_NAME = "ml_samples"
DEFAULT_SEED = 42
DEFAULT_RADIUS = 60.0
STRATEGY = {
    "unit": "reference_grid_cell",
    "municipality_rule": "cell center strictly inside polygon; boundary excluded",
    "validity_rule": "all seven features finite, unmasked and different from declared NoData",
    "positive_rule": "one row per occupied cell; retain all original landslide_id values",
    "background_rule": "joint mask minus positive cells; center distance >= exclusion radius",
    "distance": "Euclidean meters to union of municipal and complete statewide point inventories",
    "sampling": "uniform without replacement from ascending cell_id candidates",
    "rng": "numpy.random.Generator(PCG64)",
    "default_balance": "one background cell per positive cell",
    "ordering": "landslide then background; ascending cell_id within each class",
    "label_caveat": "background is pseudoabsence, not confirmed absence of landslides",
}


@dataclass(frozen=True)
class Grid:
    crs: rasterio.crs.CRS
    transform: rasterio.Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass
class Inputs:
    grid: Grid
    features: dict[str, np.ndarray]
    valid_mask: np.ndarray
    occurrences: gpd.GeoDataFrame
    exclusion_xy: np.ndarray
    municipality_mask: np.ndarray


def cell_centers(cell_ids: np.ndarray, grid: Grid) -> np.ndarray:
    rows, cols = np.divmod(np.asarray(cell_ids, dtype=np.int64), grid.width)
    x, y = grid.transform * (cols + 0.5, rows + 0.5)
    return np.column_stack((x, y))


def assert_grid_matches(dataset, grid: Grid) -> None:
    if (dataset.count != 1 or dataset.crs != grid.crs
            or dataset.transform != grid.transform
            or dataset.width != grid.width or dataset.height != grid.height):
        raise ValueError(f"Raster does not exactly match the reference grid: {dataset.name}")


def joint_valid_mask(
    municipality_mask: np.ndarray,
    features: dict[str, np.ndarray],
    validity: dict[str, np.ndarray],
) -> np.ndarray:
    if set(features) != set(FEATURE_PATHS) or set(validity) != set(FEATURE_PATHS):
        raise ValueError("Exactly the seven environmental features are required.")
    result = municipality_mask.astype(bool, copy=True)
    for name, data in features.items():
        if data.shape != result.shape or validity[name].shape != result.shape:
            raise ValueError(f"Invalid array shape for {name}.")
        result &= validity[name] & np.isfinite(data)
    return result


def validate_points(points: gpd.GeoDataFrame, grid: Grid, label: str) -> None:
    if points.empty or points.crs != grid.crs:
        raise ValueError(f"{label} must be nonempty and use the reference CRS.")
    if (points.geometry.isna().any() or points.geometry.is_empty.any()
            or not points.geometry.is_valid.all()
            or not points.geom_type.eq("Point").all()):
        raise ValueError(f"{label} must contain valid, nonempty Point geometries.")
    if not np.isfinite(np.column_stack((points.geometry.x, points.geometry.y))).all():
        raise ValueError(f"{label} contains nonfinite coordinates.")


def load_inputs(project_root: Path = PROJECT_ROOT) -> Inputs:
    """Read source data without reprojection or modifications to any input."""
    with rasterio.open(project_root / FEATURE_PATHS["elevation"]) as reference:
        grid = Grid(reference.crs, reference.transform, reference.width, reference.height)
        if (grid.crs is None or grid.crs.to_epsg() != 31982
                or grid.transform.a != 30 or grid.transform.e != -30
                or grid.transform.b != 0 or grid.transform.d != 0):
            raise ValueError("Expected a north-up EPSG:31982 reference grid of 30 x 30 m.")
    features, validity = {}, {}
    for name, path in FEATURE_PATHS.items():
        with rasterio.open(project_root / path) as source:
            assert_grid_matches(source, grid)
            band = source.read(1, masked=True)
            data = np.asarray(band.data)
            valid = ~np.ma.getmaskarray(band) & np.isfinite(data)
            if source.nodata is not None:
                valid &= data != source.nodata
            features[name], validity[name] = data, valid

    area = gpd.read_file(project_root / VECTOR_PATHS["study_area"])
    if (len(area) != 1 or area.crs != grid.crs or area.geometry.isna().any()
            or area.geometry.is_empty.any() or not area.geometry.is_valid.all()
            or not area.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Expected one valid municipal polygon in the reference CRS.")
    polygon = area.geometry.union_all()
    centers = cell_centers(np.arange(grid.width * grid.height), grid)
    municipality_mask = contains_xy(polygon, centers[:, 0], centers[:, 1]).reshape(grid.shape)
    valid_mask = joint_valid_mask(municipality_mask, features, validity)
    if not valid_mask.any():
        raise ValueError("No cells remain in the joint valid mask.")

    local = gpd.read_file(project_root / VECTOR_PATHS["occurrences"])
    statewide = gpd.read_file(project_root / VECTOR_PATHS["exclusion_inventory"])
    validate_points(local, grid, "Municipal occurrences")
    validate_points(statewide, grid, "Statewide inventory")
    if not local.geometry.intersects(polygon).all():
        raise ValueError("Municipal inventory includes points outside the municipality.")
    # Include both sources even if a local point is missing from the statewide file.
    exclusion_xy = np.unique(np.vstack([
        np.column_stack((local.geometry.x, local.geometry.y)),
        np.column_stack((statewide.geometry.x, statewide.geometry.y)),
    ]), axis=0)
    return Inputs(grid, features, valid_mask, local, exclusion_xy, municipality_mask)


def positive_cells(occurrences: gpd.GeoDataFrame, grid: Grid, valid_mask: np.ndarray) -> pd.DataFrame:
    """Aggregate occurrences; fail instead of silently dropping any original point."""
    validate_points(occurrences, grid, "Occurrences")
    if ("landslide_id" not in occurrences
            or not pd.api.types.is_integer_dtype(occurrences["landslide_id"])
            or occurrences["landslide_id"].isna().any()
            or not occurrences["landslide_id"].is_unique):
        raise ValueError("landslide_id must contain unique, non-null integers.")
    rows, cols = rowcol(grid.transform, occurrences.geometry.x, occurrences.geometry.y)
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    if ((rows < 0) | (rows >= grid.height) | (cols < 0) | (cols >= grid.width)).any():
        raise ValueError("An occurrence lies outside the reference grid.")
    usable = valid_mask[rows, cols]
    if not usable.all():
        rejected = occurrences.loc[~usable, "landslide_id"].tolist()
        raise ValueError(f"Occurrences outside the joint mask; no silent exclusion: {rejected}")
    groups = pd.DataFrame({
        "cell_id": rows * grid.width + cols,
        "landslide_id": occurrences["landslide_id"].to_numpy(),
    }).groupby("cell_id", sort=True)["landslide_id"]
    records = []
    for cell_id, ids in groups:
        values = sorted(int(value) for value in ids)
        records.append({"cell_id": int(cell_id), "landslide_count": len(values),
                        "landslide_ids": json.dumps(values, separators=(",", ":"))})
    return pd.DataFrame(records)


def background_candidates(
    valid_mask: np.ndarray, positive_ids: np.ndarray, grid: Grid,
    exclusion_xy: np.ndarray, radius: float,
) -> np.ndarray:
    """Return sorted eligible cell IDs; points exactly at radius remain eligible."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Exclusion radius must be finite and nonnegative.")
    points = np.asarray(exclusion_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Exclusion coordinates must be a nonempty finite N x 2 array.")
    candidates = np.setdiff1d(np.flatnonzero(valid_mask), positive_ids)
    if not candidates.size:
        return candidates
    distances, _ = cKDTree(points).query(cell_centers(candidates, grid), k=1, eps=0)
    return candidates[distances >= radius]


def sample_background(candidates: np.ndarray, count: int, seed: int) -> np.ndarray:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Background count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Random seed must be a nonnegative integer.")
    candidates = np.unique(np.asarray(candidates, dtype=np.int64))
    if count > len(candidates):
        raise ValueError(f"Requested {count} background cells; only {len(candidates)} available.")
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.sort(rng.choice(candidates, size=count, replace=False))


def extract_features(cell_ids: np.ndarray, inputs: Inputs) -> pd.DataFrame:
    rows, cols = np.divmod(cell_ids, inputs.grid.width)
    if not inputs.valid_mask[rows, cols].all():
        raise ValueError("Feature extraction includes cells outside the joint mask.")
    frame = pd.DataFrame({name: values[rows, cols] for name, values in inputs.features.items()})
    cover = frame["land_cover"].to_numpy()
    if not np.equal(cover, np.floor(cover)).all():
        raise ValueError("MapBiomas values must be original integer class codes.")
    frame["land_cover"] = cover.astype(np.int64)
    if not frame["aspect"].between(0, 360, inclusive="left").all():
        raise ValueError("Aspect must be in [0, 360) degrees.")
    if not frame["ndvi_pre_event"].between(-1, 1).all():
        raise ValueError("NDVI must be in [-1, 1].")
    radians = np.deg2rad(frame["aspect"].to_numpy(dtype=np.float64))
    frame["aspect_sin"], frame["aspect_cos"] = np.sin(radians), np.cos(radians)
    return frame


def assemble_dataset(
    inputs: Inputs, radius: float = DEFAULT_RADIUS, seed: int = DEFAULT_SEED,
    background_count: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    positives = positive_cells(inputs.occurrences, inputs.grid, inputs.valid_mask)
    candidates = background_candidates(inputs.valid_mask, positives.cell_id.to_numpy(),
                                       inputs.grid, inputs.exclusion_xy, radius)
    count = len(positives) if background_count is None else background_count
    background_ids = sample_background(candidates, count, seed)
    positives = positives.assign(target=1, sample_type="landslide")
    background = pd.DataFrame({"cell_id": background_ids, "landslide_count": 0,
                               "landslide_ids": "[]", "target": 0, "sample_type": "background"})
    table = pd.concat([positives, background], ignore_index=True)
    cell_ids = table.cell_id.to_numpy(dtype=np.int64)
    table["row"], table["col"] = np.divmod(cell_ids, inputs.grid.width)
    centers = cell_centers(cell_ids, inputs.grid)
    table["x"], table["y"] = centers[:, 0], centers[:, 1]
    table["sample_id"] = [f"sample_{cell_id:06d}" for cell_id in cell_ids]
    table = pd.concat([table, extract_features(cell_ids, inputs)], axis=1)[COLUMNS]
    counts = {
        "original_occurrences": len(inputs.occurrences),
        "positive_cells": len(positives), "background_cells": len(background),
        "total_samples": len(table), "municipality_cells": int(inputs.municipality_mask.sum()),
        "joint_valid_cells": int(inputs.valid_mask.sum()),
        "background_candidates": len(candidates),
    }
    return table, counts


def source_manifest(project_root: Path) -> dict:
    manifest = {}
    for name, relative in {**FEATURE_PATHS, **VECTOR_PATHS}.items():
        path = project_root / relative
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest[name] = {"path": relative, "sha256": digest.hexdigest()}
    return manifest


def grid_metadata(grid: Grid) -> dict:
    return {
        "crs": grid.crs.to_string(), "resolution": [grid.transform.a, -grid.transform.e],
        "width": grid.width, "height": grid.height, "transform": list(grid.transform)[:6],
        "cell_id_formula": "row * width + col; row and col are zero-based",
    }


def make_metadata(inputs: Inputs, counts: dict, radius: float, seed: int, sources: dict) -> dict:
    return {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "grid": grid_metadata(inputs.grid), "counts": counts,
        "exclusion_radius_m": radius, "random_seed": seed,
        "features": [{"name": name, "path": path} for name, path in FEATURE_PATHS.items()],
        "sources": sources, "sampling_strategy": STRATEGY,
        "columns": COLUMNS, "predictor_columns": PREDICTOR_COLUMNS,
        "traceability_columns": TRACE_COLUMNS,
        "land_cover_encoding": "original MapBiomas code; categorical; no one-hot encoding",
        "aspect_encoding": "raw degrees retained; sine/cosine calculated from radians",
        "outputs": OUTPUT_NAMES, "gpkg_layer": LAYER_NAME,
        "runtime": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "geopandas", "rasterio", "scipy", "shapely")
        }},
    }


def write_products(table: pd.DataFrame, metadata: dict, grid: Grid,
                   project_root: Path = PROJECT_ROOT, overwrite: bool = False) -> None:
    output_dir = project_root / "data/processed"
    existing = [name for name in OUTPUT_NAMES.values() if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Outputs already exist; use --overwrite explicitly: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage complete files before replacing outputs. Inputs are always read-only.
    with TemporaryDirectory(prefix="ml_dataset_", dir=output_dir) as temporary:
        staging = Path(temporary)
        table.to_csv(staging / OUTPUT_NAMES["csv"], index=False, encoding="utf-8", float_format="%.17g")
        spatial = gpd.GeoDataFrame(table.copy(), geometry=gpd.points_from_xy(table.x, table.y), crs=grid.crs)
        spatial.to_file(staging / OUTPUT_NAMES["gpkg"], layer=LAYER_NAME, driver="GPKG", index=False)
        (staging / OUTPUT_NAMES["metadata"]).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
        )
        for name in OUTPUT_NAMES.values():
            (staging / name).replace(output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclusion-radius", type=float, default=DEFAULT_RADIUS, help="Meters; default: 60")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--background-count", type=int, default=None, help="Default: number of positive cells")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    before = source_manifest(PROJECT_ROOT)
    inputs = load_inputs()
    table, counts = assemble_dataset(inputs, args.exclusion_radius, args.seed, args.background_count)
    if source_manifest(PROJECT_ROOT) != before:
        raise RuntimeError("Input files changed during dataset generation.")
    metadata = make_metadata(inputs, counts, args.exclusion_radius, args.seed, before)
    write_products(table, metadata, inputs.grid, overwrite=args.overwrite)
    print(json.dumps(counts, indent=2))
    print("Dataset written to data/processed. Background labels represent pseudoabsences.")


if __name__ == "__main__":
    main()
