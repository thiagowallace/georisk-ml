"""Audit NDVI coverage against the actual valid DEM footprint, read-only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEM_FILE = PROJECT_ROOT / "data/interim/dem/santa_tereza_dem_30m.tif"
NDVI_FILE = PROJECT_ROOT / "data/interim/ndvi/santa_tereza_ndvi_pre_event_2024.tif"
MAX_COORDINATES = 20


def compare_grids(dem, ndvi) -> bool:
    checks = (
        ("CRS", dem.crs, ndvi.crs),
        ("transform", dem.transform, ndvi.transform),
        ("width", dem.width, ndvi.width),
        ("height", dem.height, ndvi.height),
        ("bounds", dem.bounds, ndvi.bounds),
        ("resolution", dem.res, ndvi.res),
    )
    matches = True
    for label, left, right in checks:
        equal = left == right
        matches = matches and equal
        print(f"Same {label}: {equal}; DEM={left}; NDVI={right}")
    print(f"Grid matches DEM: {matches}")
    return matches


def read_valid(src) -> tuple[np.ndarray, np.ndarray]:
    """Honor declared nodata, dataset masks and nonfinite values."""
    if src.count != 1:
        raise ValueError("Expected a single-band raster.")
    band = src.read(1, masked=True)
    data = np.asarray(band.data)
    valid = ~np.ma.getmaskarray(band) & np.isfinite(data)
    # Explicit nodata check also covers datasets whose internal mask takes
    # precedence over nodata when rasterio constructs the masked array.
    if src.nodata is not None:
        valid &= ~np.isnan(data) if np.isnan(src.nodata) else data != src.nodata
    return data, valid


def numpy_edge_distance(valid: np.ndarray) -> np.ndarray:
    """Exact Euclidean distances up to 3 pixels; larger distances are inf."""
    height, width = valid.shape
    padded = np.pad(valid, 3, constant_values=False)
    distance = np.full(valid.shape, np.inf, dtype=np.float64)
    # Disk-shaped offsets reproduce EDT thresholds, including diagonal
    # distances. Array exterior and internal nodata holes are both invalid.
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            squared = dr * dr + dc * dc
            if squared > 9:
                continue
            neighbor_invalid = ~padded[3 + dr:3 + dr + height,
                                       3 + dc:3 + dc + width]
            np.minimum(distance, np.sqrt(squared), out=distance,
                       where=neighbor_invalid)
    return distance


def edge_distance(valid: np.ndarray) -> np.ndarray:
    if distance_transform_edt is None:
        print("Edge distance method: NumPy Euclidean offsets (exact up to 3 pixels).")
        return numpy_edge_distance(valid)
    print("Edge distance method: scipy.ndimage.distance_transform_edt.")
    # Pad explicitly: the array exterior is invalid even for a fully valid DEM.
    return distance_transform_edt(np.pad(valid, 1, constant_values=False))[1:-1, 1:-1]


def print_extreme_rows_columns(dem_valid: np.ndarray, missing: np.ndarray) -> None:
    rows = np.flatnonzero(dem_valid.any(axis=1))
    cols = np.flatnonzero(dem_valid.any(axis=0))
    if not rows.size:
        print("DEM valid footprint is empty; first/last valid rows and columns do not exist.")
        return
    first_row, last_row = int(rows[0]), int(rows[-1])
    first_col, last_col = int(cols[0]), int(cols[-1])
    print(f"DEM valid footprint extreme rows: {first_row}, {last_row}")
    print(f"DEM valid footprint extreme columns: {first_col}, {last_col}")
    row_count = np.count_nonzero(missing[np.unique([first_row, last_row]), :])
    col_count = np.count_nonzero(missing[:, np.unique([first_col, last_col])])
    print(f"Missing NDVI on first or last valid-footprint row: {row_count}")
    print(f"Missing NDVI on first or last valid-footprint column: {col_count}")
    # Also report directional extrema along each transect of an irregular
    # footprint. Distance diagnostics below additionally include internal holes.
    column_edges = np.zeros_like(dem_valid)
    column_edges[np.argmax(dem_valid[:, cols], axis=0), cols] = True
    column_edges[dem_valid.shape[0] - 1 - np.argmax(dem_valid[::-1, cols], axis=0), cols] = True
    row_edges = np.zeros_like(dem_valid)
    row_edges[rows, np.argmax(dem_valid[rows, :], axis=1)] = True
    row_edges[rows, dem_valid.shape[1] - 1 - np.argmax(dem_valid[rows, ::-1], axis=1)] = True
    print("Missing NDVI on first/last valid row within each column: "
          f"{np.count_nonzero(missing & column_edges)}")
    print("Missing NDVI on first/last valid column within each row: "
          f"{np.count_nonzero(missing & row_edges)}")


def audit_coverage(dem_valid: np.ndarray, ndvi_valid: np.ndarray, transform) -> None:
    both = dem_valid & ndvi_valid
    missing = dem_valid & ~ndvi_valid
    extra = ndvi_valid & ~dem_valid
    dem_count, ndvi_count = np.count_nonzero(dem_valid), np.count_nonzero(ndvi_valid)
    missing_count, extra_count = np.count_nonzero(missing), np.count_nonzero(extra)
    print(f"\nDEM valid pixels: {dem_count}")
    print(f"NDVI valid pixels: {ndvi_count}")
    print(f"Valid in both: {np.count_nonzero(both)}")
    print(f"DEM valid AND NDVI NoData/invalid: {missing_count}")
    print(f"NDVI valid AND DEM NoData/invalid: {extra_count}")
    print(f"Valid-count difference DEM - NDVI: {dem_count - ndvi_count} "
          f"= missing ({missing_count}) - extra ({extra_count})")
    print("NoData/invalid includes declared nodata, raster masks and nonfinite values.")
    print_extreme_rows_columns(dem_valid, missing)
    distances = edge_distance(dem_valid)
    print("Distance convention: Euclidean distance in pixel units to the nearest "
          "invalid pixel center, including holes and the array exterior. "
          "A valid pixel sharing a side with nodata has distance 1.")
    print("The <= 1, <= 2 and <= 3 counts are cumulative and must not be added.")
    for radius in (1, 2, 3):
        count = np.count_nonzero(missing & (distances <= radius))
        print(f"Missing NDVI within {radius} pixel(s) of actual valid DEM edge: {count}")
    internal = missing & (distances > 3)
    internal_count = np.count_nonzero(internal)
    near_count = missing_count - internal_count
    print(f"Clearly internal missing NDVI (> 3 pixels from edge): {internal_count}")
    if internal_count:
        print(f"First {min(MAX_COORDINATES, internal_count)} internal gaps "
              "(zero-based row/column; pixel-center X/Y in DEM CRS):")
        # Row-major order, without building a coordinate list for every gap.
        emitted = 0
        for row in np.flatnonzero(internal.any(axis=1)):
            for col in np.flatnonzero(internal[row]):
                x, y = rasterio.transform.xy(transform, int(row), int(col), offset="center")
                print(f"row={row}, col={col}, X={x:.6f}, Y={y:.6f}")
                emitted += 1
                if emitted == MAX_COORDINATES:
                    break
            if emitted == MAX_COORDINATES:
                break
    if missing_count:
        print(f"Spatial explanation: {near_count}/{missing_count} missing pixels "
              f"({100.0 * near_count / missing_count:.2f}%) are within 3 pixels of "
              f"the actual DEM edge; {internal_count} are farther inside.")
        if internal_count:
            print("The NDVI coverage deficit includes internal gaps, so it is not "
                  "confined to the three-pixel edge zone.")
        else:
            print("The NDVI coverage deficit is confined to the three-pixel edge zone.")
        print("These two rasters establish the spatial pattern, but cannot establish "
              "whether gaps originate from source coverage, SCL masking, radiometric "
              "validation, buffer clipping or reprojection.")
    else:
        print("Every valid DEM pixel has valid NDVI coverage.")
    if extra_count:
        print("Coverage inconsistency: valid NDVI exists outside the valid DEM mask.")


def audit_ndvi_values(data: np.ndarray, valid: np.ndarray) -> None:
    values = data[valid]
    if not values.size:
        print("NDVI min/max/mean/median: unavailable (no valid pixels).")
        print("NDVI min >= -1: not evaluable\nNDVI max <= 1: not evaluable")
        raise ValueError("NDVI contains no valid pixels; physical range cannot be validated.")
    minimum, maximum = float(values.min()), float(values.max())
    print(f"\nNDVI min: {minimum:.10g}\nNDVI max: {maximum:.10g}")
    print(f"NDVI mean: {values.mean(dtype=np.float64):.10g}")
    print(f"NDVI median: {np.median(values):.10g}")
    print(f"NDVI min >= -1: {minimum >= -1}")
    print(f"NDVI max <= 1: {maximum <= 1}")
    outside = np.count_nonzero((values < -1) | (values > 1))
    print(f"NDVI valid pixels outside [-1, 1]: {outside}")
    if outside:
        raise ValueError("NDVI physical range validation failed.")


def main() -> None:
    with rasterio.open(DEM_FILE, "r") as dem, rasterio.open(NDVI_FILE, "r") as ndvi:
        print(f"DEM: {DEM_FILE}\nNDVI: {NDVI_FILE}")
        if not compare_grids(dem, ndvi):
            raise ValueError("Raster grids differ; pixelwise coverage audit is not valid.")
        if dem.crs is None:
            raise ValueError("DEM CRS is missing; spatial coordinates cannot be identified.")
        print(f"DEM NoData: {dem.nodata}\nNDVI NoData: {ndvi.nodata}")
        _, dem_valid = read_valid(dem)
        ndvi_data, ndvi_valid = read_valid(ndvi)
        audit_coverage(dem_valid, ndvi_valid, dem.transform)
        audit_ndvi_values(ndvi_data, ndvi_valid)
    print("Spatial coverage audit completed; no files were written.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, rasterio.errors.RasterioError) as exc:
        raise SystemExit(f"Pre-event NDVI audit failed: {exc}") from exc
