"""Build a seven-scene median NDVI on the exact reference DEM grid."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.session import DummySession
from rasterio.warp import reproject
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_AREA = PROJECT_ROOT / "data/interim/study_area/santa_tereza.gpkg"
DEM_FILE = PROJECT_ROOT / "data/interim/dem/santa_tereza_dem_30m.tif"
OUTPUT_FILE = PROJECT_ROOT / "data/interim/ndvi/santa_tereza_ndvi_pre_event_2024.tif"
STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
COLLECTION = "sentinel-2-l2a"
S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
SELECTED_DATES = (
    "2024-03-02", "2024-03-07", "2024-03-12", "2024-03-27",
    "2024-04-01", "2024-04-06", "2024-04-21",
)
# Tile established by the representative scene used in the radiometric audit.
TILE = "T22JDN"
BUFFER_METERS = 1000
MAX_LOCAL_INVALID_PERCENT = 1.0
INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)
OUTPUT_NODATA = -9999.0


def credentials_from_environment() -> dict:
    names = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError("Missing S3 environment credentials: " + ", ".join(missing))
    return {name: os.environ[name] for name in names}


def load_buffer() -> gpd.GeoSeries:
    area = gpd.read_file(STUDY_AREA)
    if area.crs is None or area.empty:
        raise ValueError("Study area must have a CRS and nonempty polygons.")
    if (area.geometry.isna().any() or area.geometry.is_empty.any()
            or not area.geometry.is_valid.all()
            or not area.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Study area contains missing or invalid polygons.")
    area = area.to_crs("EPSG:31982")
    return gpd.GeoSeries([area.geometry.buffer(BUFFER_METERS).union_all()], crs=area.crs)


def select_scenes(items: list[dict]) -> list[dict]:
    """Never silently select among reprocessed or duplicate acquisitions."""
    selected = []
    for date in SELECTED_DATES:
        matches = [item for item in items
                   if item.get("collection") == COLLECTION
                   and (item.get("properties", {}).get("datetime") or "")[:10] == date
                   and f"_{TILE}_" in item.get("id", "")]
        if len(matches) != 1:
            ids = [item.get("id") for item in matches]
            raise RuntimeError(f"Expected one {TILE} item on {date}; found {len(matches)}: {ids}")
        selected.append(matches[0])
    return selected


def query_stac(buffered: gpd.GeoSeries) -> list[dict]:
    params = {
        "collections": COLLECTION,
        "bbox": ",".join(map(str, buffered.to_crs("EPSG:4326").total_bounds)),
        "datetime": f"{SELECTED_DATES[0]}T00:00:00Z/{SELECTED_DATES[-1]}T23:59:59Z",
        "limit": 100,
    }
    url = f"{STAC_URL}?{urlencode(params)}"
    visited, items = set(), {}
    while url:
        if url in visited:
            raise RuntimeError("Repeated STAC pagination link.")
        visited.add(url)
        with urlopen(Request(url, headers={"Accept": "application/geo+json"}), timeout=60) as response:
            page, base = json.load(response), response.geturl()
        if not isinstance(page, dict) or not isinstance(page.get("features"), list):
            raise RuntimeError("Invalid STAC response: missing features.")
        for item in page["features"]:
            items[(item.get("collection"), item["id"])] = item
        link = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
        url = None
        if link:
            if link.get("method", "GET").upper() != "GET" or not link.get("href"):
                raise RuntimeError("Unsupported STAC pagination; selection incomplete.")
            url = urljoin(base, link["href"])
    selected = select_scenes(list(items.values()))
    for date, item in zip(SELECTED_DATES, selected):
        print(f"Selected {date}: {item['id']}")
    return selected


def asset_s3_path(asset: dict) -> str:
    for location in [asset, *(asset.get("alternate") or {}).values()]:
        href = location.get("href", "")
        if href.startswith("/vsis3/"):
            href = "s3://" + href[len("/vsis3/"):]
        parsed = urlparse(href)
        if parsed.query or parsed.fragment:
            continue
        if parsed.scheme == "s3" and parsed.netloc == "eodata" and parsed.path.strip("/"):
            return "/vsis3/eodata" + parsed.path
        if (parsed.scheme == "https" and parsed.hostname == S3_ENDPOINT
                and parsed.path.startswith("/eodata/") and parsed.path[len("/eodata/"):]):
            return "/vsis3" + parsed.path
    raise RuntimeError("Asset has no supported S3 path in bucket eodata.")


@contextmanager
def open_remote(asset: dict):
    path = asset_s3_path(asset)
    print(f"Reading: {path}")
    try:
        with rasterio.open(path) as src:
            yield src
    except (OSError, ValueError, RuntimeError, rasterio.errors.RasterioError) as exc:
        raise RuntimeError(
            f"Remote processing failed; S3 path attempted: {path}. {exc}. "
            "No full-product download fallback; check S3 access and GDAL/JP2 support."
        ) from exc


def metadata_number(asset: dict, name: str, required: bool = True):
    """Accept both raster extension schemas, rejecting conflicting values."""
    sources = [asset]
    for field in ("raster:bands", "bands"):
        bands = asset.get(field)
        if bands is not None:
            if not isinstance(bands, list) or len(bands) != 1 or not isinstance(bands[0], dict):
                raise ValueError(f"Expected one band in STAC {field}.")
            sources.extend(bands)
    values = [float(source[key]) for source in sources
              for key in (f"raster:{name}", name)
              if source.get(key) is not None]
    if not values:
        if required:
            raise ValueError(f"Missing explicit STAC raster:{name}; no default assumed.")
        return None
    if any(not (value == values[0] or np.isnan(value) and np.isnan(values[0])) for value in values):
        raise ValueError(f"Conflicting STAC raster:{name}: {values}")
    if name != "nodata" and not np.isfinite(values[0]):
        raise ValueError(f"STAC raster:{name} must be finite.")
    return values[0]


def shapes_in_crs(buffered, crs) -> list:
    return [geom.__geo_interface__ for geom in buffered.to_crs(crs)]


def polygon_mask(buffered, grid: dict) -> np.ndarray:
    return geometry_mask(shapes_in_crs(buffered, grid["crs"]),
                         out_shape=(grid["height"], grid["width"]),
                         transform=grid["transform"], invert=True, all_touched=False)


def native_grid(src, buffered, resolution: int) -> tuple[dict, Window]:
    if (src.count != 1 or src.crs is None or not src.crs.is_projected
            or not np.isclose(src.crs.linear_units_factor[1], 1)
            or src.transform.b != 0 or src.transform.d != 0
            or not np.allclose((src.transform.a, src.transform.e), (resolution, -resolution))):
        raise ValueError(f"Expected a north-up single-band native {resolution} m raster.")
    window = geometry_window(src, shapes_in_crs(buffered, src.crs), boundless=True)
    grid = dict(crs=src.crs, transform=src.window_transform(window),
                width=int(window.width), height=int(window.height))
    return grid, window


def read_band(asset: dict, buffered, grid: dict | None):
    """Keep raw 10 m DNs on one shared native grid, without resampling."""
    with open_remote(asset) as src:
        candidate, window = native_grid(src, buffered, 10)
        if grid is not None and candidate != grid:
            raise ValueError("B04/B08 or scene grids differ; native 10 m alignment required.")
        data = src.read(1, window=window, boundless=True, masked=True)
    return data, candidate


def apply_scl_mask(asset: dict, buffered, grid: dict, inside: np.ndarray) -> np.ndarray:
    """Check local quality at 20 m, then transfer classes with nearest only."""
    with open_remote(asset) as src:
        scl_grid, window = native_grid(src, buffered, 20)
        scl = src.read(1, window=window, boundless=True, masked=True).filled(0)
    if not np.isin(scl, np.arange(12)).all():
        raise ValueError("Unexpected SCL class outside 0..11.")
    scl_inside = polygon_mask(buffered, scl_grid)
    if not scl_inside.any():
        raise ValueError("No SCL pixel centers inside buffer.")
    invalid = np.isin(scl, INVALID_SCL)
    percent = 100 * np.count_nonzero(invalid & scl_inside) / np.count_nonzero(scl_inside)
    print(f"Local Invalid % (native SCL): {percent:.6f}")
    if percent > MAX_LOCAL_INVALID_PERCENT:
        raise ValueError(f"Selected scene fails Local Invalid <= 1%: {percent:.6f}%.")
    aligned = np.zeros(inside.shape, dtype=np.uint8)
    reproject(source=scl, destination=aligned, src_transform=scl_grid["transform"],
              src_crs=scl_grid["crs"], dst_transform=grid["transform"],
              dst_crs=grid["crs"], src_nodata=0, dst_nodata=0,
              resampling=Resampling.nearest)
    return inside & ~np.isin(aligned, INVALID_SCL)


def dn_to_reflectance(data, scale: float, offset: float, nodata) -> np.ndarray:
    raw = np.asarray(data.data, dtype=np.float64)
    valid = ~np.ma.getmaskarray(data) & np.isfinite(raw)
    if nodata is not None:
        valid &= ~np.isnan(raw) if np.isnan(nodata) else raw != nodata
    # Mask raw nodata before the radiometric conversion.
    raw = np.where(valid, raw, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        reflectance = raw * scale + offset
    reflectance[~np.isfinite(reflectance)] = np.nan
    return reflectance


def calculate_ndvi(red: np.ndarray, nir: np.ndarray, scl_valid: np.ndarray):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = nir + red
        eligible = scl_valid & np.isfinite(red) & np.isfinite(nir)
        bad_denominator = eligible & ((denominator == 0) | ~np.isfinite(denominator))
        usable = eligible & ~bad_denominator
        ndvi = np.full(red.shape, np.nan, dtype=np.float64)
        np.divide(nir - red, denominator, out=ndvi, where=usable)
    outside = usable & np.isfinite(ndvi) & ((ndvi < -1) | (ndvi > 1))
    nonfinite = usable & ~np.isfinite(ndvi)
    ndvi[outside | nonfinite] = np.nan
    print(f"Invalid zero/nonfinite denominator: {np.count_nonzero(bad_denominator)}")
    print(f"NDVI outside [-1, 1] (discarded): {np.count_nonzero(outside)}")
    print(f"Nonfinite NDVI (discarded): {np.count_nonzero(nonfinite)}")
    return ndvi.astype(np.float32)


def print_statistics(label: str, values: np.ndarray) -> None:
    valid = values[np.isfinite(values)]
    if not valid.size:
        print(f"{label} min/max/mean/median: unavailable (no valid pixels)")
        return
    print(f"{label} min: {valid.min():.8f}; max: {valid.max():.8f}; "
          f"mean: {valid.mean(dtype=np.float64):.8f}; median: {np.median(valid):.8f}")


def process_scene(item: dict, buffered, grid: dict | None):
    print(f"\nDate: {item['properties']['datetime']}\nItem ID: {item['id']}")
    assets = item.get("assets", {})
    for key in ("B04_10m", "B08_10m", "SCL_20m"):
        if not isinstance(assets.get(key), dict):
            raise ValueError(f"Missing asset {key} on {item['id']}")
    reflectances = []
    for key in ("B04_10m", "B08_10m"):
        asset = assets[key]
        scale = metadata_number(asset, "scale")
        offset = metadata_number(asset, "offset")
        nodata = metadata_number(asset, "nodata", required=False)
        print(f"{key} STAC scale: {scale}; offset: {offset}; nodata: {nodata}")
        data, grid = read_band(asset, buffered, grid)
        reflectances.append(dn_to_reflectance(data, scale, offset, nodata))
    inside = polygon_mask(buffered, grid)
    if not inside.any():
        raise ValueError("No 10 m pixel centers inside buffer.")
    scl_valid = apply_scl_mask(assets["SCL_20m"], buffered, grid, inside)
    print(f"Pixels masked by SCL inside buffer: {np.count_nonzero(inside & ~scl_valid)}")
    ndvi = calculate_ndvi(*reflectances, scl_valid)
    print(f"Valid pixels: {np.count_nonzero(np.isfinite(ndvi))}")
    print_statistics("Scene NDVI", ndvi)
    return ndvi, grid, inside


def temporal_composite(stack: np.ndarray, inside: np.ndarray) -> np.ndarray:
    if stack.shape[0] != len(SELECTED_DATES):
        raise ValueError("The temporal stack must contain exactly seven scenes.")
    observations = np.count_nonzero(np.isfinite(stack), axis=0)
    counts = observations[inside]
    print("\nBefore reduction to 30 m (pixel centers inside buffer):")
    print("Valid observations per pixel: represented by the following count distribution.")
    print_statistics("Valid observation count", counts)
    for count in range(8):
        print(f"Pixels with {count} valid observations: {np.count_nonzero(counts == count)}")
    composite = np.full(inside.shape, np.nan, dtype=np.float32)
    observed = inside & (observations > 0)
    # Avoid all-NaN slices while preserving missing pixels as NaN.
    composite[observed] = np.nanmedian(stack[:, observed], axis=0)
    print_statistics("10 m median composite NDVI", composite)
    if not observed.any():
        raise ValueError("Composite has no valid observations.")
    return composite


def grid_matches_dem(dataset, dem) -> bool:
    return (dataset.crs == dem.crs and dataset.transform == dem.transform
            and dataset.width == dem.width and dataset.height == dem.height
            and dataset.bounds == dem.bounds and dataset.res == dem.res)


def harmonize_and_write(composite: np.ndarray, grid: dict, buffered) -> None:
    with rasterio.open(DEM_FILE) as dem:
        if dem.crs is None or dem.count != 1:
            raise ValueError("Reference DEM must have a CRS and one band.")
        result = np.full((dem.height, dem.width), OUTPUT_NODATA, dtype=np.float32)
        reproject(source=composite, destination=result,
                  src_transform=grid["transform"], src_crs=grid["crs"], src_nodata=np.nan,
                  dst_transform=dem.transform, dst_crs=dem.crs, dst_nodata=OUTPUT_NODATA,
                  resampling=Resampling.average)
        elevation = dem.read(1, masked=True)
        dem_invalid = np.ma.getmaskarray(elevation) | ~np.isfinite(elevation.data)
        final_grid = dict(crs=dem.crs, transform=dem.transform, width=dem.width, height=dem.height)
        outside_buffer = ~polygon_mask(buffered, final_grid)
        physical_invalid = ((result != OUTPUT_NODATA) & np.isfinite(result)
                            & ((result < -1) | (result > 1)))
        print(f"\n30 m NDVI outside [-1, 1] (discarded): {np.count_nonzero(physical_invalid)}")
        result[dem_invalid | outside_buffer | physical_invalid | ~np.isfinite(result)] = OUTPUT_NODATA
        if not np.any(result != OUTPUT_NODATA):
            raise ValueError("No valid NDVI remains on the reference DEM grid.")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        profile = dict(driver="GTiff", count=1, dtype="float32", nodata=OUTPUT_NODATA,
                       compress="deflate", **final_grid)
        with rasterio.open(OUTPUT_FILE, "w", **profile) as dst:
            if not grid_matches_dem(dst, dem):
                raise RuntimeError("Grid matches DEM: False")
            dst.write(result, 1)
        with rasterio.open(OUTPUT_FILE) as final:
            matches = grid_matches_dem(final, dem)
            print(f"Final CRS: {final.crs}\nResolution: {final.res}")
            print(f"Dimensions (width, height): {final.width}, {final.height}")
            print(f"Bounds: {final.bounds}\nNoData: {final.nodata}")
            print(f"Grid matches DEM: {matches}")
            if not matches:
                raise RuntimeError("Final NDVI grid does not exactly match the DEM.")
            saved = final.read(1)
            if not np.array_equal(saved, result):
                raise RuntimeError("Saved NDVI values differ from the validated result.")
            valid = saved != OUTPUT_NODATA
            print(f"Valid pixels: {np.count_nonzero(valid)}")
            print(f"NoData pixels: {np.count_nonzero(~valid)}")
            print_statistics("Final NDVI", saved[valid])
    print(f"Output: {OUTPUT_FILE}")


def main() -> None:
    credentials = credentials_from_environment()
    if not DEM_FILE.is_file():
        raise FileNotFoundError(f"Reference DEM not found: {DEM_FILE}")
    buffered = load_buffer()
    items = query_stac(buffered)
    scenes, grid = [], None
    # DummySession prevents discovery of credentials from profiles or boto3.
    with rasterio.Env(
        session=DummySession(), AWS_S3_ENDPOINT=S3_ENDPOINT, AWS_REGION="default",
        AWS_VIRTUAL_HOSTING=False, AWS_HTTPS=True, AWS_NO_SIGN_REQUEST=False,
        AWS_SESSION_TOKEN="", AWS_EC2_METADATA_DISABLED=True,
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="60",
        GDAL_HTTP_MAX_RETRY="0", CPL_VSIL_CURL_CHUNK_SIZE="16384",
    ):
        rasterio.env.setenv(**credentials)
        for item in items:
            ndvi, grid, inside = process_scene(item, buffered, grid)
            scenes.append(ndvi)
    stack = np.stack(scenes, axis=0)
    del scenes
    composite = temporal_composite(stack, inside)
    harmonize_and_write(composite, grid, buffered)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, rasterio.errors.RasterioError) as exc:
        raise SystemExit(f"Pre-event NDVI processing failed: {exc}") from exc
