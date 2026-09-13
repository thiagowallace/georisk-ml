"""Inspect local Sentinel-2 SCL quality without downloading full products."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, geometry_window


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_AREA_FILE = (
    PROJECT_ROOT / "data" / "interim" / "study_area" / "santa_tereza.gpkg"
)
STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
COLLECTION = "sentinel-2-l2a"
DATE_INTERVAL = "2024-03-01T00:00:00Z/2024-04-21T23:59:59Z"
S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
S3_BUCKET = "eodata"
BUFFER_METERS = 1000
INVALID_CODES = (0, 1, 3, 8, 9, 10, 11)
SCL_LABELS = {
    0: "No Data",
    1: "Saturated/Defective",
    2: "Dark areas/Topographic shadows (not automatically cloud)",
    3: "Cloud Shadows",
    4: "Vegetation",
    5: "Not vegetated",
    6: "Water",
    7: "Unclassified/Low probability (not automatically cloud)",
    8: "Cloud Medium Probability",
    9: "Cloud High Probability",
    10: "Thin Cirrus",
    11: "Snow/Ice",
}


def get_buffer() -> gpd.GeoSeries:
    """Create the geometry in the required projected CRS."""

    area = gpd.read_file(STUDY_AREA_FILE)
    if area.crs is None or area.crs.to_epsg() != 31982:
        raise ValueError("Study area must be in EPSG:31982.")
    if (
        area.empty
        or area.geometry.isna().any()
        or area.geometry.is_empty.any()
        or not area.geometry.is_valid.all()
        or not area.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    ):
        raise ValueError("Study area must contain valid, nonempty polygons.")
    return gpd.GeoSeries(
        [area.geometry.buffer(BUFFER_METERS).union_all()], crs=area.crs
    )


def fetch_items(bbox: list[float]) -> list[dict]:
    """Query every catalog page without a cloud-cover filter."""

    params = {
        "collections": COLLECTION,
        "bbox": ",".join(str(value) for value in bbox),
        "datetime": DATE_INTERVAL,
        "limit": 100,
    }
    url = f"{STAC_URL}?{urlencode(params)}"
    visited = set()
    items = {}
    while url:
        if url in visited:
            raise RuntimeError("Repeated STAC pagination URL; results incomplete.")
        visited.add(url)
        try:
            request = Request(url, headers={"Accept": "application/geo+json"})
            with urlopen(request, timeout=60) as response:
                page = json.load(response)
                base_url = response.geturl()
        except HTTPError as exc:
            mechanism = exc.headers.get("WWW-Authenticate", "not advertised")
            raise RuntimeError(
                f"STAC HTTP {exc.code}: {exc.reason}. "
                f"Authentication challenge: {mechanism}"
            ) from exc
        except (URLError, OSError, ValueError) as exc:
            raise RuntimeError(f"STAC request failed: {exc}") from exc
        if not isinstance(page, dict) or not isinstance(page.get("features"), list):
            raise RuntimeError("Invalid STAC response: missing features list.")
        for item in page["features"]:
            items[(item.get("collection"), item["id"])] = item
        print(f"Catalog page {len(visited)}: {len(page['features'])} items")
        link = next(
            (link for link in page.get("links", []) if link.get("rel") == "next"),
            None,
        )
        url = None
        if link:
            if link.get("method", "GET").upper() != "GET" or not link.get("href"):
                raise RuntimeError("Unsupported STAC next link; results incomplete.")
            url = urljoin(base_url, link["href"])
    return list(items.values())


def item_date(item: dict) -> str:
    properties = item.get("properties") or {}
    return properties.get("datetime") or properties.get("start_datetime") or "N/A"


def date_key(item: dict) -> tuple:
    date = item_date(item)
    parsed = datetime.max.replace(tzinfo=timezone.utc)
    if date != "N/A":
        parsed = datetime.fromisoformat(date.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed, item["id"]


def print_assets(item: dict) -> None:
    """Show only the relevant band keys, without asset metadata."""

    relevant = {"scl_20m", "b04_10m", "b08_10m"}
    keys = [key for key in item.get("assets", {}) if key.lower() in relevant]
    print(f"Relevant asset keys: {', '.join(keys) or 'none'}")


def find_scl(item: dict) -> tuple[str, dict]:
    """Require the native 20 m classification asset."""

    candidates = [
        (key, asset) for key, asset in item.get("assets", {}).items()
        if key.lower() == "scl_20m"
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError("Missing or ambiguous SCL_20m asset.")


def s3_href(asset: dict) -> str:
    """Extract a bucket/key from an advertised S3 or endpoint location."""

    locations = [asset, *(asset.get("alternate") or {}).values()]
    for location in locations:
        href = location.get("href", "")
        if href.startswith("/vsis3/"):
            href = "s3://" + href[len("/vsis3/"):]
        parsed = urlparse(href)
        if parsed.query or parsed.fragment:
            continue
        if parsed.scheme == "s3" and parsed.netloc == S3_BUCKET:
            if parsed.path.strip("/"):
                return href
        if parsed.scheme == "https" and parsed.hostname == S3_ENDPOINT:
            prefix = f"/{S3_BUCKET}/"
            if parsed.path.startswith(prefix) and parsed.path[len(prefix):]:
                return f"s3://{S3_BUCKET}/{parsed.path[len(prefix):]}"
    raise RuntimeError("SCL_20m has no supported S3 location in bucket eodata.")


def s3_credentials() -> dict[str, str]:
    """Use only the two explicitly authorized environment variables."""

    names = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError("Missing S3 credentials in environment: " + ", ".join(missing))
    return {name: os.environ[name] for name in names}


def read_local_scl(src, buffered: gpd.GeoSeries) -> np.ndarray:
    """Read a small native window, keeping SCL 0 in the polygon denominator."""

    if src.crs is None:
        raise ValueError("SCL raster has no CRS.")
    geometry = buffered.to_crs(src.crs)
    shapes = [geom.__geo_interface__ for geom in geometry]
    # Include the full buffer even if it crosses a tile edge. Uncovered pixels
    # receive SCL 0 instead of silently shrinking the local denominator.
    window = geometry_window(src, shapes, boundless=True)
    data = src.read(1, window=window, boundless=True, masked=True)
    inside = geometry_mask(
        shapes,
        out_shape=data.shape,
        transform=src.window_transform(window),
        invert=True,
        all_touched=False,
    )
    values = data.filled(0)[inside]
    if values.size == 0:
        raise ValueError("No SCL pixel centers inside the buffer.")
    if not np.isin(values, list(SCL_LABELS)).all():
        raise ValueError(f"Unexpected SCL codes: {np.unique(values).tolist()}")
    return values


def inspect_scene(
    item: dict, buffered: gpd.GeoSeries, credentials: dict[str, str]
) -> dict:
    """Calculate class fractions only after a successful remote window read."""

    href = "unavailable (asset location not resolved)"
    print_assets(item)
    try:
        key, asset = find_scl(item)
        href = s3_href(asset)
        print(f"Reading SCL asset: {key}\nHref: {href}")
        # Use range-based access, never streaming or a full-file download fallback.
        with rasterio.Env(
            AWS_S3_ENDPOINT=S3_ENDPOINT,
            AWS_REGION="default",
            AWS_VIRTUAL_HOSTING=False,
            AWS_HTTPS=True,
            AWS_NO_SIGN_REQUEST=False,
            AWS_SESSION_TOKEN="",
            AWS_EC2_METADATA_DISABLED=True,
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_TIMEOUT="60",
            GDAL_HTTP_MAX_RETRY="0",
            CPL_VSIL_CURL_CHUNK_SIZE="16384",
        ):
            # Set explicit GDAL credentials without a boto3/profile session.
            rasterio.env.setenv(**credentials)
            with rasterio.open("/vsis3/" + href[len("s3://"):]) as src:
                values = read_local_scl(src, buffered)
    except (RuntimeError, OSError, ValueError, rasterio.errors.RasterioError) as exc:
        raise RuntimeError(
            f"Stopped on item {item['id']}: {exc}. S3 path attempted: {href}. "
            "No full-file download fallback. Local results are incomplete. "
            "Check S3 credentials/permissions and rasterio/GDAL S3 and JP2 support "
            "before choosing another access strategy."
        ) from exc

    counts = np.bincount(values.astype(np.int64), minlength=12)
    invalid = 100.0 * np.count_nonzero(np.isin(values, INVALID_CODES)) / values.size
    cloud = (item.get("properties") or {}).get("eo:cloud_cover")
    print(f"Date: {item_date(item)}\nItem ID: {item['id']}")
    print(f"Catalog cloud cover: {cloud if cloud is not None else 'N/A'}")
    print(f"Total SCL pixels inside buffer: {values.size}")
    print(f"{'SCL':>4} {'Pixels':>12} {'Percent':>10} Class")
    for code, count in enumerate(counts):
        print(f"{code:>4} {count:>12} {100.0 * count / values.size:>9.4f}% "
              f"{SCL_LABELS[int(code)]}")
    print(f"Local invalid for NDVI: {invalid:.4f}%")
    print(f"Local valid: {100.0 - invalid:.4f}%")
    return {"item": item, "cloud": cloud, "invalid": invalid}


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"{'Date':<28} {'Catalog Cloud %':>16} {'Local Invalid %':>16} "
          f"{'Local Valid %':>14} Item ID")
    for row in rows:
        cloud = str(row["cloud"]) if row["cloud"] is not None else "N/A"
        print(f"{item_date(row['item']):<28} {cloud:>16} {row['invalid']:>16.4f} "
              f"{100.0 - row['invalid']:>14.4f} {row['item']['id']}")


def main() -> None:
    """Inspect all candidate scenes, stopping if asset access is unavailable."""

    print("GeoRisk ML - Inspect Sentinel-2 Local SCL Quality")
    credentials = s3_credentials()
    buffered = get_buffer()
    bbox = buffered.to_crs("EPSG:4326").total_bounds.tolist()
    print(f"Endpoint: {STAC_URL}\nCollection: {COLLECTION}")
    print(f"Interval: {DATE_INTERVAL}\nBuffer: {BUFFER_METERS} m\nBBox: {bbox}")
    items = sorted(fetch_items(bbox), key=date_key)
    print(f"Total candidate items: {len(items)}")
    if not items:
        print("No scenes found in the requested interval.")
        return
    print(f"Local invalid SCL codes: {INVALID_CODES}")
    print("SCL 2 and 7 are reported separately and retained by this initial rule.")
    print("Percentages include all pixel centers inside the buffer, including SCL 0.")
    rows = [inspect_scene(item, buffered, credentials) for item in items]
    print_table(rows, "Scenes ordered by date")
    print_table(
        sorted(rows, key=lambda row: (row["invalid"], date_key(row["item"]))),
        "Scenes ordered by local invalid percentage (ascending)",
    )
    print("Local valid means retained by the initial SCL rule, not guaranteed NDVI quality.")
    print("Local SCL inspection completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(f"Local SCL inspection failed: {exc}") from exc
