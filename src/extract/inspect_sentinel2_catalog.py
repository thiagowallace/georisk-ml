"""Inspect pre-event Sentinel-2 L2A metadata in the Copernicus STAC catalog."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_AREA_FILE = (
    PROJECT_ROOT / "data" / "interim" / "study_area" / "santa_tereza.gpkg"
)
STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
COLLECTION = "sentinel-2-l2a"
DATE_INTERVAL = "2024-01-01T00:00:00Z/2024-04-26T23:59:59Z"
BUFFER_METERS = 1000


def get_bbox() -> list[float]:
    """Buffer in metres before transforming the geometry to geographic CRS."""

    if not STUDY_AREA_FILE.is_file():
        raise FileNotFoundError(f"Study area not found: {STUDY_AREA_FILE}")

    study_area = gpd.read_file(STUDY_AREA_FILE)
    if study_area.crs is None or study_area.crs.to_epsg() != 31982:
        raise ValueError("Study area must be in EPSG:31982.")
    if (
        study_area.empty
        or study_area.geometry.isna().any()
        or study_area.geometry.is_empty.any()
        or not study_area.geometry.is_valid.all()
        or not study_area.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    ):
        raise ValueError("Study area must contain valid, nonempty polygons.")

    buffered = study_area.geometry.buffer(BUFFER_METERS)
    return buffered.to_crs("EPSG:4326").total_bounds.tolist()


def fetch_items(bbox: list[float]) -> list[dict]:
    """Read every metadata page without requesting product assets."""

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
            raise RuntimeError("STAC pagination repeated a URL; results are incomplete.")
        visited.add(url)
        request = Request(url, headers={"Accept": "application/geo+json"}, method="GET")
        try:
            with urlopen(request, timeout=60) as response:
                page = json.load(response)
                response_url = response.geturl()
        except HTTPError as exc:
            raise RuntimeError(f"STAC HTTP error {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"STAC network error: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise RuntimeError(f"STAC connection error: {exc}") from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("STAC returned invalid JSON.") from exc

        if not isinstance(page, dict) or not isinstance(page.get("features"), list):
            raise RuntimeError("STAC response does not contain an item list.")
        for item in page["features"]:
            if not isinstance(item, dict) or not item.get("id"):
                raise RuntimeError("STAC returned an invalid item without an ID.")
            items[(item.get("collection"), item["id"])] = item

        print(f"Page {len(visited)}: {len(page['features'])} items")
        next_link = next(
            (link for link in page.get("links", []) if link.get("rel") == "next"),
            None,
        )
        url = None
        if next_link is not None:
            if next_link.get("method", "GET").upper() != "GET":
                raise RuntimeError("STAC next link requires a method other than GET.")
            if not next_link.get("href"):
                raise RuntimeError("STAC next link has no URL.")
            url = urljoin(response_url, next_link["href"])

    return list(items.values())


def extract_item(item: dict) -> dict:
    """Normalize optional catalog fields without filtering scenes."""

    properties = item.get("properties") or {}

    def first(*keys):
        return next(
            (properties[key] for key in keys if properties.get(key) is not None),
            None,
        )

    date = first("datetime", "start_datetime")
    sort_date = datetime.max.replace(tzinfo=timezone.utc)
    if date:
        try:
            sort_date = datetime.fromisoformat(date.replace("Z", "+00:00"))
            if sort_date.tzinfo is None:
                sort_date = sort_date.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass

    cloud = first("eo:cloud_cover")
    try:
        cloud = float(cloud) if cloud is not None else None
    except (ValueError, TypeError):
        cloud = None
    if cloud is not None and not math.isfinite(cloud):
        cloud = None

    tile = first("s2:mgrs_tile", "mgrs:tile", "grid:code", "tileId", "tile_id")
    if tile is None:
        parts = [first(key) for key in (
            "mgrs:utm_zone", "mgrs:latitude_band", "mgrs:grid_square"
        )]
        if all(part is not None for part in parts):
            tile = "".join(str(part) for part in parts)
        else:
            match = re.search(r"_T(\d{2}[A-Z]{3})(?:_|$)", item["id"])
            tile = match.group(1) if match else None

    return {
        "date": date or "N/A",
        "sort_date": sort_date,
        "id": item["id"],
        "cloud": cloud,
        "platform": first("platform") or "N/A",
        "constellation": first("constellation") or "N/A",
        "product": first(
            "product:type", "s2:product_type", "productType", "product_type",
            "processing:level", "s2:processing_level"
        ) or "N/A",
        "tile": tile or "N/A",
    }


def main() -> None:
    """Print all scenes and cloud statistics for the requested interval."""

    print("GeoRisk ML - Inspect Sentinel-2 L2A Catalog")
    bbox = get_bbox()
    print(f"Endpoint: {STAC_URL}")
    print(f"Collection: {COLLECTION}")
    print(f"Interval: {DATE_INTERVAL}")
    print(f"Buffer: {BUFFER_METERS} m")
    print(f"BBox (EPSG:4326): {bbox}")
    rows = sorted(
        (extract_item(item) for item in fetch_items(bbox)),
        key=lambda row: (row["sort_date"], row["id"]),
    )

    print()
    print(f"{'Date':<28} {'Cloud %':>8} {'Platform':<14} {'Tile':<16} "
          f"{'Constellation':<14} {'Product':<12} Item ID")
    for row in rows:
        cloud = f"{row['cloud']:.2f}" if row["cloud"] is not None else "N/A"
        print(f"{row['date']:<28} {cloud:>8} {row['platform']:<14} "
              f"{row['tile']:<16} {row['constellation']:<14} "
              f"{row['product']:<12} {row['id']}")

    clouds = [row["cloud"] for row in rows if row["cloud"] is not None]
    print()
    print(f"Total items found: {len(rows)}")
    print(f"Items with unavailable cloud cover: {len(rows) - len(clouds)}")
    if clouds:
        print(f"Minimum cloud cover: {min(clouds):.2f}%")
        print(f"Maximum cloud cover: {max(clouds):.2f}%")
        print(f"Mean cloud cover: {sum(clouds) / len(clouds):.2f}%")
    else:
        print("Minimum cloud cover: N/A")
        print("Maximum cloud cover: N/A")
        print("Mean cloud cover: N/A")
    for threshold in (10, 20, 30):
        print(f"Scenes with cloud cover <= {threshold}%: "
              f"{sum(cloud <= threshold for cloud in clouds)}")
    print("Cloud cover is catalog scene metadata, not a measurement over the buffer.")
    print("Catalog inspection completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(f"Catalog inspection failed: {exc}") from exc
