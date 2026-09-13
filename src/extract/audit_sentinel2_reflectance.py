"""Audit STAC radiometry and a small raw-DN sample of two L2A bands."""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import Polygon

if __package__:
    from .inspect_sentinel2_local_quality import (
        COLLECTION, S3_ENDPOINT, STAC_URL, get_buffer, s3_credentials, s3_href,
    )
else:
    from inspect_sentinel2_local_quality import (
        COLLECTION, S3_ENDPOINT, STAC_URL, get_buffer, s3_credentials, s3_href,
    )

ITEM_ID = "S2A_MSIL2A_20240302T133151_N0510_R081_T22JDN_20240302T180059"
ASSET_KEYS = ("B04_10m", "B08_10m")
SAMPLE_SIZE = 64


def fetch_item() -> dict:
    """Search by exact ID and reject an absent or ambiguous response."""
    params = urlencode({"collections": COLLECTION, "ids": ITEM_ID, "limit": 2})
    request = Request(
        f"{STAC_URL}?{params}", headers={"Accept": "application/geo+json"}
    )
    with urlopen(request, timeout=60) as response:
        page = json.load(response)
    if not isinstance(page, dict) or not isinstance(page.get("features"), list):
        raise RuntimeError("Invalid STAC response: missing features list.")
    matches = [
        item for item in page["features"]
        if item.get("id") == ITEM_ID and item.get("collection") == COLLECTION
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one matching STAC item; found {len(matches)}.")
    return matches[0]


def show(label: str, value) -> None:
    print(f"{label}: {json.dumps(value, ensure_ascii=False) if value is not None else 'not exposed'}")


def print_metadata(item: dict, key: str, asset: dict, href: str) -> None:
    """Report selected fields with their source, including both band schemas."""
    print(f"\nAsset key: {key}\nS3 href: {href}")
    sources = [("asset", asset)]
    for field in ("raster:bands", "bands", "eo:bands"):
        bands = asset.get(field)
        show(f"{field} metadata", bands)
        if isinstance(bands, list):
            sources.extend(
                (f"{field}[{index}]", band)
                for index, band in enumerate(bands) if isinstance(band, dict)
            )
    for label, aliases in (
        ("data_type", ("data_type", "raster:data_type")),
        ("nodata", ("nodata", "raster:nodata")),
        ("raster:scale", ("raster:scale", "scale")),
        ("raster:offset", ("raster:offset", "offset")),
        ("statistics", ("statistics", "raster:statistics")),
    ):
        found = False
        for source, metadata in sources:
            for alias in aliases:
                if alias in metadata and metadata[alias] is not None:
                    prefix = "*** STAC RADIOMETRY *** " if label in ("raster:scale", "raster:offset") else ""
                    show(f"{prefix}{label} ({source}.{alias})", metadata[alias])
                    found = True
        if not found:
            show(label, None)
    properties = item.get("properties") or {}
    for field in ("gsd", "proj:code", "proj:shape", "proj:transform"):
        source = "asset" if asset.get(field) is not None else "item.properties"
        show(f"{field} ({source})", asset.get(field) if source == "asset" else properties.get(field))


def report_scale_offset(src) -> None:
    """Do not interpret GDAL's identity defaults as calibration metadata."""
    scales, offsets = src.scales, src.offsets
    scale = scales[0] if scales else None
    offset = offsets[0] if offsets else None
    show("Dataset rasterio scale (reported)", scale)
    show("Dataset rasterio offset (reported)", offset)
    if scale is None or offset is None:
        print("Dataset does not expose a complete scale/offset pair; no values assumed.")
    elif scale == 1.0 and offset == 0.0:
        print(
            "Rasterio reports the identity pair (1, 0), which may be GDAL defaults. "
            "This API does not distinguish absent metadata from explicit identity values. "
            "Radiometric scale/offset are therefore not confirmed; no calibration assumed."
        )
    else:
        print("Dataset exposes a non-identity scale/offset pair; reported only, not applied.")


def sample_dn(src, buffered) -> None:
    """Read at most 64 x 64 native pixels and mask to the actual buffer."""
    if src.crs is None or src.count != 1:
        raise ValueError("Expected a single-band raster with a CRS.")
    geometry = buffered.to_crs(src.crs).iloc[0]
    footprint = Polygon([
        src.transform * corner for corner in
        ((0, 0), (src.width, 0), (src.width, src.height), (0, src.height))
    ])
    overlap = geometry.intersection(footprint)
    if overlap.is_empty or overlap.area <= 0:
        raise ValueError("Band does not overlap Santa Tereza + 1000 m buffer.")
    point = overlap.representative_point()
    row, col = src.index(point.x, point.y)
    width, height = min(SAMPLE_SIZE, src.width), min(SAMPLE_SIZE, src.height)
    col_start = max(0, min(col - width // 2, src.width - width))
    row_start = max(0, min(row - height // 2, src.height - height))
    window = Window(col_start, row_start, width, height)
    data = src.read(1, window=window, masked=True)
    inside = geometry_mask(
        [geometry.__geo_interface__], out_shape=data.shape,
        transform=src.window_transform(window), invert=True, all_touched=False,
    )
    valid = inside & ~np.ma.getmaskarray(data) & np.isfinite(data.data)
    values = data.data[valid]
    print(f"Sample window: {window}; CRS: {src.crs}")
    show("dtype", src.dtypes[0])
    show("Dataset nodata", src.nodata)
    report_scale_offset(src)
    print(f"Pixel centers inside buffer: {np.count_nonzero(inside)}")
    print(f"DN sample pixels (excluding nodata/masked/nonfinite): {values.size}")
    if values.size:
        show("Minimum DN", values.min().item())
        show("Maximum DN", values.max().item())
        show("Mean DN", float(values.mean()))
    else:
        print("Minimum DN / Maximum DN / Mean DN: unavailable; no valid sample pixels.")
    print("Statistics describe this small raw-DN sample only; scale/offset were not applied.")


def main() -> None:
    credentials = s3_credentials()
    item = fetch_item()
    print(f"Item ID: {item['id']}")
    selected = []
    for key in ASSET_KEYS:
        asset = (item.get("assets") or {}).get(key)
        if not isinstance(asset, dict):
            raise RuntimeError(f"Required asset missing: {key}")
        try:
            href = s3_href(asset)
        except RuntimeError:
            raise RuntimeError(f"{key}: no supported S3 location in bucket eodata.") from None
        print_metadata(item, key, asset, href)
        selected.append((key, href))
    buffered = get_buffer()
    for key, href in selected:
        print(f"\nRemote DN audit: {key}\nS3 path: {href}")
        try:
            with rasterio.Env(
                AWS_S3_ENDPOINT=S3_ENDPOINT, AWS_REGION="default",
                AWS_VIRTUAL_HOSTING=False, AWS_HTTPS=True,
                AWS_NO_SIGN_REQUEST=False, AWS_SESSION_TOKEN="",
                AWS_EC2_METADATA_DISABLED=True,
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="0",
                CPL_VSIL_CURL_CHUNK_SIZE="16384",
            ):
                rasterio.env.setenv(**credentials)
                with rasterio.open("/vsis3/" + href[len("s3://"):]) as src:
                    sample_dn(src, buffered)
        except (RuntimeError, OSError, ValueError, rasterio.errors.RasterioError) as exc:
            raise RuntimeError(
                f"Remote audit failed for {key}; S3 path attempted: {href}. {exc}. "
                "Check authentication and GDAL S3/JP2 support. "
                "Audit stopped without a full-product download fallback."
            ) from exc
    print("Reflectance metadata audit completed.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(f"Reflectance audit failed: {exc}") from exc
