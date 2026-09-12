"""Download Copernicus DEM GLO-30 tiles required by the study area."""

from __future__ import annotations

import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STUDY_AREA_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "study_area"
    / "santa_tereza.gpkg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "copernicus_dem"
    / "glo30"
)

BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def format_latitude(latitude: int) -> str:
    """Convert integer latitude to Copernicus tile notation."""

    hemisphere = "N" if latitude >= 0 else "S"

    return f"{hemisphere}{abs(latitude):02d}_00"


def format_longitude(longitude: int) -> str:
    """Convert integer longitude to Copernicus tile notation."""

    hemisphere = "E" if longitude >= 0 else "W"

    return f"{hemisphere}{abs(longitude):03d}_00"


def get_required_tiles(
    study_area: gpd.GeoDataFrame,
) -> list[tuple[int, int]]:
    """Determine all 1-degree DEM tiles intersecting the study area."""

    geographic = study_area.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = geographic.total_bounds

    print("--- STUDY AREA BOUNDS (EPSG:4326) ---")
    print(f"Min longitude: {minx:.6f}")
    print(f"Min latitude:  {miny:.6f}")
    print(f"Max longitude: {maxx:.6f}")
    print(f"Max latitude:  {maxy:.6f}")
    print()

    min_lon = math.floor(minx)
    max_lon = math.floor(maxx)

    min_lat = math.floor(miny)
    max_lat = math.floor(maxy)

    tiles: list[tuple[int, int]] = []

    for latitude in range(min_lat, max_lat + 1):
        for longitude in range(min_lon, max_lon + 1):
            tiles.append((latitude, longitude))

    return tiles


def build_tile_name(latitude: int, longitude: int) -> str:
    """Build the official Copernicus DEM GLO-30 tile name."""

    lat_code = format_latitude(latitude)
    lon_code = format_longitude(longitude)

    return (
        f"Copernicus_DSM_COG_10_"
        f"{lat_code}_{lon_code}_DEM"
    )


def download_tile(latitude: int, longitude: int) -> Path:
    """Download one DEM tile if it is not already available."""

    tile_name = build_tile_name(latitude, longitude)

    filename = f"{tile_name}.tif"

    output_file = OUTPUT_DIR / filename

    url = (
        f"{BASE_URL}/"
        f"{tile_name}/"
        f"{filename}"
    )

    if output_file.exists():
        print(f"[OK] Already exists: {filename}")
        return output_file

    print(f"Downloading: {filename}")
    print(f"URL: {url}")

    try:
        urlretrieve(url, output_file)

    except HTTPError as exc:
        if output_file.exists():
            output_file.unlink()

        raise RuntimeError(
            f"HTTP error downloading {filename}: {exc.code}"
        ) from exc

    except URLError as exc:
        if output_file.exists():
            output_file.unlink()

        raise RuntimeError(
            f"Network error downloading {filename}: {exc.reason}"
        ) from exc

    print(f"[OK] Download completed: {filename}")

    return output_file


def main() -> None:
    """Download all GLO-30 tiles required by the study area."""

    print("GeoRisk ML - Copernicus DEM GLO-30 Download")
    print()

    if not STUDY_AREA_FILE.exists():
        raise FileNotFoundError(
            f"Study area not found: {STUDY_AREA_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    study_area = gpd.read_file(STUDY_AREA_FILE)

    tiles = get_required_tiles(study_area)

    print("--- REQUIRED TILES ---")

    for latitude, longitude in tiles:
        print(build_tile_name(latitude, longitude))

    print()
    print(f"Total tiles required: {len(tiles)}")
    print()

    downloaded_files = []

    for latitude, longitude in tiles:
        downloaded_files.append(
            download_tile(latitude, longitude)
        )

    print()
    print("--- RESULT ---")
    print(f"Downloaded/available tiles: {len(downloaded_files)}")
    print(f"Destination: {OUTPUT_DIR}")
    print()
    print("DEM acquisition completed successfully.")


if __name__ == "__main__":
    main()