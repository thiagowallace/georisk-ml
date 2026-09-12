"""Prepare Copernicus DEM GLO-30 for the GeoRisk ML study area."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEM_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "copernicus_dem"
    / "glo30"
)

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
    / "interim"
    / "dem"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "santa_tereza_dem_30m.tif"
)

TARGET_CRS = "EPSG:31982"
TARGET_RESOLUTION = 30.0

# Buffer used to reduce edge effects when terrain
# derivatives are calculated later.
BUFFER_METERS = 1000.0

OUTPUT_NODATA = -9999.0


def find_dem_file() -> Path:
    """Find the downloaded Copernicus DEM tile."""

    files = sorted(DEM_DIR.glob("*.tif"))

    if not files:
        raise FileNotFoundError(
            f"No DEM files found in: {DEM_DIR}"
        )

    if len(files) != 1:
        raise ValueError(
            f"Expected 1 DEM tile, found {len(files)}."
        )

    return files[0]


def main() -> None:
    """Clip and reproject the DEM for the pilot study area."""

    print("GeoRisk ML - Prepare Copernicus DEM")
    print()

    dem_file = find_dem_file()

    study_area = gpd.read_file(STUDY_AREA_FILE)

    print(f"DEM: {dem_file.name}")
    print(f"Study area CRS: {study_area.crs}")

    if study_area.crs is None:
        raise ValueError("Study area has no CRS.")

    # Generate the buffer in a projected CRS because
    # BUFFER_METERS is expressed in metres.
    study_area_projected = study_area.to_crs(
        TARGET_CRS
    )

    buffered_geometry = (
        study_area_projected
        .geometry
        .buffer(BUFFER_METERS)
        .union_all()
    )

    buffered = gpd.GeoDataFrame(
        geometry=[buffered_geometry],
        crs=TARGET_CRS,
    )

    with rasterio.open(dem_file) as src:

        print(f"Source CRS: {src.crs}")
        print(
            "Source resolution: "
            f"{src.res[0]} x {src.res[1]}"
        )
        print(f"Source nodata: {src.nodata}")

        # Reproject only the study-area geometry to the
        # native CRS of the DEM before clipping.
        buffered_source_crs = buffered.to_crs(
            src.crs
        )

        geometries = [
            geom.__geo_interface__
            for geom in buffered_source_crs.geometry
        ]

        # Keep the mask explicitly so that areas outside
        # the geometry can receive a controlled NoData value.
        clipped_data, clipped_transform = mask(
            src,
            geometries,
            crop=True,
            filled=False,
        )

        clipped_height = clipped_data.shape[1]
        clipped_width = clipped_data.shape[2]

        source_array = (
            clipped_data[0]
            .filled(OUTPUT_NODATA)
            .astype(np.float32)
        )

        # IMPORTANT:
        # These bounds are still in the SOURCE CRS
        # (EPSG:4326) and must be passed as such to
        # calculate_default_transform.
        left, bottom, right, top = array_bounds(
            clipped_height,
            clipped_width,
            clipped_transform,
        )

        dst_transform, dst_width, dst_height = (
            calculate_default_transform(
                src.crs,
                TARGET_CRS,
                clipped_width,
                clipped_height,
                left,
                bottom,
                right,
                top,
                resolution=TARGET_RESOLUTION,
            )
        )

        destination = np.full(
            (dst_height, dst_width),
            OUTPUT_NODATA,
            dtype=np.float32,
        )

        reproject(
            source=source_array,
            destination=destination,
            src_transform=clipped_transform,
            src_crs=src.crs,
            src_nodata=OUTPUT_NODATA,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=OUTPUT_NODATA,
            resampling=Resampling.bilinear,
        )

        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        count=1,
        dtype="float32",
        crs=TARGET_CRS,
        transform=dst_transform,
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile,
    ) as dst:
        dst.write(destination, 1)

    valid = destination[
        destination != OUTPUT_NODATA
    ]

    print()
    print("--- RESULT ---")
    print(f"Buffer: {BUFFER_METERS:.0f} m")
    print(f"Output CRS: {TARGET_CRS}")
    print(
        "Output resolution: "
        f"{abs(dst_transform.a):.2f} x "
        f"{abs(dst_transform.e):.2f} m"
    )
    print(
        f"Output dimensions: "
        f"{dst_width} x {dst_height}"
    )
    print(f"NoData: {OUTPUT_NODATA}")
    print(f"Valid pixels: {valid.size}")

    if valid.size > 0:
        print(
            f"Elevation min: {valid.min():.2f} m"
        )
        print(
            f"Elevation max: {valid.max():.2f} m"
        )
        print(
            f"Elevation mean: {valid.mean():.2f} m"
        )

    print(f"Output: {OUTPUT_FILE}")
    print()
    print("DEM preparation completed successfully.")


if __name__ == "__main__":
    main()