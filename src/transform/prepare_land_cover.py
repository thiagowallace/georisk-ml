"""Harmonize MapBiomas 2023 land cover to the prepared DEM grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LAND_COVER_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mapbiomas"
    / "brazil_coverage-col11_2023.tif"
)

DEM_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dem"
    / "santa_tereza_dem_30m.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "land_cover"
)

OUTPUT_FILE = OUTPUT_DIR / "santa_tereza_land_cover_2023.tif"

# The raw raster has no explicit NoData. In the harmonized raster,
# 0 is reserved exclusively for NoData and is not a valid class.
OUTPUT_NODATA = 0


def main() -> None:
    """Reproject categorical land cover and verify the saved grid."""

    print("GeoRisk ML - Prepare MapBiomas Land Cover 2023")
    print()

    for input_file in (LAND_COVER_FILE, DEM_FILE):
        if not input_file.is_file():
            raise FileNotFoundError(f"Input not found: {input_file}")

    with rasterio.open(DEM_FILE) as reference:
        if reference.crs is None:
            raise ValueError("Reference DEM has no CRS.")

        reference_crs = reference.crs
        reference_transform = reference.transform
        reference_width = reference.width
        reference_height = reference.height
        reference_bounds = reference.bounds
        reference_resolution = reference.res

    destination = np.zeros(
        (reference_height, reference_width),
        dtype=np.uint8,
    )

    with rasterio.open(LAND_COVER_FILE) as src:
        if src.crs is None:
            raise ValueError("Land cover raster has no CRS.")

        # Read through the raster band without loading the national raster.
        # Nearest-neighbor resampling preserves categorical class codes.
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=OUTPUT_NODATA,
            resampling=Resampling.nearest,
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=OUTPUT_NODATA,
        compress="deflate",
        crs=reference_crs,
        transform=reference_transform,
        width=reference_width,
        height=reference_height,
    ) as dst:
        dst.write(destination, 1)

    with rasterio.open(OUTPUT_FILE) as result:
        grid_matches_dem = (
            result.crs == reference_crs
            and result.transform == reference_transform
            and result.width == reference_width
            and result.height == reference_height
            and result.bounds == reference_bounds
            and result.res == reference_resolution
        )

        print(f"Source: {LAND_COVER_FILE}")
        print(f"Reference raster: {DEM_FILE}")
        print(f"Output: {OUTPUT_FILE}")
        print(f"CRS: {result.crs}")
        print(f"Resolution: {result.res[0]} x {result.res[1]}")
        print(f"Dimensions: {result.width} x {result.height}")
        print(f"Bounds: {result.bounds}")
        print(f"NoData: {result.nodata}")
        print(f"Grid matches DEM: {grid_matches_dem}")

        if not grid_matches_dem:
            raise RuntimeError("Output grid does not exactly match the DEM.")

        output_array = result.read(1)

    valid = output_array[output_array != OUTPUT_NODATA]
    codes, counts = np.unique(valid, return_counts=True)

    print()
    print(f"Valid unique codes: {codes.tolist()}")
    print("Percentages use valid land cover pixels as the denominator.")
    print(f"{'Code':>8} {'Pixels':>12} {'Percent':>12}")

    for code, count in zip(codes, counts):
        percentage = 100.0 * int(count) / valid.size
        print(f"{code:>8} {int(count):>12} {percentage:>11.4f}%")

    print(f"NoData pixels: {np.count_nonzero(output_array == OUTPUT_NODATA)}")
    print()
    print("Land cover preparation completed successfully.")


if __name__ == "__main__":
    main()
