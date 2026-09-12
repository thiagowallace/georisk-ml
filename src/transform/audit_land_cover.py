"""Audit native MapBiomas land cover within the buffered study area."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LAND_COVER_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mapbiomas"
    / "brazil_coverage-col11_2023.tif"
)

STUDY_AREA_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "study_area"
    / "santa_tereza.gpkg"
)

TARGET_CRS = "EPSG:31982"
BUFFER_METERS = 1000


def main() -> None:
    """Report raster metadata and class counts without writing data."""

    print("GeoRisk ML - Audit MapBiomas Land Cover 2023")
    print()

    for input_file in (LAND_COVER_FILE, STUDY_AREA_FILE):
        if not input_file.is_file():
            raise FileNotFoundError(f"Input not found: {input_file}")

    study_area = gpd.read_file(STUDY_AREA_FILE)

    print(f"Study area CRS: {study_area.crs}")

    if study_area.crs is None or study_area.crs.to_epsg() != 31982:
        raise ValueError(f"Study area must be in {TARGET_CRS}.")

    if (
        study_area.empty
        or study_area.geometry.isna().any()
        or study_area.geometry.is_empty.any()
        or not study_area.geometry.is_valid.all()
        or not study_area.geometry.geom_type.isin(
            ["Polygon", "MultiPolygon"]
        ).all()
    ):
        raise ValueError("Study area must contain valid, nonempty polygons.")

    buffered_geometry = (
        study_area.geometry.buffer(BUFFER_METERS).union_all()
    )
    buffered = gpd.GeoDataFrame(
        geometry=[buffered_geometry],
        crs=study_area.crs,
    )

    with rasterio.open(LAND_COVER_FILE) as src:
        print(f"Raster: {LAND_COVER_FILE.name}")
        print(f"Raster CRS: {src.crs}")
        print(f"Resolution: {src.res[0]} x {src.res[1]} (raster CRS units)")
        print(f"Dtype: {src.dtypes[0]}")
        print(f"Metadata NoData: {src.nodata}")

        if src.crs is None:
            raise ValueError("Land cover raster has no CRS.")

        # Transform only the buffered geometry; keep the native raster grid.
        buffered_source_crs = buffered.to_crs(src.crs)
        geometries = [
            geom.__geo_interface__
            for geom in buffered_source_crs.geometry
        ]

        # Retain the source mask and the geometry mask without filling with 0.
        clipped, _ = mask(
            src,
            geometries,
            indexes=1,
            crop=True,
            filled=False,
        )

    height, width = clipped.shape
    values = clipped.compressed()
    codes, counts = np.unique(values, return_counts=True)

    print()
    print(f"Buffer: {BUFFER_METERS} m")
    print(f"Clip dimensions: {width} x {height} (width x height)")
    print(f"Total pixels in clip: {clipped.size}")
    print(f"Unmasked pixels in buffer: {values.size}")
    print(f"Masked pixels (outside buffer or source mask): {clipped.size - values.size}")
    print()
    print("Classes found in Santa Tereza + 1 km buffer")
    print(f"Unique codes: {codes.tolist()}")
    print("Percentages use unmasked pixels in the buffer as the denominator.")
    print(f"{'Code':>8} {'Pixels':>12} {'Percent':>12}")

    for code, count in zip(codes, counts):
        percentage = 100.0 * int(count) / values.size
        print(f"{code:>8} {int(count):>12} {percentage:>11.4f}%")

    zero_count = int(np.count_nonzero(values == 0))
    print()
    print(f"Code 0 present among unmasked pixels: {zero_count > 0}")
    print(f"Code 0 pixel count: {zero_count}")
    print("Source-masked values are excluded; code 0 is not assumed to be NoData.")
    print()
    print("Land cover audit completed successfully.")


if __name__ == "__main__":
    main()
