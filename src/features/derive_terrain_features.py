"""Derive topographic features from the prepared Copernicus DEM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    / "terrain_features"
)

SLOPE_FILE = OUTPUT_DIR / "santa_tereza_slope.tif"
ASPECT_FILE = OUTPUT_DIR / "santa_tereza_aspect.tif"

OUTPUT_NODATA = -9999.0


def write_raster(
    output_file: Path,
    array: np.ndarray,
    profile: dict,
) -> None:
    """Write a single-band raster."""

    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    with rasterio.open(
        output_file,
        "w",
        **output_profile,
    ) as dst:
        dst.write(array.astype(np.float32), 1)


def main() -> None:
    """Generate slope and aspect from the prepared DEM."""

    print("GeoRisk ML - Derive Terrain Features")
    print()

    if not DEM_FILE.exists():
        raise FileNotFoundError(
            f"Prepared DEM not found: {DEM_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(DEM_FILE) as src:

        dem = src.read(1).astype(np.float64)

        profile = src.profile.copy()

        nodata = src.nodata

        x_resolution = abs(src.transform.a)
        y_resolution = abs(src.transform.e)

        print(f"DEM: {DEM_FILE.name}")
        print(f"CRS: {src.crs}")
        print(
            "Resolution: "
            f"{x_resolution:.2f} x "
            f"{y_resolution:.2f} m"
        )

    valid_mask = np.isfinite(dem)

    if nodata is not None:
        valid_mask &= dem != nodata

    # Use NaN internally so derivatives close to NoData
    # cells are not treated as real terrain values.
    working_dem = dem.copy()

    working_dem[~valid_mask] = np.nan

    # np.gradient axis 0 follows increasing array rows
    # (southward in a north-up raster).
    # Therefore the northing derivative receives a
    # negative sign.
    gradient_row, gradient_col = np.gradient(
        working_dem,
        y_resolution,
        x_resolution,
    )

    dz_dx = gradient_col
    dz_dy = -gradient_row

    # Slope in degrees.
    slope = np.degrees(
        np.arctan(
            np.sqrt(
                dz_dx ** 2
                + dz_dy ** 2
            )
        )
    )

    # Downslope direction expressed as an azimuth:
    # 0° = North
    # 90° = East
    # 180° = South
    # 270° = West
    aspect = (
        np.degrees(
            np.arctan2(
                -dz_dx,
                -dz_dy,
            )
        )
        + 360.0
    ) % 360.0

    derivative_mask = (
        valid_mask
        & np.isfinite(slope)
        & np.isfinite(aspect)
    )

    slope_output = np.full(
        dem.shape,
        OUTPUT_NODATA,
        dtype=np.float32,
    )

    aspect_output = np.full(
        dem.shape,
        OUTPUT_NODATA,
        dtype=np.float32,
    )

    slope_output[derivative_mask] = (
        slope[derivative_mask]
    )

    aspect_output[derivative_mask] = (
        aspect[derivative_mask]
    )

    write_raster(
        SLOPE_FILE,
        slope_output,
        profile,
    )

    write_raster(
        ASPECT_FILE,
        aspect_output,
        profile,
    )

    valid_slope = slope_output[
        slope_output != OUTPUT_NODATA
    ]

    valid_aspect = aspect_output[
        aspect_output != OUTPUT_NODATA
    ]

    print()
    print("--- SLOPE ---")
    print(f"Valid pixels: {valid_slope.size}")
    print(
        f"Minimum: {valid_slope.min():.2f} degrees"
    )
    print(
        f"Maximum: {valid_slope.max():.2f} degrees"
    )
    print(
        f"Mean: {valid_slope.mean():.2f} degrees"
    )
    print(f"Output: {SLOPE_FILE}")

    print()
    print("--- ASPECT ---")
    print(f"Valid pixels: {valid_aspect.size}")
    print(
        f"Minimum: {valid_aspect.min():.2f} degrees"
    )
    print(
        f"Maximum: {valid_aspect.max():.2f} degrees"
    )
    print(f"Output: {ASPECT_FILE}")

    print()
    print("Terrain feature derivation completed successfully.")


if __name__ == "__main__":
    main()