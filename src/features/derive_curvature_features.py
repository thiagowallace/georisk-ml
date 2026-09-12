"""Derive profile and plan curvature from the prepared DEM."""

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

PROFILE_CURVATURE_FILE = (
    OUTPUT_DIR
    / "santa_tereza_profile_curvature.tif"
)

PLAN_CURVATURE_FILE = (
    OUTPUT_DIR
    / "santa_tereza_plan_curvature.tif"
)

OUTPUT_NODATA = -9999.0

MIN_SLOPE_DEGREES = 0.1
MIN_GRADIENT = np.tan(np.radians(MIN_SLOPE_DEGREES))


def calculate_curvatures(
    elevation: np.ndarray,
    cellsize_x: float,
    cellsize_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate profile and plan curvature.

    Parameters
    ----------
    elevation:
        DEM array. Invalid locations should be represented by NaN.

    cellsize_x:
        Horizontal pixel size in map units.

    cellsize_y:
        Vertical pixel size in map units.

    Returns
    -------
    profile_curvature, plan_curvature
    """

    # First derivatives.
    dz_drow, dz_dx = np.gradient(
        elevation,
        cellsize_y,
        cellsize_x,
    )

    # Raster rows increase southward. Multiplying by -1
    # converts the row derivative to the northing direction.
    dz_dy = -dz_drow

    # Second derivatives.
    _, d2z_dx2 = np.gradient(
        dz_dx,
        cellsize_y,
        cellsize_x,
    )

    d_drow_dy, _ = np.gradient(
        dz_dy,
        cellsize_y,
        cellsize_x,
    )

    d2z_dy2 = -d_drow_dy

    d_drow_dx, _ = np.gradient(
        dz_dx,
        cellsize_y,
        cellsize_x,
    )

    d2z_dxdy = -d_drow_dx

    p = dz_dx
    q = dz_dy
    r = d2z_dx2
    s = d2z_dxdy
    t = d2z_dy2

    gradient_squared = p**2 + q**2

    # Exclude nearly horizontal surfaces to avoid unstable curvature.
    gradient = np.sqrt(gradient_squared)
    valid_gradient = gradient >= MIN_GRADIENT
    # Use the previous gradient criterion only as a reporting baseline.
    valid_before_threshold = (
        np.isfinite(elevation)
        & np.isfinite(gradient)
        & (gradient_squared > 1e-12)
        & np.isfinite(r)
        & np.isfinite(s)
        & np.isfinite(t)
    )
    removed_by_threshold = valid_before_threshold & ~valid_gradient
    valid_after_threshold = valid_before_threshold & valid_gradient

    print(
        f"Minimum slope for curvature: {MIN_SLOPE_DEGREES:.2f} degrees"
    )
    print(
        "Valid curvature pixels before slope threshold: "
        f"{np.count_nonzero(valid_before_threshold)}"
    )
    print(
        "Pixels removed by slope threshold: "
        f"{np.count_nonzero(removed_by_threshold)}"
    )
    print(
        "Valid curvature pixels after threshold: "
        f"{np.count_nonzero(valid_after_threshold)}"
    )

    profile_curvature = np.full(
        elevation.shape,
        np.nan,
        dtype=np.float64,
    )

    plan_curvature = np.full(
        elevation.shape,
        np.nan,
        dtype=np.float64,
    )

    # Profile curvature:
    # curvature in the direction of maximum slope.
    profile_numerator = (
        r * p**2
        + 2.0 * s * p * q
        + t * q**2
    )

    profile_denominator = (
        gradient_squared
        * np.power(
            1.0 + gradient_squared,
            1.5,
        )
    )

    profile_curvature[valid_gradient] = (
        -profile_numerator[valid_gradient]
        / profile_denominator[valid_gradient]
    )

    # Plan curvature:
    # curvature perpendicular to maximum slope.
    plan_numerator = (
        r * q**2
        - 2.0 * s * p * q
        + t * p**2
    )

    plan_denominator = np.power(
        gradient_squared,
        1.5,
    )

    plan_curvature[valid_gradient] = (
        plan_numerator[valid_gradient]
        / plan_denominator[valid_gradient]
    )

    return (
        profile_curvature,
        plan_curvature,
    )


def write_raster(
    output_file: Path,
    array: np.ndarray,
    profile: dict,
) -> None:
    """Write a curvature raster."""

    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    output_array = np.full(
        array.shape,
        OUTPUT_NODATA,
        dtype=np.float32,
    )

    valid = np.isfinite(array)

    output_array[valid] = (
        array[valid]
        .astype(np.float32)
    )

    with rasterio.open(
        output_file,
        "w",
        **output_profile,
    ) as dst:
        dst.write(output_array, 1)


def main() -> None:
    """Derive terrain curvatures."""

    print("GeoRisk ML - Derive Curvature Features")
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

        cellsize_x = abs(src.transform.a)
        cellsize_y = abs(src.transform.e)

        print(f"DEM: {DEM_FILE.name}")
        print(f"CRS: {src.crs}")
        print(
            f"Resolution: "
            f"{cellsize_x:.2f} x "
            f"{cellsize_y:.2f} m"
        )

    valid_mask = np.isfinite(dem)

    if nodata is not None:
        valid_mask &= dem != nodata

    working_dem = dem.copy()

    working_dem[~valid_mask] = np.nan

    profile_curvature, plan_curvature = (
        calculate_curvatures(
            working_dem,
            cellsize_x,
            cellsize_y,
        )
    )

    # Require original DEM validity as well.
    profile_curvature[~valid_mask] = np.nan
    plan_curvature[~valid_mask] = np.nan

    write_raster(
        PROFILE_CURVATURE_FILE,
        profile_curvature,
        profile,
    )

    write_raster(
        PLAN_CURVATURE_FILE,
        plan_curvature,
        profile,
    )

    valid_profile = profile_curvature[
        np.isfinite(profile_curvature)
    ]

    valid_plan = plan_curvature[
        np.isfinite(plan_curvature)
    ]

    print()
    print("--- PROFILE CURVATURE ---")
    print(f"Valid pixels: {valid_profile.size}")

    if valid_profile.size > 0:
        print(
            f"Minimum: {valid_profile.min():.8f}"
        )
        print(
            f"Maximum: {valid_profile.max():.8f}"
        )
        print(
            f"Mean: {valid_profile.mean():.8f}"
        )

    print(f"Output: {PROFILE_CURVATURE_FILE}")

    print()
    print("--- PLAN CURVATURE ---")
    print(f"Valid pixels: {valid_plan.size}")

    if valid_plan.size > 0:
        print(
            f"Minimum: {valid_plan.min():.8f}"
        )
        print(
            f"Maximum: {valid_plan.max():.8f}"
        )
        print(
            f"Mean: {valid_plan.mean():.8f}"
        )

    print(f"Output: {PLAN_CURVATURE_FILE}")

    print()
    print(
        "Curvature feature derivation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
