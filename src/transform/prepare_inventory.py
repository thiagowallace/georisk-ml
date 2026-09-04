"""Prepare the landslide initiation inventory for the GeoRisk ML pipeline."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "landslides_rs_2024"
    / "Landslides_Initiation_RS_Brazil_2024.shp"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "landslides_rs_2024"
OUTPUT_FILE = OUTPUT_DIR / "landslide_initiation.gpkg"

LAYER_NAME = "landslide_initiation"

# Spatial tolerance in meters.
DUPLICATE_TOLERANCE = 0.01


def remove_spatial_duplicates(
    gdf: gpd.GeoDataFrame,
    tolerance: float = DUPLICATE_TOLERANCE,
) -> gpd.GeoDataFrame:
    """Remove points occupying the same location within a spatial tolerance."""

    working = gdf.copy()

    working["_x_round"] = (working.geometry.x / tolerance).round()
    working["_y_round"] = (working.geometry.y / tolerance).round()

    duplicated = working.duplicated(
        subset=["_x_round", "_y_round"],
        keep="first",
    )

    duplicate_count = int(duplicated.sum())

    print(
        f"Spatial duplicates within {tolerance} m tolerance: "
        f"{duplicate_count}"
    )

    working = working.loc[~duplicated].copy()

    working.drop(
        columns=["_x_round", "_y_round"],
        inplace=True,
    )

    return working


def main() -> None:
    """Prepare and export the intermediate inventory."""

    print("GeoRisk ML - Prepare Landslide Inventory")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    gdf = gpd.read_file(INPUT_FILE)

    print(f"Input records: {len(gdf)}")

    # Preserve attributes supplied by the original dataset.
    gdf = gdf.rename(
        columns={
            "Id": "source_id",
            "X": "source_x",
            "Y": "source_y",
        }
    )

    gdf = remove_spatial_duplicates(gdf)

    # Reset the index after duplicate removal.
    gdf = gdf.reset_index(drop=True)

    # Create a unique identifier for the GeoRisk ML pipeline.
    gdf.insert(
        0,
        "landslide_id",
        range(1, len(gdf) + 1),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf.to_file(
        OUTPUT_FILE,
        layer=LAYER_NAME,
        driver="GPKG",
    )

    print(f"Output records: {len(gdf)}")
    print(f"Unique landslide_id: {gdf['landslide_id'].is_unique}")
    print(f"CRS: {gdf.crs}")
    print()
    print("Intermediate inventory created successfully.")


if __name__ == "__main__":
    main()