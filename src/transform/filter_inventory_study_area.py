"""Filter the landslide inventory to the GeoRisk ML pilot study area."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INVENTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "landslides_rs_2024"
    / "landslide_initiation.gpkg"
)

STUDY_AREA_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "study_area"
    / "santa_tereza.gpkg"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "study_area"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "landslide_initiation_santa_tereza.gpkg"
)

OUTPUT_LAYER = "landslide_initiation_santa_tereza"


def main() -> None:
    """Filter landslide initiation points to Santa Tereza/RS."""

    print("GeoRisk ML - Filter Inventory to Study Area")
    print()

    inventory = gpd.read_file(INVENTORY_FILE)
    study_area = gpd.read_file(STUDY_AREA_FILE)

    print(f"Inventory records: {len(inventory)}")
    print(f"Inventory CRS: {inventory.crs}")
    print(f"Study area records: {len(study_area)}")
    print(f"Study area CRS: {study_area.crs}")

    if inventory.crs != study_area.crs:
        raise ValueError(
            "Inventory and study area must use the same CRS."
        )

    if len(study_area) != 1:
        raise ValueError(
            f"Expected one study-area polygon, found {len(study_area)}."
        )

    study_geometry = study_area.geometry.union_all()

    mask = inventory.geometry.intersects(study_geometry)

    selected = inventory.loc[mask].copy()

    selected = selected.reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected.to_file(
        OUTPUT_FILE,
        layer=OUTPUT_LAYER,
        driver="GPKG",
    )

    municipality_area = float(
        study_area["AREA_KM2"].iloc[0]
    )

    density = (
        len(selected) / municipality_area
        if municipality_area > 0
        else float("nan")
    )

    print()
    print("--- RESULT ---")
    print(f"RS inventory records: {len(inventory)}")
    print(f"Santa Tereza records: {len(selected)}")
    print(f"Municipality area: {municipality_area:.3f} km²")
    print(
        "Initiation-point density: "
        f"{density:.2f} points/km²"
    )
    print(f"CRS: {selected.crs}")
    print(
        "Invalid geometries: "
        f"{int((~selected.geometry.is_valid).sum())}"
    )
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("Study-area inventory created successfully.")


if __name__ == "__main__":
    main()