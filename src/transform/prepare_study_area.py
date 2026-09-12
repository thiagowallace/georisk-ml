"""Prepare the Santa Tereza/RS municipal boundary for GeoRisk ML."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ibge"
    / "municipalities_2024"
    / "RS_Municipios_2024.shp"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "study_area"
OUTPUT_FILE = OUTPUT_DIR / "santa_tereza.gpkg"

TARGET_MUNICIPALITY = "Santa Tereza"
TARGET_CRS = "EPSG:31982"


def main() -> None:
    """Select Santa Tereza and export it in the project CRS."""

    print("GeoRisk ML - Prepare Study Area")
    print(f"Input: {INPUT_FILE}")
    print(f"Municipality: {TARGET_MUNICIPALITY}")
    print()

    gdf = gpd.read_file(INPUT_FILE)

    print(f"Input records: {len(gdf)}")
    print(f"Input CRS: {gdf.crs}")

    study_area = gdf.loc[
        gdf["NM_MUN"].str.strip().str.casefold()
        == TARGET_MUNICIPALITY.casefold()
    ].copy()

    if study_area.empty:
        raise ValueError(
            f"Municipality '{TARGET_MUNICIPALITY}' not found."
        )

    if len(study_area) != 1:
        raise ValueError(
            f"Expected 1 municipality, found {len(study_area)}."
        )

    print()
    print("Municipality found:")
    print(
        study_area[
            ["CD_MUN", "NM_MUN", "SIGLA_UF", "AREA_KM2"]
        ].to_string(index=False)
    )

    study_area = study_area.to_crs(TARGET_CRS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    study_area.to_file(
        OUTPUT_FILE,
        layer="santa_tereza",
        driver="GPKG",
    )

    print()
    print(f"Output CRS: {study_area.crs}")
    print(f"Output records: {len(study_area)}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("Study area created successfully.")


if __name__ == "__main__":
    main()