"""Audit the 2024 Rio Grande do Sul landslide initiation inventory."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "landslides_rs_2024"
    / "Landslides_Initiation_RS_Brazil_2024.shp"
)


def main() -> None:
    """Run structural and spatial quality checks on the inventory."""

    print("GeoRisk ML - Landslide Inventory Audit")
    print(f"Input: {INPUT_FILE}")
    print()

    gdf = gpd.read_file(INPUT_FILE)

    print("=== STRUCTURE ===")
    print(f"Rows: {len(gdf)}")
    print(f"Columns: {len(gdf.columns)}")
    print(f"Column names: {gdf.columns.tolist()}")
    print(f"CRS: {gdf.crs}")
    print()

    print("=== GEOMETRY ===")
    print(f"Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")
    print(f"Null geometries: {gdf.geometry.isna().sum()}")
    print(f"Empty geometries: {gdf.geometry.is_empty.sum()}")
    print(f"Invalid geometries: {(~gdf.geometry.is_valid).sum()}")
    print()

    print("=== DUPLICATES ===")
    print(f"Duplicated IDs: {gdf['Id'].duplicated().sum()}")
    print(f"Duplicated geometries: {gdf.geometry.duplicated().sum()}")

    duplicated_xy = gdf.duplicated(subset=["X", "Y"]).sum()
    print(f"Duplicated X/Y pairs: {duplicated_xy}")
    print()

    print("=== COORDINATE CONSISTENCY ===")

    geometry_x = gdf.geometry.x
    geometry_y = gdf.geometry.y

    x_difference = (geometry_x - gdf["X"]).abs()
    y_difference = (geometry_y - gdf["Y"]).abs()

    print(f"Maximum |geometry.x - X|: {x_difference.max()}")
    print(f"Maximum |geometry.y - Y|: {y_difference.max()}")
    print()

    print("=== SPATIAL EXTENT ===")

    minx, miny, maxx, maxy = gdf.total_bounds

    print(f"Min X: {minx}")
    print(f"Min Y: {miny}")
    print(f"Max X: {maxx}")
    print(f"Max Y: {maxy}")
    print()

    print("=== ATTRIBUTE SUMMARY ===")

    summary = gdf[["Id", "X", "Y"]].describe()
    print(summary)
    print()

    checks = {
        "null_geometry": int(gdf.geometry.isna().sum()),
        "empty_geometry": int(gdf.geometry.is_empty.sum()),
        "invalid_geometry": int((~gdf.geometry.is_valid).sum()),
        "duplicated_id": int(gdf["Id"].duplicated().sum()),
        "duplicated_geometry": int(gdf.geometry.duplicated().sum()),
        "duplicated_xy": int(duplicated_xy),
    }

    critical_problems = sum(checks.values())

    print("=== AUDIT RESULT ===")

    if critical_problems == 0:
        print("[OK] No structural quality problems detected.")
    else:
        print("[WARNING] Potential quality issues were detected.")

        for check_name, count in checks.items():
            if count > 0:
                print(f"  - {check_name}: {count}")


if __name__ == "__main__":
    main()