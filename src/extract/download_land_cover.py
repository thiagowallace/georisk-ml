"""Download the annual MapBiomas Brazil land cover raster."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mapbiomas"
)

COLLECTION = 11
YEAR = 2023
FILENAME = f"brazil_coverage-col{COLLECTION}_{YEAR}.tif"

URL = (
    "https://storage.googleapis.com/mapbiomas-public/"
    f"initiatives/brasil/collection{COLLECTION}/lulc/coverage/"
    f"brazil_coverage/{FILENAME}"
)


def download_land_cover() -> Path:
    """Download the raster atomically or reuse a nonempty local file."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = OUTPUT_DIR / FILENAME

    if output_file.is_file() and output_file.stat().st_size > 0:
        print(f"[OK] Reusing existing file: {output_file.name}")
        return output_file

    partial_file = output_file.with_suffix(output_file.suffix + ".part")

    print(f"Downloading: {output_file.name}")

    try:
        urlretrieve(URL, partial_file)

        if partial_file.stat().st_size == 0:
            raise RuntimeError("Downloaded land cover raster is empty.")

        partial_file.replace(output_file)
    finally:
        partial_file.unlink(missing_ok=True)

    print(f"[OK] Download completed: {output_file.name}")

    return output_file


def main() -> None:
    """Acquire the MapBiomas Brazil land cover raster for 2023."""

    print("GeoRisk ML - MapBiomas Land Cover Download")
    print()
    print("Dataset: MapBiomas Brasil - Cobertura do Brasil - Landsat 30 m")
    print(f"Collection: {COLLECTION}")
    print(f"Year: {YEAR}")
    print(f"URL: {URL}")
    print(f"Destination: {OUTPUT_DIR / FILENAME}")
    print()

    output_file = download_land_cover()

    size_mb = output_file.stat().st_size / 1_000_000

    if size_mb >= 1000:
        print(f"Final file size: {size_mb / 1000:.2f} GB")
    else:
        print(f"Final file size: {size_mb:.2f} MB")

    print("Land cover acquisition completed successfully.")


if __name__ == "__main__":
    main()
