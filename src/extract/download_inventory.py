"""Download the 2024 Rio Grande do Sul landslide initiation inventory.

Source
------
Andrades-Filho et al. (2025)
Zenodo record: https://zenodo.org/records/15618462
DOI: 10.5281/zenodo.15618462
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve


RECORD_ID = "15618462"
BASE_URL = f"https://zenodo.org/records/{RECORD_ID}/files"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "landslides_rs_2024"


FILES = {
    "Landslides_Initiation_RS_Brazil_2024.cpg":
        "ae3b3df9970b49b6523e608759bc957d",
    "Landslides_Initiation_RS_Brazil_2024.dbf":
        "fa793ba90d63fcbf90c4dce6b18d2c25",
    "Landslides_Initiation_RS_Brazil_2024.prj":
        "050b81311006307959315e2c0398b1e9",
    "Landslides_Initiation_RS_Brazil_2024.shp":
        "bf7482a3be617ae57c4765300fa03a47",
    "Landslides_Initiation_RS_Brazil_2024.shx":
        "dada579deb492b5c7ba817ef31375534",
}


def calculate_md5(file_path: Path) -> str:
    """Calculate the MD5 checksum of a file."""
    md5_hash = hashlib.md5()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def download_file(filename: str, expected_md5: str) -> None:
    """Download one file and validate its checksum."""
    destination = OUTPUT_DIR / filename

    if destination.exists():
        current_md5 = calculate_md5(destination)

        if current_md5 == expected_md5:
            print(f"[OK] {filename} already exists and is valid.")
            return

        print(f"[WARNING] Invalid checksum for {filename}. Downloading again.")

    url = f"{BASE_URL}/{filename}?download=1"

    print(f"[DOWNLOAD] {filename}")
    urlretrieve(url, destination)

    downloaded_md5 = calculate_md5(destination)

    if downloaded_md5 != expected_md5:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum validation failed for {filename}. "
            f"Expected {expected_md5}, got {downloaded_md5}."
        )

    print(f"[OK] {filename} downloaded and validated.")


def main() -> None:
    """Download all required shapefile components."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("GeoRisk ML - Landslide Inventory Download")
    print(f"Destination: {OUTPUT_DIR}")
    print()

    for filename, checksum in FILES.items():
        download_file(filename, checksum)

    print()
    print("Inventory download completed successfully.")


if __name__ == "__main__":
    main()