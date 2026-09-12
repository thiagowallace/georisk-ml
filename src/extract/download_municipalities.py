"""Download the 2024 municipal boundaries for Rio Grande do Sul from IBGE."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "ibge" / "municipalities_2024"
ZIP_FILE = OUTPUT_DIR / "RS_Municipios_2024.zip"

IBGE_URL = (
    "https://geoftp.ibge.gov.br/"
    "organizacao_do_territorio/malhas_territoriais/"
    "malhas_municipais/municipio_2024/UFs/RS/"
    "RS_Municipios_2024.zip"
)


def download_file() -> None:
    """Download the IBGE municipal boundary archive if it is not present."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_FILE.exists():
        print(f"[OK] File already exists: {ZIP_FILE.name}")
        return

    print(f"Downloading: {IBGE_URL}")
    urlretrieve(IBGE_URL, ZIP_FILE)

    print(f"[OK] Download completed: {ZIP_FILE.name}")


def extract_archive() -> None:
    """Extract the IBGE archive into the raw data directory."""

    shapefile = OUTPUT_DIR / "RS_Municipios_2024.shp"

    if shapefile.exists():
        print("[OK] Municipal boundaries already extracted.")
        return

    print("Extracting archive...")

    with ZipFile(ZIP_FILE, "r") as archive:
        archive.extractall(OUTPUT_DIR)

    print("[OK] Extraction completed.")


def main() -> None:
    """Download and extract the IBGE municipal boundaries."""

    print("GeoRisk ML - IBGE Municipal Boundaries")
    print(f"Destination: {OUTPUT_DIR}")
    print()

    download_file()
    extract_archive()

    print()
    print("Municipal boundary acquisition completed successfully.")


if __name__ == "__main__":
    main()