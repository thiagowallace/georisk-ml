"""Create the GeoRisk ML portfolio figure from existing, read-only artifacts.

Usage from any directory:
    python path/to/georisk-ml/scripts/make_final_figure.py

Only outputs/figures/georisk_ml_final_figure.png is written. No model is run.
For identical inputs and rendering-library versions the PNG is deterministic.
Fallbacks (only when optional evidence is absent): the user-specified rounded
Stage 4 ROC-AUC values 0.860/0.848 and the Stage 5 test count 130. Every fallback
is printed in the JSON receipt. Existing but invalid evidence causes an error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FINAL = Path("outputs/models/final")
REQUIRED = {
    "summary": FINAL / "susceptibility_summary.json",
    "comparison": FINAL / "model_surface_comparison.csv",
    "map": FINAL / "map_logistic.png",
}
SPATIAL = Path("outputs/models/spatial_validation/summary_metrics.json")
TEST_LOG = FINAL / "test_results.log"
OUTPUT = Path("outputs/figures/georisk_ml_final_figure.png")
BG, INK, MUTED, ACCENT = "#f2f5f4", "#183642", "#586e76", "#247486"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_log(path):
    raw = path.read_bytes()
    # PowerShell redirection can emit UTF-16 with a BOM; Python/other shells use UTF-8.
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def load_evidence(root):
    inputs = {key: root / rel for key, rel in REQUIRED.items()}
    for path in inputs.values():
        if not path.is_file():
            raise ValueError(f"Missing required input: {path.relative_to(root)}")
    summary = read_json(inputs["summary"])
    cells = summary["valid_cells"]
    if isinstance(cells, bool) or not isinstance(cells, int) or cells <= 0:
        raise ValueError("susceptibility_summary.json: valid_cells must be a positive integer")
    resolution = np.asarray(summary["grid"]["resolution"], dtype=float)
    if (resolution.shape != (2,) or not np.isfinite(resolution).all()
            or np.any(resolution <= 0) or resolution[0] != resolution[1]):
        raise ValueError("Expected a finite, positive, square spatial resolution")
    with inputs["comparison"].open(encoding="utf-8-sig", newline="") as stream:
        comparison = list(csv.DictReader(stream))
    if len(comparison) != 1 or int(comparison[0]["n_cells"]) != cells:
        raise ValueError("Cell count differs between summary and surface comparison CSV")
    # Agreement is checked for provenance, but never presented as validation ROC-AUC.
    for key in ("pearson_correlation", "mean_absolute_difference", "rmse"):
        value = float(comparison[0][key])
        if not np.isfinite(value) or not np.isclose(
                value, summary["surface_comparison"][key], rtol=0, atol=1e-12):
            raise ValueError(f"Inconsistent surface comparison: {key}")
    used = list(inputs.values())
    fallbacks = []
    spatial_path = root / SPATIAL
    if spatial_path.is_file():
        spatial = read_json(spatial_path)
        auc = {name: float(spatial["models"][name]["roc_auc"]["mean"])
               for name in ("logistic", "random_forest")}
        folds = int(spatial["spatial"]["n_splits"])
        if folds < 2:
            raise ValueError("Spatial validation must contain at least two folds")
        validation_caption = f"{folds}-fold spatial cross-validation · mean ROC-AUC"
        used.append(spatial_path)
    else:
        # Explicit, requested presentation values; never substitute surface correlation.
        auc = {"logistic": 0.860, "random_forest": 0.848}
        validation_caption = "Spatial cross-validation · mean ROC-AUC"
        fallbacks.append(f"{SPATIAL.as_posix()} absent: requested ROC-AUC values 0.860 / 0.848")
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in auc.values()):
        raise ValueError("ROC-AUC values must be finite and in [0, 1]")
    log_path = root / TEST_LOG
    if log_path.is_file():
        log = read_log(log_path)
        matches = re.findall(r"(\d+) passed\b", log)
        if not matches or re.search(r"\b[1-9]\d* (?:failed|errors?)\b", log):
            raise ValueError("Test log does not contain a successful pytest summary")
        tests = int(matches[-1])
        used.append(log_path)
    else:
        # Requested Stage 5 count, not a test execution performed by this script.
        tests = 130
        fallbacks.append(f"{TEST_LOG.as_posix()} absent: requested Stage 5 count of 130 tests")
    return {"cells": cells, "resolution_m": float(resolution[0]), "auc": auc,
            "tests": tests, "validation_caption": validation_caption,
            "inputs": used, "fallbacks": fallbacks}


def extract_map_panels(path):
    """Remove neutral axes/text; retain map colors, aspect ratio and source legend.

    Stage 5 places the map in the left 84% and its colorbar on the right.
    Locate chromatic pixels in each zone; pad the map bounding box by 12 pixels.
    Neutral pixels/holes INSIDE that box remain untouched. Fail clearly if the
    source layout no longer supports this documented cropping rule.
    """
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"))
    chromatic = np.ptp(rgb.astype(np.int16), axis=2) >= 6
    split = int(rgb.shape[1] * .84)
    y, x = np.nonzero(chromatic[:, :split])
    if x.size < 1000:
        raise ValueError("Cannot locate the main map in map_logistic.png")
    left, right = max(0, x.min()-12), min(split, x.max()+13)
    top, bottom = max(0, y.min()-12), min(len(rgb), y.max()+13)
    map_rgb = rgb[top:bottom, left:right].copy()
    counts = chromatic[:, split:].sum(axis=0)
    column = split + int(np.argmax(counts))
    rows = np.flatnonzero(chromatic[:, column])
    if rows.size < rgb.shape[0] / 4:
        raise ValueError("Cannot locate the original map colorbar")
    # Source runs high-to-low from top to bottom. Reuse exact colors left-to-right 0..1.
    gradient = rgb[rows.min():rows.max()+1, column][::-1][None, :, :].copy()
    return map_rgb, gradient


def render(root, evidence):
    map_rgb, gradient = extract_map_panels(root / REQUIRED["map"])
    settings = {"font.family": "DejaVu Sans", "font.size": 12,
                "text.color": INK, "figure.facecolor": BG, "savefig.facecolor": BG}
    with plt.rc_context(settings):
        fig = plt.figure(figsize=(16, 9), dpi=150)

        def text(x, y, label, size=12, color=INK, weight="normal", **kwargs):
            return fig.text(x, y, label, fontsize=size, color=color, weight=weight,
                            va="center", **kwargs)

        def card(x, y, width, height, color="white"):
            fig.add_artist(FancyBboxPatch((x, y), width, height,
                boxstyle="round,pad=0.012,rounding_size=0.015",
                transform=fig.transFigure, facecolor=color, edgecolor="none", zorder=-1))

        text(.055, .910, "GeoRisk ML", 39, weight="bold")
        text(.055, .851, "End-to-End Data Science Project", 20, color=ACCENT)
        text(.055, .802, "Landslide susceptibility mapping — Santa Tereza, RS, Brazil", 12)

        text(.055, .690, f"{evidence['cells']:,}", 48, weight="bold")
        text(.057, .629, "cells mapped", 15, color=MUTED)
        text(.377, .690, f"{evidence['resolution_m']:g} m", 38, weight="bold")
        text(.380, .629, "spatial resolution", 13, color=MUTED)

        text(.055, .565, evidence["validation_caption"], 11, color=MUTED)
        for x, model, label, subtitle in (
            (.055, "logistic", "Spatial CV ROC-AUC", "Logistic Regression"),
            (.326, "random_forest", "Random Forest ROC-AUC", "Spatial CV"),
        ):
            card(x, .382, .239, .142)
            text(x+.013, .482, f"{evidence['auc'][model]:.3f}", 32, color=ACCENT, weight="bold")
            text(x+.013, .430, label, 11, weight="bold")
            text(x+.013, .401, subtitle, 10, color=MUTED)

        text(.055, .329, f"{evidence['tests']} automated tests", 19, weight="bold")
        text(.055, .279, "BUILT WITH", 9, color=MUTED, weight="bold")
        text(.055, .239, "Python  ·  Pandas  ·  NumPy  ·  scikit-learn", 12)
        text(.055, .202, "GeoPandas  ·  Rasterio  ·  Matplotlib  ·  pytest  ·  Git/GitHub", 11)

        card(.627, .180, .322, .754)
        text(.647, .906, "SPATIAL INFERENCE", 9, color=MUTED, weight="bold")
        text(.647, .875, "Logistic Regression", 15, weight="bold")
        ax = fig.add_axes([.646, .307, .281, .535])
        ax.imshow(map_rgb, interpolation="nearest", aspect="equal")
        ax.set_axis_off()
        legend = fig.add_axes([.665, .273, .238, .013])
        legend.imshow(gradient, aspect="auto", interpolation="nearest", extent=[0, 1, 0, 1])
        legend.set_axis_off()
        text(.665, .253, "0", 9, color=MUTED)
        text(.903, .253, "1", 9, color=MUTED, ha="right")
        text(.784, .229, "Landslide susceptibility score", 10, ha="center")
        text(.784, .202, "NoData / outside valid environmental mask", 8, color=MUTED, ha="center")

        fig.add_artist(plt.Line2D([.055, .949], [.150, .150],
                                 transform=fig.transFigure, color="#d4dedf", linewidth=1))
        text(.5, .110,
             "Data Preparation → Feature Engineering → Modeling → Spatial Validation → Spatial Inference",
             12, ha="center", weight="bold")
        text(.5, .052, "From raw geospatial data to reproducible ML inference", 12, color=MUTED, ha="center")
        output = root / OUTPUT
        output.parent.mkdir(parents=True, exist_ok=True)
        # Fixed canvas, bundled font, no date/randomness, stable metadata.
        fig.savefig(output, dpi=150, metadata={"Software": "GeoRisk ML"})
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        evidence = load_evidence(ROOT)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence["inputs"]}
        output = render(ROOT, evidence)
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != digest for p, digest in before.items()):
            raise ValueError("An input changed during rendering")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Figure generation error: {exc}\n")
    print(json.dumps({"inputs": [p.relative_to(ROOT).as_posix() for p in evidence["inputs"]],
        "output": output.relative_to(ROOT).as_posix(), "size_px": [2400, 1350],
        "values": {key: evidence[key] for key in ("cells", "resolution_m", "auc", "tests")},
        "fallbacks": evidence["fallbacks"], "inputs_unchanged": True,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
