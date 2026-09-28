"""Audit spatial products; surface agreement is not predictive validation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import rasterio

from src.models.train_final_models import (
    OUTPUT, NOTICE, load_training_frame, verify_preservation, write_json,
)
from src.models.build_inference_grid import load_environment, grid_metadata, build_grid, TRACE
from src.models.generate_susceptibility import SCORES, NODATA, raster_path, score_array

LABELS = {"logistic": "Logistic Regression", "random_forest": "Random Forest",
          "ensemble_mean": "Ensemble mean (exploratory)"}
COLORS = {"logistic": "#245b89", "random_forest": "#b57920", "ensemble_mean": "#757044"}


def distribution(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if not v.size or not np.isfinite(v).all():
        raise ValueError("Score distribution is empty or nonfinite.")
    quantiles = np.percentile(v, [5, 25, 50, 75, 95])
    return {"count": len(v), "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "median": float(np.median(v)),
            **{f"p{p}": float(q) for p, q in zip([5, 25, 50, 75, 95], quantiles)},
            "outside_0_1": int(((v < 0) | (v > 1)).sum())}


def compare_surfaces(lr, rf) -> dict:
    lr, rf = np.asarray(lr, dtype=float), np.asarray(rf, dtype=float)
    difference = lr - rf
    correlation = float(np.corrcoef(lr, rf)[0, 1]) if np.std(lr) > 0 and np.std(rf) > 0 else None
    return {"n_cells": len(lr), "pearson_correlation": correlation,
            "mean_absolute_difference": float(np.abs(difference).mean()),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "mean_signed_difference_lr_minus_rf": float(difference.mean()),
            "min_signed_difference": float(difference.min()),
            "max_signed_difference": float(difference.max()),
            "interpretation": "Descriptive agreement between surfaces; not validation metrics"}


def validate_rasters(scores, grid, mask, output_dir):
    report = {}
    for name, column in SCORES.items():
        expected = score_array(scores[column], scores, grid, mask)
        with rasterio.open(raster_path(output_dir, name)) as src:
            actual = src.read(1)
            checks = {
                "dimensions": (src.height, src.width) == grid.shape and src.count == 1,
                "crs": src.crs == grid.crs and src.crs.to_epsg() == 31982,
                "resolution": src.res == (30, 30), "transform": src.transform == grid.transform,
                "extent": list(src.bounds) == grid_metadata(grid)["bounds"],
                "nodata": src.nodata == NODATA, "dtype_float32": src.dtypes == ("float32",),
                "mask": np.array_equal(src.read_masks(1) > 0, mask),
                "outside_mask_nodata": bool(np.all(actual[~mask] == NODATA)),
                "all_valid_scored": bool(np.isfinite(actual[mask]).all() and np.all(actual[mask] != NODATA)),
                "scores_0_1": bool(np.all((actual[mask] >= 0) & (actual[mask] <= 1))),
                "matches_csv_float32": np.array_equal(actual, expected),
            }
        if not all(checks.values()):
            raise ValueError(f"Raster QA failed: {name}: {checks}")
        report[name] = {"checks": checks, "statistics_float32": distribution(actual[mask])}
    return report


def save_figures(scores, sampled, grid, mask, output_dir):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, column) in zip(axes, SCORES.items()):
        ax.hist(scores[column], bins=bins, color=COLORS[name])
        ax.set(title=LABELS[name], xlabel="Susceptibility score", ylabel="Cells", xlim=(0, 1))
    fig.suptitle(f"Santa Tereza — environmental intersection (n={len(scores):,})")
    fig.savefig(output_dir / "score_histograms.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, column) in zip(axes, SCORES.items()):
        for target, label, color, style in ((1, "Positive (n=270)", "#245b89", "-"),
                                            (0, "Background (n=270)", "#b57920", "--")):
            values = np.sort(sampled.loc[sampled.target.eq(target), column].to_numpy())
            ax.step(values, np.arange(1, len(values)+1)/len(values), where="post",
                    label=label, color=color, linestyle=style)
        ax.set(title=LABELS[name], xlabel="Susceptibility score", ylabel="Cumulative fraction", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Supervised cells — fitted on these same 540 cells; descriptive distributions")
    fig.savefig(output_dir / "positive_background_distributions.png", dpi=170)
    plt.close(fig)
    bounds = grid_metadata(grid)["bounds"]
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    cmap = LinearSegmentedColormap.from_list("susceptibility", ["#f5f6ed", "#7c9d9e", "#164566"])
    diff_cmap = LinearSegmentedColormap.from_list("difference", ["#b57920", "#f7f7f7", "#245b89"])
    arrays = {name: score_array(scores[col], scores, grid, mask) for name, col in SCORES.items()}
    # Encode signed difference on the explicitly requested 0..1 visual scale.
    arrays["difference_lr_minus_rf"] = (arrays["logistic"] - arrays["random_forest"] + 1) / 2
    for name, values in arrays.items():
        difference = name == "difference_lr_minus_rf"
        fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
        im = ax.imshow(np.ma.array(values, mask=~mask), extent=extent, origin="upper",
                       interpolation="nearest", vmin=0, vmax=1, cmap=diff_cmap if difference else cmap)
        ax.set(title="LR − RF: signed difference encoded on 0–1" if difference else LABELS[name],
               xlabel="Easting (m) — EPSG:31982", ylabel="Northing (m)")
        ax.ticklabel_format(style="plain", useOffset=False)
        bar = fig.colorbar(im, ax=ax, shrink=0.8)
        bar.set_label("(LR − RF + 1) / 2; 0.5 = equal" if difference else "Landslide susceptibility score")
        if difference:
            bar.set_ticks([0, .25, .5, .75, 1])
            ax.set_title("LR − RF: (difference + 1) / 2\n0 → −1; 0.5 → equality; 1 → +1", fontsize=12)
        fig.savefig(output_dir / f"map_{name}.png", dpi=170)
        plt.close(fig)


def run(output_dir: Path = OUTPUT) -> dict:
    scores = pd.read_csv(output_dir / "cell_scores.csv", float_precision="round_trip")
    if list(scores.columns) != TRACE + list(SCORES.values()) or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Invalid cell score schema or NaN/Inf.")
    if not np.allclose(scores.ensemble_mean_score, (scores.logistic_score + scores.random_forest_score)/2,
                       atol=0, rtol=0):
        raise ValueError("Ensemble is not the arithmetic mean.")
    grid, features, mask, municipality, _ = load_environment()
    expected_grid = build_grid(grid, features, mask)
    pd.testing.assert_frame_equal(scores[TRACE], expected_grid[TRACE], check_exact=True)
    raster_qa = validate_rasters(scores, grid, mask, output_dir)
    supervised = load_training_frame()
    sampled = supervised[["row", "col", "target"]].merge(scores, on=["row", "col"],
                how="left", validate="one_to_one")
    if sampled.isna().any().any() or sampled.target.value_counts().to_dict() != {0: 270, 1: 270}:
        raise ValueError("Missing or invalid supervised score join.")
    groups = []
    for target, label in ((1, "positive"), (0, "background")):
        for name, column in SCORES.items():
            groups.append({"model": name, "group": label, **distribution(sampled.loc[sampled.target.eq(target), column])})
    pd.DataFrame(groups).to_csv(output_dir / "positive_background_score_summary.csv", index=False, float_format="%.17g")
    comparison = compare_surfaces(scores.logistic_score, scores.random_forest_score)
    pd.DataFrame([comparison]).to_csv(output_dir / "model_surface_comparison.csv", index=False, float_format="%.17g")
    prediction = json.loads((output_dir / "prediction_metadata.json").read_text())
    warnings = [NOTICE,
                "Positive/background scores are in-sample descriptions, not independent validation or causal effects.",
                "Surface correlation, mean absolute difference and RMSE measure agreement only.",
                "Difference map displays (LR - RF + 1) / 2; numerical differences retain their signed original scale."]
    for name, unknown in prediction["unknown_land_cover"].items():
        if unknown["cells"]:
            warnings.append(f"{name}: {unknown['cells']} cells with unseen land_cover codes {unknown['codes']}; one-hot block is all zero.")
    report = {"notice": NOTICE, "valid_cells": int(mask.sum()),
              "municipality_cells": int(municipality.sum()), "grid": grid_metadata(grid),
              "nodata": NODATA, "dtype": "float32", "statistics_scope": "CSV float64 scores",
              "scores": {name: distribution(scores[column]) for name, column in SCORES.items()},
              "positive_background": groups, "surface_comparison": comparison, "raster_qa": raster_qa,
              "unknown_land_cover": prediction["unknown_land_cover"], "warnings": warnings,
              "preservation": verify_preservation(output_dir)}
    save_figures(scores, sampled, grid, mask, output_dir)
    write_json(output_dir / "susceptibility_summary.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
