"""Run Stage 4, export out-of-fold evidence and compare the frozen random baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.models.evaluate_baselines import evaluate_predictions, THRESHOLD
from src.models.spatial_validation import (
    SpatialConfig, build_spatial_folds, iter_spatial_splits, buffer_training_rows,
)
from src.models.train_baselines import (
    INPUT, ROOT, FEATURES, FORBIDDEN, CONTINUOUS, LOGISTIC_PARAMS, FOREST_PARAMS,
    METRICS, fit_baselines, positive_probabilities, select_features,
)

OUTPUT = ROOT / "outputs/models/spatial_validation"
BASELINE = ROOT / "outputs/models/baseline/metrics.json"


def evaluate_spatial(frame, config=SpatialConfig(), *, frozen_assignments=None,
                     frozen_metadata=None, buffer_m=0.0):
    if frozen_assignments is None:
        assignments, spatial = build_spatial_folds(frame, config)
    else:
        # Validate the supplied frozen partition against its source and geometry.
        expected, metadata = build_spatial_folds(frame, config)
        pd.testing.assert_frame_equal(frozen_assignments, expected)
        if frozen_metadata != metadata:
            raise ValueError("Frozen spatial metadata differs from the declared configuration.")
        assignments, spatial = frozen_assignments.copy(), dict(frozen_metadata)
    X, y = select_features(frame)
    records, matrices, diagnostics, counts = [], [], [], []
    predictions = assignments.copy()
    # Check all folds after exclusion before fitting any model.
    splits = []
    for fold, original_train, valid in iter_spatial_splits(assignments, config.n_splits):
        train, removed, _ = buffer_training_rows(frame, original_train, valid, buffer_m)
        splits.append((fold, train, valid, removed))
    for fold, train, valid, removed in splits:
        models, fit_diagnostics = fit_baselines(X.iloc[train], y.iloc[train])
        distances = cdist(frame.iloc[train][["x", "y"]], frame.iloc[valid][["x", "y"]])
        counts.append({"fold": fold, "n_train": len(train), "n_validation": len(valid),
                       "train_background": int(y.iloc[train].eq(0).sum()),
                       "train_landslide": int(y.iloc[train].eq(1).sum()),
                       "background": int(y.iloc[valid].eq(0).sum()),
                       "landslide": int(y.iloc[valid].eq(1).sum()),
                       "validation_blocks": int(assignments.iloc[valid].block_id.nunique()),
                       "min_train_validation_distance_m": float(distances.min())})
        if buffer_m > 0:
            counts[-1].update({"n_train_before_buffer": len(train) + len(removed),
                               "n_removed": len(removed),
                               "removed_background": int(y.iloc[removed].eq(0).sum()),
                               "removed_landslide": int(y.iloc[removed].eq(1).sum())})
        for name, model in models.items():
            probabilities = positive_probabilities(model, X.iloc[valid])
            scores = evaluate_predictions(y.iloc[valid], probabilities)
            records.append({"model": name, "fold": fold, "n_validation": len(valid),
                            **{metric: scores[metric] for metric in METRICS}})
            tn, fp, fn, tp = np.array(scores["confusion_matrix"]).ravel()
            matrices.append({"model": name, "fold": fold, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
            predictions.loc[valid, f"{name}_probability"] = probabilities
            predictions.loc[valid, f"{name}_prediction"] = (probabilities >= THRESHOLD).astype(int)
            prep = model.named_steps["preprocessor"]
            known = prep.named_transformers_["land_cover"].categories_[0].tolist()
            diagnostic = {"model": name, "fold": fold, **fit_diagnostics[name],
                          "fit_n_samples": len(train), "training_land_cover_categories": known,
                          "validation_unknown_land_cover_categories": sorted(set(X.iloc[valid].land_cover) - set(known)),
                          "validation_unknown_land_cover_count": int((~X.iloc[valid].land_cover.isin(known)).sum())}
            if name == "logistic":
                scaler = prep.named_transformers_["continuous"]
                diagnostic.update({"scaler_n_samples_seen": int(scaler.n_samples_seen_),
                                   "scaler_mean": scaler.mean_.tolist()})
            diagnostics.append(diagnostic)
    metrics = pd.DataFrame(records)
    summary = {name: {metric: {"mean": float(group[metric].mean()),
                              "std": float(group[metric].std(ddof=1)),
                              "min": float(group[metric].min()), "max": float(group[metric].max())}
                      for metric in METRICS} for name, group in metrics.groupby("model")}
    report = {"spatial": spatial, "features": FEATURES, "excluded_predictors": FORBIDDEN,
              "threshold": THRESHOLD, "positive_class": 1,
              "parameters": {"logistic": LOGISTIC_PARAMS, "random_forest": FOREST_PARAMS},
              "aggregation": "Unweighted fold mean; sample standard deviation (ddof=1), not a confidence interval",
              "metric_definitions": {"pr_auc": "Trapezoidal area under precision-recall curve",
                                     "average_precision": "Non-interpolated average precision",
                                     "confusion_matrix": "tn,fp,fn,tp; positive class landslide=1"},
              "preprocessing": {"fit_scope": "training rows of each fold only",
                                "logistic": "StandardScaler continuous + OneHotEncoder(handle_unknown=ignore)",
                                "random_forest": "continuous passthrough + OneHotEncoder(handle_unknown=ignore)"},
              "fold_counts": counts, "models": summary, "diagnostics": diagnostics,
              "warnings": ["Blocks may be adjacent: no buffer and no guaranteed autocorrelation independence.",
                           "1500 m is a prespecified operational scale, not an estimated autocorrelation range.",
                           "Folds may consist of disconnected blocks; this is not leave-region-out validation.",
                           "Random holdout and spatial CV differ in test population and training size; comparison is descriptive.",
                           "Balanced case-control sampling does not represent municipal prevalence; PR metrics are sample-dependent.",
                           "Same-event landslide cells may cross block boundaries; event independence is not guaranteed."]}
    if buffer_m > 0:
        report["spatial"]["buffer_m"] = float(buffer_m)
        report["spatial"]["buffer_rule"] = "Remove training centers with distance < buffer_m to any validation center"
        report["warnings"][0] = "Buffer enforces center separation, not complete spatial or event independence."
        report["warnings"][3] = "Buffered comparison fixes validation rows but changes training size and composition."
    return report, assignments, metrics, pd.DataFrame(matrices), predictions


def compare_baseline(report, baseline):
    if baseline["features"] != FEATURES or baseline["parameters"] != report["parameters"]:
        raise ValueError("Saved baseline feature or hyperparameter contract differs.")
    return pd.DataFrame([
        {"model": name, "metric": metric, "baseline_random_test": baseline["models"][name]["test"][metric],
         "spatial_mean": values["mean"], "spatial_std": values["std"],
         "spatial_minus_baseline": values["mean"] - baseline["models"][name]["test"][metric]}
        for name, model in report["models"].items() for metric, values in model.items()])


def save_figures(assignments, metrics, comparison, spatial, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = {"logistic": "Logistic Regression", "random_forest": "Random Forest"}
    colors = {"logistic": "#2266A0", "random_forest": "#C16B21"}
    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    palette = plt.get_cmap("tab20", spatial["n_splits"])
    size = spatial["block_size_m"]
    for row in assignments.drop_duplicates("block_id").itertuples():
        ax.add_patch(Rectangle((spatial["origin_x"] + row.block_x * size,
                                spatial["origin_y"] + row.block_y * size), size, size,
                               facecolor=palette(row.fold - 1), edgecolor="gray", alpha=.22))
    for fold, group in assignments.groupby("fold"):
        ax.scatter(group.x, group.y, s=15, color=palette(fold - 1), label=f"Fold {fold} (n={len(group)})")
    ax.set_aspect("equal")
    ax.set(xlabel="Easting (m), EPSG:31982", ylabel="Northing (m), EPSG:31982",
           title=f"Spatial validation: {size:g} x {size:g} m blocks\nOccupied blocks; no buffer")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend(fontsize=9)
    save(fig, "spatial_folds.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, (name, title) in zip(axes, names.items()):
        values = comparison[comparison.model.eq(name)].set_index("metric").loc[METRICS]
        positions = np.arange(len(METRICS))
        ax.bar(positions - .2, values.baseline_random_test, .4, label="Random test (n=135)", color="#999999")
        ax.bar(positions + .2, values.spatial_mean, .4, yerr=values.spatial_std,
               capsize=3, label="Spatial mean +/- SD (folds)", color=colors[name])
        ax.set(ylim=(0, 1.05), ylabel="Score", title=title)
        ax.legend(loc="lower left")
    axes[-1].set_xticks(positions, METRICS, rotation=25, ha="right")
    save(fig, "metric_comparison.png")
    for metric in ("roc_auc", "recall"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (name, title) in enumerate(names.items()):
            group = metrics[metrics.model.eq(name)]
            ax.bar(group.fold + (i - .5) * .35, group[metric], width=.35, label=title, color=colors[name])
        ax.set(xticks=sorted(metrics.fold.unique()), ylim=(0, 1), xlabel="Validation fold",
               ylabel=metric, title=f"Spatial validation: {metric} by fold")
        ax.legend(loc="lower left")
        save(fig, f"{metric}_by_fold.png")


def run(input_path=INPUT, output_dir=OUTPUT, baseline_path=BASELINE, config=SpatialConfig()):
    input_path, output_dir, baseline_path = map(Path, (input_path, output_dir, baseline_path))
    original = input_path.read_bytes()
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes)
    digest = hashlib.sha256(original).hexdigest()
    if baseline.get("source_sha256") != digest:
        raise ValueError("Dataset hash differs from the saved baseline source.")
    frame = pd.read_csv(input_path, float_precision="round_trip")
    if len(frame) != 540 or frame.target.value_counts().to_dict() != {0: 270, 1: 270}:
        raise ValueError("Official Stage 4 dataset requires 540 cells, 270 per class.")
    report, assignments, metrics, matrices, predictions = evaluate_spatial(frame, config)
    comparison = compare_baseline(report, baseline)
    report.update({"source": str(input_path), "source_sha256": digest,
                   "baseline_source": str(baseline_path),
                   "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
                   "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "runtime": {"python": platform.python_version(), **{
                       p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")}}})
    if input_path.read_bytes() != original or baseline_path.read_bytes() != baseline_bytes:
        raise RuntimeError("Source changed during evaluation; results not exported.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in (("fold_assignments", assignments), ("fold_metrics", metrics),
                            ("confusion_matrices", matrices), ("baseline_vs_spatial", comparison),
                            ("oof_predictions", predictions), ("fold_class_counts", pd.DataFrame(report["fold_counts"]))):
        table.to_csv(output_dir / f"{filename}.csv", index=False, float_format="%.17g")
    save_figures(assignments, metrics, comparison, report["spatial"], output_dir)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"spatial": report["spatial"], "fold_counts": report["fold_counts"],
                      "models": report["models"], "warnings": report["warnings"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--block-size-m", type=float, default=1500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.baseline,
        SpatialConfig(args.n_splits, args.block_size_m, args.random_state))
