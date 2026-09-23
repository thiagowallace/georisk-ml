"""Prespecified 300 m sensitivity analysis using the frozen official validation folds."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.models.evaluate_spatial_validation import OUTPUT as PRINCIPAL, evaluate_spatial
from src.models.spatial_validation import SpatialConfig, buffer_training_rows, iter_spatial_splits
from src.models.train_baselines import INPUT, ROOT, METRICS, FEATURES, LOGISTIC_PARAMS, FOREST_PARAMS

BUFFER_M = 300.0
OUTPUT = ROOT / "outputs/models/spatial_validation_buffer300"


def snapshot(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def validate_output_path(output_dir, principal_dir):
    output, principal = Path(output_dir).resolve(), Path(principal_dir).resolve()
    if output == principal or principal in output.parents or output in principal.parents:
        raise ValueError("Sensitivity output must not overlap the protected principal directory.")
    return output


def compare_principal(report, metrics, principal, principal_metrics):
    summary = pd.DataFrame([
        {"model": name, "metric": metric,
         "principal_mean": principal["models"][name][metric]["mean"],
         "principal_std": principal["models"][name][metric]["std"],
         "buffer300_mean": values["mean"], "buffer300_std": values["std"],
         "buffer300_minus_principal": values["mean"] - principal["models"][name][metric]["mean"]}
        for name, model in report["models"].items() for metric, values in model.items()])
    paired = principal_metrics.merge(metrics, on=["model", "fold"], suffixes=("_principal", "_buffer300"),
                                     validate="one_to_one", how="outer", indicator=True)
    if not paired._merge.eq("both").all() or not paired.n_validation_principal.eq(paired.n_validation_buffer300).all():
        raise ValueError("Sensitivity must preserve every principal model/fold and validation count.")
    paired = paired.drop(columns="_merge")
    for metric in METRICS:
        paired[f"{metric}_delta"] = paired[f"{metric}_buffer300"] - paired[f"{metric}_principal"]
    return summary, paired


def buffer_membership(frame, assignments):
    records = []
    for fold, original, valid in iter_spatial_splits(assignments, 5):
        kept, removed, nearest = buffer_training_rows(frame, original, valid, BUFFER_M)
        table = assignments[["source_row", "sample_id", "cell_id", "target", "block_id"]].copy()
        table.insert(0, "evaluation_fold", fold)
        table["role"] = "validation"
        table.loc[kept, "role"] = "train"
        table.loc[removed, "role"] = "removed_by_buffer"
        table["distance_to_validation_m"] = 0.0
        table.loc[original, "distance_to_validation_m"] = nearest
        records.append(table)
    return pd.concat(records, ignore_index=True)


def save_comparison_figure(comparison, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, name in zip(axes, ("logistic", "random_forest")):
        rows = comparison[comparison.model.eq(name)].set_index("metric").loc[METRICS]
        x = np.arange(len(rows))
        ax.bar(x - .2, rows.principal_mean, .4, yerr=rows.principal_std, capsize=3,
               color="#777777", label="Principal: no buffer")
        ax.bar(x + .2, rows.buffer300_mean, .4, yerr=rows.buffer300_std, capsize=3,
               color="#2266A0", label="Sensitivity: 300 m buffer")
        ax.set(ylim=(0, 1.05), ylabel="Mean +/- fold SD", title=name.replace("_", " ").title())
        ax.legend(loc="lower left")
    axes[-1].set_xticks(x, METRICS, rotation=25, ha="right")
    fig.suptitle("Same validation folds; only training centers within 300 m excluded")
    fig.tight_layout()
    fig.savefig(output / "principal_vs_buffer300.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(output_dir=OUTPUT, principal_dir=PRINCIPAL, input_path=INPUT):
    principal_dir, input_path = Path(principal_dir), Path(input_path)
    output_dir = validate_output_path(output_dir, principal_dir)
    protected_paths = [input_path, *sorted(p for p in principal_dir.rglob("*") if p.is_file())]
    before = snapshot(protected_paths)
    principal = json.loads((principal_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    if principal["source_sha256"] != before[str(input_path)]:
        raise ValueError("Dataset does not match the principal analysis source.")
    spatial = principal["spatial"]
    if any(spatial[k] != v for k, v in {"block_size_m": 1500, "n_splits": 5,
                                       "random_state": 42, "buffer_m": 0}.items()):
        raise ValueError("Principal must be the official 1500 m / 5 folds / seed 42 / no buffer analysis.")
    if (principal["features"] != FEATURES or principal["threshold"] != .5
            or principal["parameters"] != {"logistic": LOGISTIC_PARAMS, "random_forest": FOREST_PARAMS}):
        raise ValueError("Principal model contract differs from the preserved baselines.")
    frame = pd.read_csv(input_path, float_precision="round_trip")
    assignments_path = principal_dir / "fold_assignments.csv"
    assignments = pd.read_csv(assignments_path, float_precision="round_trip")
    principal_metrics = pd.read_csv(principal_dir / "fold_metrics.csv", float_precision="round_trip")
    result = evaluate_spatial(frame, SpatialConfig(), frozen_assignments=assignments,
                              frozen_metadata=spatial, buffer_m=BUFFER_M)
    report, assignments, metrics, matrices, predictions = result
    if report["preprocessing"] != principal["preprocessing"]:
        raise ValueError("Preprocessing contract changed.")
    comparison, fold_comparison = compare_principal(report, metrics, principal, principal_metrics)
    membership = buffer_membership(frame, assignments)
    report.update({"analysis_role": "Prespecified sensitivity only; official principal results remain unchanged",
                   "source": str(input_path), "source_sha256": before[str(input_path)],
                   "principal_source": str(principal_dir), "protected_source_hashes": before,
                   "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "runtime": {"python": platform.python_version(), **{
                       p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")}}})
    if snapshot(protected_paths) != before:
        raise RuntimeError("A protected source changed during sensitivity evaluation.")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Byte-for-byte copy makes frozen validation membership directly auditable.
    (output_dir / "fold_assignments.csv").write_bytes(assignments_path.read_bytes())
    for name, table in (("fold_metrics", metrics), ("confusion_matrices", matrices),
                        ("oof_predictions", predictions), ("fold_class_counts", pd.DataFrame(report["fold_counts"])),
                        ("buffer_membership", membership), ("principal_vs_buffer300", comparison),
                        ("fold_comparison", fold_comparison)):
        table.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.17g")
    save_comparison_figure(comparison, output_dir)
    report["protected_sources_unchanged"] = snapshot(protected_paths) == before
    if not report["protected_sources_unchanged"]:
        raise RuntimeError("A protected source changed while exporting sensitivity results.")
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(report["fold_counts"]).to_string(index=False))
    print(comparison.to_string(index=False))
    return report


if __name__ == "__main__":
    run()
