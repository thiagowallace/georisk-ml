"""Evaluate preliminary baselines; random holdout does not control spatial autocorrelation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs/models/baseline"
MODEL_NAMES = ("logistic", "random_forest")
THRESHOLD = 0.5
EVALUATION_NOTICE = (
    "Preliminary stratified random holdout: does not control spatial autocorrelation "
    "and is not the final project evaluation. Stage 4 will replace it with spatial validation."
)


def validate_probabilities(y, probabilities) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or not labels.size:
        raise ValueError("Labels and probabilities must be nonempty equal-length vectors.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both binary classes 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return labels.astype(int), scores


def evaluate_predictions(y, probabilities) -> dict:
    labels, scores = validate_probabilities(y, probabilities)
    predicted = (scores >= THRESHOLD).astype(int)
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    return {
        "n_samples": len(labels),
        "class_counts": {str(label): int(np.sum(labels == label)) for label in (0, 1)},
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def save_evaluation_figures(predictions: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"logistic": "#2266A0", "random_forest": "#C16B21"}
    labels = {"logistic": "Logistic Regression", "random_forest": "Random Forest"}
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
    y = predictions.target.to_numpy()

    def save(fig, name):
        fig.text(0.5, 0.025, "Preliminary random holdout; spatial autocorrelation is not controlled.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(output_dir / name, dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        plt.close(fig)

    for name in MODEL_NAMES:
        result = evaluate_predictions(y, predictions[f"{name}_probability"])
        matrix = np.asarray(result["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=20,
                        color="white" if matrix[row, col] > matrix.max() / 2 else "black")
        tick_names = ["Background (0)", "Landslide (1)"]
        ax.set_xticks([0, 1], tick_names); ax.set_yticks([0, 1], tick_names)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("Observed sample label")
        ax.set_title(f"{labels[name]}: test confusion matrix\nThreshold = {THRESHOLD}; n = {len(y)}")
        fig.colorbar(im, ax=ax, label="Sample cells")
        save(fig, f"confusion_matrix_{name}.png")

    for kind in ("roc", "precision_recall"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for name in MODEL_NAMES:
            scores = predictions[f"{name}_probability"].to_numpy()
            result = evaluate_predictions(y, scores)
            if kind == "roc":
                x, curve_y, _ = roc_curve(y, scores, pos_label=1)
                metric = result["roc_auc"]
            else:
                precision, recall, _ = precision_recall_curve(y, scores, pos_label=1)
                x, curve_y, metric = recall, precision, result["pr_auc"]
            ax.plot(x, curve_y, color=colors[name], linewidth=2,
                    linestyle="-" if name == "logistic" else "--",
                    label=f"{labels[name]} (AUC={metric:.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ":", color="gray", label="Chance reference")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
            ax.set_title("Test ROC curves — positive class: landslide")
        else:
            prevalence = float(np.mean(y == 1))
            ax.axhline(prevalence, linestyle=":", color="gray", label=f"Test sample prevalence={prevalence:.3f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title("Test Precision–Recall curves — trapezoidal PR-AUC")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.2); ax.legend(loc="best")
        save(fig, f"{kind}_curves.png")


def main() -> None:
    """Check saved test metrics and regenerate figures, without retraining."""
    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv", float_precision="round_trip")
    recorded = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    for name in MODEL_NAMES:
        actual = evaluate_predictions(predictions.target, predictions[f"{name}_probability"])
        expected = recorded["models"][name]["test"]
        for metric, value in actual.items():
            if isinstance(value, float):
                matches = np.isclose(value, expected[metric], rtol=0, atol=1e-12)
            else:
                matches = value == expected[metric]
            if not matches:
                raise ValueError(f"Saved test metrics mismatch: {name}/{metric}")
        predicted = (predictions[f"{name}_probability"].to_numpy() >= THRESHOLD).astype(int)
        if not np.array_equal(predicted, predictions[f"{name}_prediction"]):
            raise ValueError(f"Saved predictions mismatch: {name}")
    save_evaluation_figures(predictions, OUTPUT_DIR)
    print(EVALUATION_NOTICE)
    print("Saved test metrics verified; figures regenerated.")


if __name__ == "__main__":
    main()
