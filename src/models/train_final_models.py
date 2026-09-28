"""Fit the frozen Stage 3/4 pipelines on all 540 supervised cells."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import pandas as pd

from src.models.train_baselines import (
    ROOT, INPUT, FEATURES, FORBIDDEN, LOGISTIC_PARAMS, FOREST_PARAMS,
    select_features, fit_baselines,
)

OUTPUT = ROOT / "outputs/models/final"
NOTICE = (
    "Landslide susceptibility score from balanced landslide/background sampling; "
    "not calibrated to territorial occurrence frequency. Background is pseudoabsence. "
    "ensemble_mean_score is an exploratory arithmetic mean, not separately validated."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def verify_preservation(output_dir: Path = OUTPUT) -> dict:
    path = output_dir / "preservation_manifest.json"
    if not path.exists():
        raise ValueError("Create a preservation manifest before starting Stage 5.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    changed = [name for name, digest in manifest.items()
               if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f"Preserved files changed: {changed}")
    return {"checked_files": len(manifest), "changed_files": changed, "passed": True}


def ensure_manifest(output_dir: Path) -> None:
    """Freeze existing inputs/results once; never replace an earlier manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "preservation_manifest.json"
    if not path.exists():
        manifest = {}
        for folder in ("data", "outputs", "src", "tests", "docs", "scripts"):
            for item in sorted((ROOT / folder).rglob("*")):
                if (item.is_file() and "__pycache__" not in item.parts
                        and output_dir.resolve() not in item.resolve().parents):
                    manifest[item.relative_to(ROOT).as_posix()] = sha256(item)
        write_json(path, manifest)
    verify_preservation(output_dir)


def load_training_frame() -> pd.DataFrame:
    frame = pd.read_csv(INPUT, float_precision="round_trip")
    if len(frame) != 540 or frame.target.value_counts().to_dict() != {0: 270, 1: 270}:
        raise ValueError("Final fit requires all 540 samples, 270 per class.")
    if frame.cell_id.isna().any() or not frame.cell_id.is_unique or frame.duplicated(["row", "col"]).any():
        raise ValueError("Supervised cells must be unique and non-null.")
    select_features(frame)
    return frame


def run(output_dir: Path = OUTPUT) -> dict:
    ensure_manifest(output_dir)
    frame = load_training_frame()
    parameters = {"logistic": LOGISTIC_PARAMS, "random_forest": FOREST_PARAMS}
    source_digest = sha256(INPUT)
    references = {}
    for relative in ("baseline/metrics.json", "spatial_validation/summary_metrics.json",
                     "spatial_validation_buffer300/summary_metrics.json"):
        path = ROOT / "outputs/models" / relative
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if (frozen["features"] != FEATURES or frozen["parameters"] != parameters
                or frozen["source_sha256"] != source_digest):
            raise ValueError(f"Frozen Stage 3/4 contract differs: {relative}")
        references[relative] = sha256(path)
    X, y = select_features(frame)
    models, diagnostics = fit_baselines(X, y)
    model_files = {}
    for name, model in models.items():
        path = output_dir / f"{name}.joblib"
        joblib.dump(model, path)
        model_files[path.name] = sha256(path)
        prep = model.named_steps["preprocessor"]
        diagnostics[name]["land_cover_categories"] = [int(v) for v in
            prep.named_transformers_["land_cover"].categories_[0]]
        diagnostics[name]["transformed_features"] = prep.get_feature_names_out().tolist()
    report = {
        "notice": NOTICE, "features": FEATURES, "excluded_predictors": FORBIDDEN,
        "source": INPUT.relative_to(ROOT).as_posix(), "source_sha256": source_digest,
        "n_training_samples": len(frame), "class_counts": {"positive": 270, "background": 270},
        "parameters": parameters, "tuning": False, "positive_class": 1,
        "preprocessing": {"implementation": "src.models.train_baselines.make_pipelines",
            "fit_scope": "all 540 supervised samples", "logistic_continuous": "StandardScaler",
            "random_forest_continuous": "passthrough",
            "land_cover": "OneHotEncoder(handle_unknown=ignore, sparse_output=False)",
            "aspect": "sin/cos of degrees converted to float64 radians"},
        "diagnostics": diagnostics, "frozen_reference_sha256": references,
        "model_sha256": model_files,
        "runtime": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ("scikit-learn", "numpy", "pandas", "scipy", "rasterio", "joblib", "matplotlib")}},
        "preservation": verify_preservation(output_dir),
    }
    write_json(output_dir / "final_model_metadata.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
