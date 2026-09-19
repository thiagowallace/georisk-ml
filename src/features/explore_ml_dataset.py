"""Descriptive EDA and statistical QA of the cell-based ML dataset; no modeling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/santa_tereza_ml_dataset.csv"
OUTPUT = ROOT / "outputs/eda"
GROUPS = ("landslide", "background")
CONTINUOUS = ("elevation", "slope", "aspect_sin", "aspect_cos", "plan_curvature",
              "profile_curvature", "ndvi_pre_event")
TRACE = ("sample_id", "cell_id", "row", "col", "x", "y", "landslide_ids", "landslide_count")
EXPECTED = (*TRACE, "target", "sample_type", *CONTINUOUS, "aspect", "land_cover")
LABELS = {
    "elevation": "Elevation (m)", "slope": "Slope (degrees)",
    "aspect": "Aspect (degrees; circular)", "aspect_sin": "Aspect sine",
    "aspect_cos": "Aspect cosine", "plan_curvature": "Plan curvature (1/m)",
    "profile_curvature": "Profile curvature (1/m)", "ndvi_pre_event": "Pre-event NDVI",
}
COLORS = {"landslide": "#2266A0", "background": "#C16B21"}


def validate_dataset(frame: pd.DataFrame, expected_rows: int = 540) -> dict:
    missing = sorted(set(EXPECTED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(frame)}.")
    nulls = frame.isna().sum()
    numeric = [*CONTINUOUS, "aspect", "land_cover", "target", "cell_id", "row", "col",
               "x", "y", "landslide_count"]
    values = frame[numeric].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    infinite = int(np.isinf(values).sum())
    duplicates = {"rows": int(frame.duplicated().sum()),
                  "cell_id": int(frame.cell_id.duplicated().sum()),
                  "sample_id": int(frame.sample_id.duplicated().sum())}
    if nulls.any() or not np.isfinite(values).all() or any(duplicates.values()):
        raise ValueError(f"QA failed: missing={int(nulls.sum())}, infinite={infinite}, duplicates={duplicates}")
    counts = frame.sample_type.value_counts().to_dict()
    if counts != {"landslide": expected_rows // 2, "background": expected_rows // 2}:
        raise ValueError(f"Classes must be balanced: {counts}")
    if not frame.target.eq(frame.sample_type.map({"landslide": 1, "background": 0})).all():
        raise ValueError("target and sample_type are inconsistent.")
    if (frame[list(CONTINUOUS)].eq(-9999).any().any() or frame.land_cover.le(0).any()
            or not np.equal(frame.land_cover, np.floor(frame.land_cover)).all()):
        raise ValueError("NoData or invalid land-cover code.")
    if not frame.aspect.between(0, 360, inclusive="left").all():
        raise ValueError("Aspect outside [0, 360).")
    for name, transform in (("aspect_sin", np.sin), ("aspect_cos", np.cos)):
        if (not frame[name].between(-1, 1).all()
                or not np.allclose(frame[name], transform(np.deg2rad(frame.aspect)), atol=1e-12, rtol=0)):
            raise ValueError(f"Invalid circular component: {name}")
    if not frame.ndvi_pre_event.between(-1, 1).all() or not frame.slope.between(0, 90).all():
        raise ValueError("NDVI or slope outside physical bounds.")
    return {"rows": len(frame), "missing_by_column": nulls.astype(int).to_dict(),
            "infinite_values": infinite, "duplicates": duplicates, "status": "passed",
            "identical_feature_rows": int(frame[[*CONTINUOUS, "land_cover"]].duplicated().sum())}


def circular_statistics(degrees: np.ndarray) -> dict:
    radians = np.deg2rad(np.asarray(degrees, dtype=float))
    sine, cosine = np.sin(radians).mean(), np.cos(radians).mean()
    resultant = float(np.clip(np.hypot(sine, cosine), 0, 1))
    direction = float(np.degrees(np.arctan2(sine, cosine)) % 360) if resultant > 1e-12 else None
    if direction is not None and np.isclose(direction, 360, atol=1e-10, rtol=0):
        direction = 0.0
    return {"mean_direction_deg": direction, "resultant_length": resultant,
            "circular_variance": 1 - resultant, "count": len(radians)}


def iqr_flags(values: pd.Series) -> tuple[pd.Series, float, float]:
    q1, q3 = values.quantile([0.25, 0.75])
    lower, upper = float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))
    return (values < lower) | (values > upper), lower, upper


def analyze(frame: pd.DataFrame, expected_rows: int = 540) -> tuple[dict[str, pd.DataFrame], dict]:
    qa = validate_dataset(frame, expected_rows)
    frame = frame.sort_values("cell_id", kind="stable").reset_index(drop=True)
    distribution = frame.groupby(["target", "sample_type"], sort=True).size().rename("count").reset_index()
    distribution["percent"] = distribution["count"] / len(frame) * 100
    summaries, circular, outlier_summary, outliers, comparisons = [], [], [], [], []
    for group in GROUPS:
        subset = frame.loc[frame.sample_type.eq(group)]
        circular.append({"sample_type": group, **circular_statistics(subset.aspect.to_numpy())})
        for name in CONTINUOUS:
            values = subset[name]
            q = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            summaries.append({"sample_type": group, "variable": name, "count": len(values),
                              "mean": values.mean(), "std": values.std(ddof=1), "min": values.min(),
                              "p05": q.loc[0.05], "q25": q.loc[0.25], "median": q.loc[0.5],
                              "q75": q.loc[0.75], "p95": q.loc[0.95], "max": values.max()})
            flags, lower, upper = iqr_flags(values)
            outlier_summary.append({"sample_type": group, "variable": name, "count": int(flags.sum()),
                                    "percent": flags.mean() * 100, "lower_fence": lower, "upper_fence": upper})
            for index in flags.index[flags]:
                outliers.append({"sample_id": frame.at[index, "sample_id"],
                                 "cell_id": int(frame.at[index, "cell_id"]), "sample_type": group,
                                 "variable": name, "value": frame.at[index, name],
                                 "lower_fence": lower, "upper_fence": upper})
    for name in CONTINUOUS:
        a = frame.loc[frame.sample_type.eq("landslide"), name].to_numpy()
        b = frame.loc[frame.sample_type.eq("background"), name].to_numpy()
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        ranks = rankdata(np.concatenate([a, b]), method="average")
        u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2
        comparisons.append({"variable": name, "mean_difference": a.mean() - b.mean(),
                            "median_difference": np.median(a) - np.median(b),
                            "standardized_mean_difference": (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan,
                            "rank_biserial": 2 * u / (len(a) * len(b)) - 1})
    tables = {"class_distribution": distribution, "descriptive_by_class": pd.DataFrame(summaries),
              "aspect_circular": pd.DataFrame(circular), "continuous_comparison": pd.DataFrame(comparisons),
              "outlier_summary": pd.DataFrame(outlier_summary),
              "outlier_records": pd.DataFrame(outliers, columns=["sample_id", "cell_id", "sample_type",
                                                                 "variable", "value", "lower_fence", "upper_fence"])}
    codes = sorted(int(code) for code in frame.land_cover.unique())
    cover = frame.groupby(["sample_type", "land_cover"]).size().reindex(
        pd.MultiIndex.from_product([GROUPS, codes], names=["sample_type", "land_cover"]), fill_value=0,
    ).rename("count").reset_index()
    cover["percent_within_class"] = cover["count"] / cover.groupby("sample_type")["count"].transform("sum") * 100
    tables["land_cover_by_class"] = cover
    high = []
    for scope in ("pooled", *GROUPS):
        subset = frame if scope == "pooled" else frame.loc[frame.sample_type.eq(scope)]
        for method in ("pearson", "spearman"):
            matrix = subset[list(CONTINUOUS)].corr(method=method)
            matrix.index.name = "variable"
            tables[f"correlation_{method}_{scope}"] = matrix.reset_index()
            for i, first in enumerate(CONTINUOUS):
                for second in CONTINUOUS[i + 1:]:
                    value = matrix.loc[first, second]
                    if np.isfinite(value) and abs(value) >= 0.8:
                        high.append({"scope": scope, "method": method, "first": first, "second": second, "coefficient": value})
    tables["high_correlations"] = pd.DataFrame(high, columns=["scope", "method", "first", "second", "coefficient"])
    qa.update({"land_cover_codes": codes, "outlier_unique_cells": len({r["cell_id"] for r in outliers}),
               "correlation_variables": list(CONTINUOUS), "high_correlation_threshold": 0.8,
               "outlier_rule": "Within-class Tukey fences: Q1 - 1.5 IQR and Q3 + 1.5 IQR; no removal",
               "caveats": ["Background represents pseudoabsence, not confirmed absence.",
                           "Balanced sample proportions do not estimate territorial prevalence.",
                           "Descriptive differences and correlations are not causal or model importance.",
                           "Raw aspect is circular: excluded from linear correlations and Tukey flags.",
                           "Undefined correlations/effect sizes are null, not zero; no significance tests."]})
    return tables, qa


def render_figures(frame: pd.DataFrame, tables: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    destination.mkdir(parents=True, exist_ok=True)
    legend = [Patch(facecolor=COLORS[g], label=f"{g} (n={int(frame.sample_type.eq(g).sum())})",
                    hatch="//" if g == "background" else "") for g in GROUPS]

    def save(fig, name, title, note):
        fig.suptitle(title, fontsize=16, y=0.995)
        fig.text(0.5, 0.01, note, ha="center", fontsize=10)
        fig.savefig(destination / name, dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        plt.close(fig)

    variables = (*CONTINUOUS, "aspect")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, name in zip(axes.flat, variables):
        bins = np.linspace(0, 360, 19) if name == "aspect" else np.histogram_bin_edges(frame[name], bins=20)
        for group in GROUPS:
            values = frame.loc[frame.sample_type.eq(group), name]
            ax.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                    histtype="step", linewidth=2, color=COLORS[group],
                    linestyle="--" if group == "background" else "-")
        ax.set_title(LABELS[name]); ax.set_ylabel("% within class"); ax.grid(axis="y", alpha=0.2)
        if "curvature" in name:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    axes.flat[-1].axis("off"); axes.flat[-1].legend(handles=legend, loc="center")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save(fig, "histograms.png", "Environmental distributions by sample class",
         "Shared bins per variable; full observed ranges. Raw aspect has an arbitrary seam at 0/360 degrees.")

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, name in zip(axes.flat, CONTINUOUS):
        result = ax.boxplot([frame.loc[frame.sample_type.eq(g), name] for g in GROUPS],
                            patch_artist=True, medianprops={"color": "black"},
                            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none"})
        for patch, group in zip(result["boxes"], GROUPS):
            patch.set_facecolor(COLORS[group]); patch.set_alpha(0.6)
            if group == "background": patch.set_hatch("//")
        ax.set_xticks([1, 2], GROUPS); ax.set_title(LABELS[name]); ax.grid(axis="y", alpha=0.2)
        if "curvature" in name: ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes.flat[-1].axis("off"); axes.flat[-1].legend(handles=legend, loc="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save(fig, "boxplots.png", "Continuous features: landslide versus background",
         "Median, quartiles and 1.5 IQR whiskers; all outliers retained. Circular aspect is shown separately.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
    names = ["elevation", "slope", "aspect sin", "aspect cos", "plan curv.", "profile curv.", "NDVI"]
    for ax, method in zip(axes, ("pearson", "spearman")):
        matrix = tables[f"correlation_{method}_pooled"].set_index("variable").to_numpy()
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(7), names, rotation=45, ha="right"); ax.set_yticks(range(7), names)
        ax.set_title(method.capitalize(), pad=12)
        for row in range(7):
            for col in range(7):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center",
                        color="white" if abs(value) > 0.6 else "black")
    fig.colorbar(im, ax=axes, shrink=0.65, label="Correlation coefficient")
    save(fig, "correlations.png", "Pooled correlations: continuous features only",
         "Raw aspect, MapBiomas codes, labels and traceability fields excluded; within-class matrices supplied as CSV.")

    fig, ax = plt.subplots(figsize=(11, 6))
    cover = tables["land_cover_by_class"]
    codes = sorted(cover.land_cover.unique()); positions = np.arange(len(codes))
    for i, group in enumerate(GROUPS):
        subset = cover.loc[cover.sample_type.eq(group)].set_index("land_cover").loc[codes]
        bars = ax.bar(positions + (i - 0.5) * 0.38, subset.percent_within_class, width=0.38,
                      color=COLORS[group], label=group, hatch="//" if i else "")
        ax.bar_label(bars, labels=[str(n) for n in subset["count"]], padding=3, fontsize=9)
    ax.set_xticks(positions, [str(c) for c in codes]); ax.set_xlabel("Original MapBiomas code (categorical)")
    ax.set_ylabel("% within class"); ax.legend(); ax.set_ylim(0, max(cover.percent_within_class) * 1.17)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, "land_cover.png", "MapBiomas 2023 by sample class", "Bar labels show counts; percentages use each class as denominator.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), subplot_kw={"projection": "polar"})
    bins = np.linspace(0, 2 * np.pi, 17)
    for ax, group in zip(axes, GROUPS):
        values = frame.loc[frame.sample_type.eq(group), "aspect"]
        counts, _ = np.histogram(np.deg2rad(values), bins=bins)
        ax.bar((bins[1:] + bins[:-1]) / 2, counts / len(values) * 100, width=np.diff(bins),
               color=COLORS[group], edgecolor="white", hatch="//" if group == "background" else "")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_title(group, pad=25)
        ax.set_ylim(0, 25)
    max_height = max(ax.patches[i].get_height() for ax in axes for i in range(16))
    for ax in axes: ax.set_ylim(0, max(10, np.ceil(max_height / 5) * 5))
    fig.subplots_adjust(top=0.79, bottom=0.14, wspace=0.4)
    save(fig, "aspect_circular.png", "Aspect orientation by sample class", "North = 0 degrees; clockwise azimuth; radial axis = % within class; shared radial scale.")

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = [int(frame.sample_type.eq(g).sum()) for g in GROUPS]
    bars = ax.bar(GROUPS, counts, color=[COLORS[g] for g in GROUPS]); bars[1].set_hatch("//")
    ax.bar_label(bars, padding=4); ax.set_ylabel("Sample cells"); ax.set_ylim(0, max(counts) * 1.2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, "class_distribution.png", "Sample class balance", "landslide: target=1; background: target=0 (pseudoabsence).")


def run(input_path: Path = INPUT, output_dir: Path = OUTPUT) -> dict:
    raw = input_path.read_bytes()
    frame = pd.read_csv(input_path, float_precision="round_trip")
    tables, qa = analyze(frame)
    qa["input_sha256"] = hashlib.sha256(raw).hexdigest()
    qa["source"] = "data/processed/santa_tereza_ml_dataset.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.17g")
    summary = {"qa": qa, "tables": {name: json.loads(table.to_json(orient="records", double_precision=15))
                                     for name, table in tables.items()}}
    (output_dir / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    render_figures(frame, tables, output_dir)
    if input_path.read_bytes() != raw:
        raise RuntimeError("Input changed during analysis; outputs must be regenerated.")
    print(json.dumps(qa, indent=2))
    return summary


if __name__ == "__main__":
    run()
