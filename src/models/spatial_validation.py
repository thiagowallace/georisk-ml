"""Deterministic spatial block cross-validation; coordinates never enter models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedGroupKFold

from src.models.train_baselines import select_features


@dataclass(frozen=True)
class SpatialConfig:
    n_splits: int = 5
    block_size_m: float = 1500.0
    random_state: int = 42

    def validate(self):
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, int) or self.n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2.")
        if (not np.isfinite(self.block_size_m) or self.block_size_m < 30
                or not np.isclose(self.block_size_m / 30, round(self.block_size_m / 30))):
            raise ValueError("block_size_m must be a finite positive multiple of 30 m.")


def build_spatial_folds(frame: pd.DataFrame, config: SpatialConfig = SpatialConfig()):
    """Tile 30 m cells with half-open squares, then stratify whole occupied blocks.

    Origin is the southwest edge of the sampled grid. No seed search or retry is
    performed: infeasible partitions fail explicitly before fitting any model.
    Returned indices and source_row are positional, independent of input labels.
    """
    config.validate()
    _, target = select_features(frame)
    if not {"x", "y", "cell_id", "sample_id"}.issubset(frame.columns):
        raise ValueError("Spatial validation requires x, y, cell_id and sample_id.")
    xy = frame[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all() or frame.duplicated(["x", "y"]).any():
        raise ValueError("Coordinates must be finite and unique per cell.")
    for column in ("cell_id", "sample_id"):
        if frame[column].isna().any() or not frame[column].is_unique:
            raise ValueError(f"{column} must be unique and non-null.")
    origin = xy.min(axis=0) - 15.0
    grid_positions = (xy - (origin + 15.0)) / 30.0
    if not np.allclose(grid_positions, np.rint(grid_positions), atol=1e-7, rtol=0):
        raise ValueError("Coordinates must be centers on the documented 30 m grid.")
    # Integer cell offsets avoid floating-point ambiguity at block boundaries.
    cells_per_side = int(round(config.block_size_m / 30))
    block_xy = np.rint(grid_positions).astype(np.int64) // cells_per_side
    assignments = frame[["sample_id", "cell_id", "x", "y", "target"]].reset_index(drop=True).copy()
    assignments.insert(0, "source_row", np.arange(len(frame)))
    assignments["block_x"], assignments["block_y"] = block_xy.T
    assignments["block_id"] = [f"{x}:{y}" for x, y in block_xy]
    assignments["fold"] = -1
    if assignments.block_id.nunique() < config.n_splits:
        raise ValueError("Not enough occupied spatial blocks for n_splits.")
    if target.value_counts().min() < config.n_splits:
        raise ValueError("Not enough samples of each class for n_splits.")
    splitter = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True,
                                   random_state=config.random_state)
    for fold, (_, validation) in enumerate(splitter.split(
            np.zeros((len(frame), 1)), target.to_numpy(), assignments.block_id), start=1):
        assignments.loc[validation, "fold"] = fold
    list(iter_spatial_splits(assignments, config.n_splits))
    metadata = {**asdict(config), "method": "StratifiedGroupKFold on square spatial blocks",
                "crs": "EPSG:31982", "cell_size_m": 30,
                "cells_per_block_side": cells_per_side,
                "origin_x": float(origin[0]), "origin_y": float(origin[1]),
                "occupied_blocks": int(assignments.block_id.nunique()),
                "grid_columns": int(block_xy[:, 0].max() + 1),
                "grid_rows": int(block_xy[:, 1].max() + 1),
                "buffer_m": 0, "boundary_rule": "half-open [min, max); aligned to cell edges"}
    return assignments, metadata


def iter_spatial_splits(assignments: pd.DataFrame, n_splits: int):
    """Validate the full partition before yielding train/validation row positions."""
    if (assignments.empty or not assignments.source_row.is_unique
            or set(assignments.source_row) != set(range(len(assignments)))
            or not assignments.cell_id.is_unique
            or assignments.duplicated(["x", "y"]).any()
            or assignments[["block_id", "fold", "target"]].isna().any().any()):
        raise ValueError("Assignments must contain every unique cell exactly once.")
    if set(assignments.fold) != set(range(1, n_splits + 1)):
        raise ValueError("Invalid or missing folds.")
    if assignments.groupby("block_id").fold.nunique().ne(1).any():
        raise ValueError("A spatial block occurs in multiple folds.")
    splits = []
    for fold in range(1, n_splits + 1):
        validation = assignments[assignments.fold.eq(fold)]
        train = assignments[assignments.fold.ne(fold)]
        if set(validation.target) != {0, 1} or set(train.target) != {0, 1}:
            raise ValueError(f"Fold {fold} must contain both classes in training and validation; "
                             "revise the declared spatial configuration, without performance-based selection.")
        if set(train.block_id) & set(validation.block_id) or set(train.cell_id) & set(validation.cell_id):
            raise ValueError("Spatial overlap between training and validation.")
        splits.append((fold, train.source_row.to_numpy(), validation.source_row.to_numpy()))
    yield from splits


def buffer_training_rows(frame, train, validation, buffer_m=300.0):
    """Remove only training centers strictly closer than buffer_m to validation.

    Distances are Euclidean in the documented projected metric CRS. Centers
    exactly on the boundary remain eligible. Validation is never modified.
    """
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("buffer_m must be finite and nonnegative.")
    train, validation = np.asarray(train), np.asarray(validation)
    if not len(train) or not len(validation) or np.intersect1d(train, validation).size:
        raise ValueError("Training and validation must be nonempty and disjoint.")
    xy = frame[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("Buffer coordinates must be finite.")
    nearest = cdist(xy[train], xy[validation]).min(axis=1)
    keep = nearest >= buffer_m
    remaining, removed = train[keep], train[~keep]
    if (set(frame.iloc[remaining].target) != {0, 1}
            or set(frame.iloc[validation].target) != {0, 1}):
        raise ValueError("Buffer requires both classes in remaining training and validation.")
    return remaining, removed, nearest
