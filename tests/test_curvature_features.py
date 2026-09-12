import numpy as np

from src.features.derive_curvature_features import (
    calculate_curvatures,
)


def test_planar_surface_has_zero_curvature():
    """A tilted plane should have approximately zero curvature."""

    y, x = np.mgrid[0:20, 0:20]

    elevation = (
        2.0 * x
        + 3.0 * y
        + 100.0
    ).astype(float)

    profile, plan = calculate_curvatures(
        elevation,
        cellsize_x=1.0,
        cellsize_y=1.0,
    )

    valid_profile = profile[np.isfinite(profile)]
    valid_plan = plan[np.isfinite(plan)]

    assert np.allclose(
        valid_profile,
        0.0,
        atol=1e-10,
    )

    assert np.allclose(
        valid_plan,
        0.0,
        atol=1e-10,
    )


def test_curved_surface_is_not_zero():
    """A paraboloid should produce non-zero curvature."""

    y, x = np.mgrid[-10:11, -10:11]

    elevation = (
        x**2
        + y**2
    ).astype(float)

    profile, plan = calculate_curvatures(
        elevation,
        cellsize_x=1.0,
        cellsize_y=1.0,
    )

    valid_profile = profile[np.isfinite(profile)]
    valid_plan = plan[np.isfinite(plan)]

    assert np.any(
        np.abs(valid_profile) > 1e-6
    )

    assert np.any(
        np.abs(valid_plan) > 1e-6
    )


def test_surface_below_minimum_slope_has_nan_curvature(capsys):
    """A plane below 0.1 degrees should have undefined curvature."""

    y, x = np.mgrid[0:20, 0:20]
    gradient = np.tan(np.radians(0.05))
    elevation = 100.0 + gradient * (x + y) / np.sqrt(2.0)

    profile, plan = calculate_curvatures(
        elevation,
        cellsize_x=1.0,
        cellsize_y=1.0,
    )

    assert np.isnan(profile).all()
    assert np.isnan(plan).all()
    output = capsys.readouterr().out
    assert "Minimum slope for curvature: 0.10 degrees" in output
    assert "Valid curvature pixels before slope threshold: 400" in output
    assert "Pixels removed by slope threshold: 400" in output
    assert "Valid curvature pixels after threshold: 0" in output


def test_plane_above_minimum_slope_has_zero_curvature(capsys):
    """A plane above 0.1 degrees should retain finite, zero curvature."""

    y, x = np.mgrid[0:20, 0:20]
    gradient = np.tan(np.radians(0.11))
    elevation = 100.0 + gradient * (x + y) / np.sqrt(2.0)

    profile, plan = calculate_curvatures(
        elevation,
        cellsize_x=1.0,
        cellsize_y=1.0,
    )

    assert np.isfinite(profile).all()
    assert np.isfinite(plan).all()
    assert np.allclose(profile, 0.0, atol=1e-10)
    assert np.allclose(plan, 0.0, atol=1e-10)
    output = capsys.readouterr().out
    assert "Minimum slope for curvature: 0.10 degrees" in output
    assert "Valid curvature pixels before slope threshold: 400" in output
    assert "Pixels removed by slope threshold: 0" in output
    assert "Valid curvature pixels after threshold: 400" in output
