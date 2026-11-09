"""Test coordinates module."""

from ims_utils.coordinates import get_scan_direction, get_spacing, get_spacing_consecutive


def test_consecutive_spacing_sparse_histogram_has_a_finite_estimate() -> None:
    assert get_spacing_consecutive([0.0, 1.0, 101.0], [0.0, 0.0, 0.0]) == (1.0, 0.0)
    assert get_spacing_consecutive([], []) == (0.0, 0.0)


def test_scan_direction_ties_and_constant_axes() -> None:
    assert get_scan_direction([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]) == ("x", 1, 1)
    assert get_scan_direction([0.0, 0.0], [0.0, 0.0]) == ("x", 0, 0)


def test_get_scan_direction():
    """Test get_scan_direction function."""
    x_coords = [0, 1, 2, 3, 4]
    y_coords = [0, 0, 0, 0, 0]
    fast_axis, x_dir, y_dir = get_scan_direction(x_coords, y_coords)
    assert fast_axis == "x"
    assert x_dir == 1
    assert y_dir == 0

    x_coords = [0, 0, 0, 0, 0]
    y_coords = [4, 3, 2, 1, 0]
    fast_axis, x_dir, y_dir = get_scan_direction(x_coords, y_coords)
    assert fast_axis == "y"
    assert x_dir == 0
    assert y_dir == -1


def test_get_spacing():
    """Test get_spacing function."""
    x_coords = [0, 1, 2, 3, 4]
    y_coords = [0, 0, 0, 0, 0]
    x_spacing, y_spacing = get_spacing(x_coords, y_coords)
    assert x_spacing == 1.0
    assert y_spacing == 0.0

    x_coords = [0, 0, 0, 0, 0]
    y_coords = [4, 3, 2, 1, 0]
    x_spacing, y_spacing = get_spacing(x_coords, y_coords)
    assert x_spacing == 0.0
    assert y_spacing == 1.0


def test_get_spacing_consecutive():
    """Test get_spacing_consecutive function."""
    x_coords = [0, 1, 2, 3, 4]
    y_coords = [0, 0, 0, 0, 0]
    x_spacing, y_spacing = get_spacing_consecutive(x_coords, y_coords)
    assert x_spacing == 1.0
    assert y_spacing == 0.0

    x_coords = [0, 0, 0, 0, 0]
    y_coords = [4, 3, 2, 1, 0]
    x_spacing, y_spacing = get_spacing_consecutive(x_coords, y_coords)
    assert x_spacing == 0.0
    assert y_spacing == 1.0
