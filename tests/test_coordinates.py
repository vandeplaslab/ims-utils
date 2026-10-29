"""Test coordinates module."""

from ims_utils.coordinates import get_scan_direction, get_spacing, get_spacing_consecutive


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
