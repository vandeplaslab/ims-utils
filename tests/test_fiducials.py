import numpy as np
import pytest

from ims_utils.fiducials import contour_to_points, distance2line, filter_contours, simplify_contour


def test_distance_to_line_horizontal_vertical_and_diagonal() -> None:
    assert distance2line([0, 0], [4, 0], [2, 3]) == pytest.approx(3.0)
    assert distance2line([0, 0], [0, 4], [3, 2]) == pytest.approx(3.0)
    assert distance2line([0, 0], [1, 1], [1, 0]) == pytest.approx(1 / np.sqrt(2))


def test_contour_filtering_and_conversion() -> None:
    small = np.array([[[1, 2]]])
    contour = np.array([[[0, 0]], [[1, 1]], [[2, 0]]])
    assert filter_contours([small, contour]) == [contour]
    np.testing.assert_array_equal(contour_to_points(contour), [[0, 0], [1, 1], [2, 0]])
    np.testing.assert_array_equal(simplify_contour([contour], 0.1)[0], contour[:2])
