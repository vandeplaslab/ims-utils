"""Coordinate utilities for IMS data."""

from __future__ import annotations

import numpy as np


def get_scan_direction(x_coords: np.ndarray, y_coords: np.ndarray, tolerance: float = 1e-6) -> tuple[str, int, int]:
    """Determine the primary scan axis and direction of motion.

    Parameters
    ----------
    x_coords : np.ndarray
        Motor X positions in acquisition order.
    y_coords : np.ndarray
        Motor Y positions in acquisition order.
    tolerance : float
        Differences with absolute value below this are treated as no movement.

    Returns
    -------
    tuple[str, int, int]
        Fast axis ("x" or "y"), x direction and y direction (1 increasing,
        -1 decreasing, 0 constant). Direction is the sign of the most common
        non-zero step along the axis, so line returns do not dominate.
    """

    def _direction(values: np.ndarray) -> int:
        steps = np.diff(values)
        steps = steps[np.abs(steps) > tolerance]
        if steps.size == 0:
            return 0
        n_positive = np.count_nonzero(steps > 0)
        n_negative = steps.size - n_positive
        return 1 if n_positive >= n_negative else -1

    x_steps = np.abs(np.diff(x_coords))
    y_steps = np.abs(np.diff(y_coords))
    n_x_moves = np.count_nonzero(x_steps > tolerance)
    n_y_moves = np.count_nonzero(y_steps > tolerance)

    fast_axis = "x" if n_x_moves >= n_y_moves else "y"
    return fast_axis, _direction(x_coords), _direction(y_coords)


def get_spacing(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    tolerance: float = 1e-6,
    decimals: int = 1,
) -> tuple[float, float]:
    """Estimate the grid spacing along x and y from motor positions.

    Parameters
    ----------
    x_coords : np.ndarray
        Motor X positions in acquisition order.
    y_coords : np.ndarray
        Motor Y positions in acquisition order.
    tolerance : float
        Minimum absolute difference to be considered a step.
    decimals : int
        Number of decimals to round the result to.

    Returns
    -------
    tuple[float, float]
        Spacing along x and y in the same units as the input.
    """

    def _spacing(values: np.ndarray) -> float:
        # unique positions sorted; wobble creates near-duplicates, so cluster them
        unique_values = np.sort(np.unique(values))
        steps = np.diff(unique_values)
        steps = steps[steps > tolerance]
        if steps.size == 0:
            return 0.0
        # estimate pitch as the median of the smallest steps (wobble-robust):
        # steps within 50% of the smallest sizeable step are a single pitch
        smallest = np.percentile(steps, 10)
        # ignore wobble-sized steps (less than half the pitch)
        valid = steps[steps > 0.5 * smallest * 2]
        pitch_guess = np.median(valid) if valid.size else smallest
        # discard wobble-induced small steps and merge multiples into pitch units
        valid = steps[steps > 0.5 * pitch_guess]
        multiples = np.maximum(np.round(valid / pitch_guess), 1)
        return float(np.round(np.sum(valid) / np.sum(multiples), decimals))

    return _spacing(x_coords), _spacing(y_coords)


def get_spacing_consecutive(x_coords: np.ndarray, y_coords: np.ndarray, decimals: int = 1) -> tuple[float, float]:
    """Estimate spacing from consecutive-step statistics (robust alternative)."""

    def _spacing(values: np.ndarray) -> float:
        steps = np.abs(np.diff(values))
        steps = steps[steps > 1e-6]
        if steps.size == 0:
            return 0.0
        # histogram mode of steps is the pitch; take median of steps near the mode
        counts, edges = np.histogram(steps, bins="auto")
        mode_center = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
        near_mode = steps[np.abs(steps - mode_center) < 0.25 * mode_center]
        return float(np.round(np.median(near_mode), decimals))

    return _spacing(x_coords), _spacing(y_coords)
