"""Deisotoping functions."""

from __future__ import annotations

import numpy as np


def deisotope(
    xs: np.ndarray, ys: np.ndarray, tolerance: float = 1e-4, distance: float = 1.00235, n_iter: int = 5
) -> tuple[list[np.ndarray], list[float] | np.ndarray, list[float] | np.ndarray]:
    """De-isotope peaklist.

    This algorithm iterates over the list of peaks several times to ensure that peaks that were in-between isotopes
    are not ignored. Usually 3-5 passes is sufficient.

    Parameters
    ----------
    xs : np.ndarray
        Ascending one-dimensional m/z array.
    ys : np.ndarray
        array of y-axis intensities
    tolerance : float
        Absolute tolerance around the expected isotope distance.
    distance : float
        distance between isotopes
    n_iter : int
        Maximum number of passes. Zero leaves the input unchanged.

    Returns
    -------
    isotopes : list[np.ndarray]
        Each group has shape (n_peaks, 2), holding m/z and intensity pairs.
    remaining_x, remaining_y : list[float] or np.ndarray
        Ungrouped m/z and intensities. Lists after processing; original arrays
        when no passes are requested. Inputs are not modified.
    """
    if len(xs) != len(ys):
        raise ValueError("m/z and intensities must have the same length.")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative.")
    min_dist, max_dist = distance - tolerance, distance + tolerance

    isotopes = []
    n_last = 0  # the number of isotopes in the previous iteration
    while n_iter != 0 and len(xs) > 0:
        _isotopes, xs, ys = _deisotope(xs, ys, min_dist, max_dist)
        isotopes.extend(_isotopes)
        # break early if there is no addition
        if n_last == len(isotopes):
            break
        n_last = len(isotopes)
        n_iter -= 1
    # let's return the isotope lists as numpy arrays for easier slicing
    isotopes = [np.asarray(iso) for iso in isotopes]
    return isotopes, xs, ys


def _deisotope(
    xs: np.ndarray | list[float], ys: np.ndarray | list[float], min_dist: float, max_dist: float
) -> tuple[list[list[tuple[float, float]]], list[float], list[float]]:
    used: set[int] = set()
    isotopes = []
    for i in range(len(xs)):
        if i in used:
            continue
        indices = [i]
        for j in range(i + 1, len(xs)):
            if j in used:
                continue
            difference = xs[j] - xs[indices[-1]]
            if difference > max_dist:
                break
            if min_dist <= difference <= max_dist:
                indices.append(j)
        if len(indices) > 1:
            isotopes.append([(xs[index], ys[index]) for index in indices])
            used.update(indices)
    remaining_x = [value for i, value in enumerate(xs) if i not in used]
    remaining_y = [value for i, value in enumerate(ys) if i not in used]
    return isotopes, remaining_x, remaining_y


def oms_deisotope(
    px: np.ndarray,
    py: np.ndarray,
    charge_range: tuple[int, int],
    fragment_tolerance: int = 10,
    fragment_unit_ppm: bool = True,
    min_isotopes: int = 2,
    max_isotopes: int = 5,
    use_decreasing_model: bool = True,
    start_intensity_check: int = 3,
    annotate_iso_peak_count: bool = False,
    annotate_charge: bool = False,
    make_single_charged: bool = False,
    keep_only_deisotoped: bool = False,
    add_up_intensity: bool = False,
    annotate_features: bool = False,
):
    """Deisotope an m/z peak list with OpenMS and return its resulting peaks.

    Parameters
    ----------
    px, py : np.ndarray
        One-dimensional m/z values in Da and matching intensities. OpenMS
        expects the m/z values in ascending order.
    charge_range : tuple[int, int]
        Inclusive minimum and maximum charge states to consider.
    fragment_tolerance : int
        M/z tolerance, interpreted as ppm when ``fragment_unit_ppm`` is true
        and as Da otherwise.
    fragment_unit_ppm : bool
        Interpret the fragment tolerance in ppm when true, or Da when false.
    min_isotopes, max_isotopes : int
        Minimum and maximum isotope count for a candidate group.
    use_decreasing_model : bool
        Use the decreasing isotope intensity model.
    start_intensity_check : int
        Isotope index at which to check the intensity trend.
    annotate_iso_peak_count, annotate_charge : bool
        Request isotope count and charge annotations from OpenMS.
    make_single_charged : bool
        Convert accepted groups to singly charged m/z values.
    keep_only_deisotoped : bool
        Remove peaks not assigned to accepted isotope groups.
    add_up_intensity : bool
        Combine isotope intensities during de-isotoping.
    annotate_features : bool
        Request feature annotations when supported by the installed OpenMS.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The m/z values and intensities retrieved from the OpenMS spectrum after
        processing. The input arrays are passed to a newly created spectrum and
        are not directly modified by this function.
    """
    from pyopenms import Deisotoper, MSSpectrum  # type: ignore

    spec = MSSpectrum()
    spec.set_peaks((px, py))

    min_charge, max_charge = charge_range

    try:
        Deisotoper.deisotopeAndSingleCharge(
            spec,
            fragment_tolerance,
            fragment_unit_ppm,
            abs(min_charge),
            abs(max_charge),
            keep_only_deisotoped,
            min_isotopes,
            max_isotopes,
            make_single_charged,
            annotate_charge,
            annotate_iso_peak_count,
            use_decreasing_model,
            start_intensity_check,
            add_up_intensity,
            annotate_features,
        )
    except TypeError:
        Deisotoper.deisotopeAndSingleCharge(
            spec,
            fragment_tolerance,
            fragment_unit_ppm,
            abs(min_charge),
            abs(max_charge),
            keep_only_deisotoped,
            min_isotopes,
            max_isotopes,
            make_single_charged,
            annotate_charge,
            annotate_iso_peak_count,
            use_decreasing_model,
            start_intensity_check,
            add_up_intensity,
        )
    px, py = spec.get_peaks()
    return px, py
