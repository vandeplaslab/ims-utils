"""Deisotoping functions."""
import typing as ty

import numpy as np


def deisotope(xs: np.ndarray, ys: np.ndarray, tolerance: float = 1e-4, distance: float = 1.00235, n_iter: int = 5):
    """De-isotope peaklist.

    This algorithm iterates over the list of peaks several times to ensure that peaks that were in-between isotopes
    are not ignored. Usually 3-5 passes is sufficient.

    Parameters
    ----------
    xs : np.ndarray
        array of x-axis values
    ys : np.ndarray
        array of y-axis intensities
    tolerance : float
        maximum distance between isotopes
    distance : float
        distance between isotopes
    n_iter : int
        maximum number of in iterations
    """
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


def _deisotope(xs, ys, min_dist, max_dist):
    i, n = 0, len(xs) - 1
    used, remaining_x, remaining_y, isotopes, isotope = [], [], [], [], []
    while i < n:
        j, k = 0, 1
        px, py = xs[i], ys[i]
        ij, ik = i + j, i + k
        while (ij < n and ik < n) and abs(px - xs[ik]) <= max_dist:
            ik = i + k
            # check whether distance between current peak and next peak is within the limits
            if min_dist <= abs(px - xs[ik]) <= max_dist:
                if not isotope:
                    isotope = [(px, py)]
                    used.append(px)
                px, py = xs[ik], ys[ik]
                isotope.append((px, py))
            j += 1
            k += 1
        used.extend([iso[0] for iso in isotope])
        # add isotopes
        if isotope:
            isotopes.append(isotope)
            i = i + j - 1
        else:
            i += 1
        isotope = []
    # tidy up afterwards
    for px, py in zip(xs, ys):
        if px not in used:
            remaining_x.append(px)
            remaining_y.append(py)
    return isotopes, remaining_x, remaining_y


def oms_deisotope(
    px: np.ndarray,
    py: np.ndarray,
    charge_range: ty.Tuple[int, int],
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
):
    """Deisotope peaklist using pyopenms methods."""
    from pyopenms import Deisotoper, MSSpectrum  # type: ignore

    spec = MSSpectrum()
    spec.set_peaks((px, py))

    min_charge, max_charge = charge_range

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
