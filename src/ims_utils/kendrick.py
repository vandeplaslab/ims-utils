"""Kendrick mass defect calculation and related functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

FORMULA_TO_REF = {
    "H": 1.0078250322,
    "H2": 2.01565006454,
    "C": 12.0,
    "CH2": 14.01565006454,
    "CH4": 16.0313,
    "C2H4": 28.03130012908,
}


def calculate_kendrick_mass_defect(mzs: np.ndarray, ref: str | float) -> tuple[np.ndarray, np.ndarray]:
    """Return Kendrick mass and truncated-nominal-minus-exact defect.

    ``ref`` is a formula in ``FORMULA_TO_REF`` or a finite positive exact mass.
    Both nominal reference mass and nominal Kendrick mass use truncation,
    retaining the library's existing convention. Input m/z values are unchanged.
    """
    if isinstance(ref, str):
        if ref not in FORMULA_TO_REF:
            raise ValueError(f"Unknown formula: {ref}")
        ref = FORMULA_TO_REF[ref]
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError("Reference mass must be finite and positive.")
    _part, nominal = np.modf(ref)

    # calculate kendrick mass
    km = mzs * (nominal / ref)

    # calculate kendrick mass defect
    _km_part, km_nominal = np.modf(km)
    kmd = km_nominal - km
    return km, kmd


def calculate_mass_defect(mzs: np.ndarray) -> np.ndarray:
    """Calculate mass defect for specified m/z values."""
    part, _ = np.modf(mzs)
    return part


def filter_by_rules(
    df: pd.DataFrame, filter_regions: list[tuple[float, float, float, float]], select: bool = False
) -> pd.DataFrame:
    """Exclude rectangular regions, or retain their union when ``select=True``.

    Each region is (minimum m/z, maximum m/z, maximum KMD, minimum KMD),
    inclusive at all edges. Empty rules retain all rows, or none when selecting.
    The input dataframe and its index are preserved.
    """
    if not filter_regions:
        return df.iloc[:0] if select else df.copy()
    mask = np.any(
        [
            (df.mzs >= region[0]) & (df.mzs <= region[1]) & (df.kmd <= region[2]) & (df.kmd >= region[3])
            for region in filter_regions
        ],
        axis=0,
    )
    if select:
        return df[mask]
    return df[~mask]
