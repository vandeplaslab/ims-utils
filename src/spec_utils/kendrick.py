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
    """Calculate KM and KMD for specified formula"""
    if ref not in FORMULA_TO_REF:
        raise ValueError(f"Unknown formula: {ref}")
    if isinstance(ref, str):
        ref = FORMULA_TO_REF[ref]
    part, nominal = np.modf(ref)

    # calculate kendrick mass
    km = mzs * (nominal / ref)

    # calculate kendrick mass defect
    km_part, km_nominal = np.modf(km)
    kmd = km_nominal - km
    return km, kmd


def calculate_mass_defect(mzs: np.ndarray) -> np.ndarray:
    """Calculate mass defect for specified m/z values"""
    part, _ = np.modf(mzs)
    return part


def filter_by_rules(
    df: pd.DataFrame, filter_regions: list[tuple[float, float, float, float]], select: bool = False
) -> pd.DataFrame:
    """Filter dataframe based on specified rules"""
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
