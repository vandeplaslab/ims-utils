from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ims_utils.kendrick import (  # noqa: E402 - optional pandas must be checked before this module import
    FORMULA_TO_REF,
    calculate_kendrick_mass_defect,
    calculate_mass_defect,
    filter_by_rules,
)


def test_numeric_and_formula_references_agree() -> None:
    mz = np.array([100.0, 200.125, 300.5])
    mass, defect = calculate_kendrick_mass_defect(mz, "CH2")
    numeric = calculate_kendrick_mass_defect(mz, FORMULA_TO_REF["CH2"])
    np.testing.assert_allclose(mass, mz * (14 / FORMULA_TO_REF["CH2"]))
    np.testing.assert_allclose(defect, np.trunc(mass) - mass)
    for actual, expected in zip(numeric, (mass, defect)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(calculate_mass_defect(mz), [0.0, 0.125, 0.5])


@pytest.mark.parametrize("reference", ["unknown", 0.0, -1.0, np.inf, np.nan])
def test_invalid_reference(reference: str | float) -> None:
    with pytest.raises(ValueError):
        calculate_kendrick_mass_defect(np.array([100.0]), reference)


def test_filter_rectangles_and_empty_rules_preserve_index() -> None:
    frame = pd.DataFrame({"mzs": [100.0, 200.0, 300.0], "kmd": [-0.1, -0.2, -0.3]}, index=[3, 5, 9])
    rules = [(100.0, 200.0, -0.1, -0.2)]
    pd.testing.assert_frame_equal(filter_by_rules(frame, rules), frame.iloc[2:])
    pd.testing.assert_frame_equal(filter_by_rules(frame, rules, select=True), frame.iloc[:2])
    pd.testing.assert_frame_equal(filter_by_rules(frame, []), frame)
    pd.testing.assert_frame_equal(filter_by_rules(frame, [], select=True), frame.iloc[:0])
