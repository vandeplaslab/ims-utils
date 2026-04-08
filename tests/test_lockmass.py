"""Tests for ims_utils.lockmass module."""

import warnings

import numpy as np
import pytest

from ims_utils.lockmass import (
    LOCKMASS_THRESHOLD,
    MaximumIntensityLockmassEstimator,
    WeightedIntensityLockmassEstimator,
    _nb_estimate_lockmass_maximum,
    _nb_estimate_lockmass_shifts,
    _prepare_lockmass,
    fast_roll,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(n: int = 1000, peak_mz: float = 500.0, window: float = 0.5, peak_intensity: float = 2000.0):
    """Return (mz_x, mz_y) with a single Gaussian peak centred at peak_mz."""
    mz_x = np.linspace(499.0, 501.0, n)
    sigma = 0.05
    mz_y = peak_intensity * np.exp(-0.5 * ((mz_x - peak_mz) / sigma) ** 2).astype(np.float32)
    return mz_x, mz_y


# ---------------------------------------------------------------------------
# fast_roll
# ---------------------------------------------------------------------------


class TestFastRoll:
    def test_zero_shift_returns_same_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = fast_roll(arr, 0)
        assert result is arr

    def test_positive_shift(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = fast_roll(arr, 2)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 2.0])

    def test_negative_shift(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = fast_roll(arr, -2)
        np.testing.assert_array_equal(result, [3.0, 4.0, 0.0, 0.0])

    def test_custom_fill_value(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = fast_roll(arr, 1, fill_value=-1.0)
        np.testing.assert_array_equal(result, [-1.0, 1.0, 2.0])

    def test_float_shift_is_rounded(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        # 1.6 rounds to 2
        result = fast_roll(arr, 1.6)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 2.0])

    def test_numpy_int_shift_accepted(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = fast_roll(arr, np.int64(2))
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 2.0])

    def test_numpy_float_shift_accepted(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = fast_roll(arr, np.float32(2.0))
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 2.0])

    def test_invalid_type_raises(self):
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="`num` must be a numeric value"):
            fast_roll(arr, "2")  # type: ignore[arg-type]

    def test_float_zero_does_not_raise(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = fast_roll(arr, 0.0)
        assert result is arr

    def test_validation_runs_before_allocation(self):
        """Invalid type should raise before any work is done."""
        arr = np.ones(10)
        with pytest.raises(ValueError):
            fast_roll(arr, "bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _prepare_lockmass
# ---------------------------------------------------------------------------


class TestPrepareLockmass:
    def test_returns_correct_shapes(self):
        mz_x = np.linspace(490, 510, 1000)
        peaks = np.array([495.0, 500.0, 505.0])
        mz_indices, peak_indices, masks, offsets, starts, stops, offsets_arr = _prepare_lockmass(mz_x, peaks)

        assert mz_indices.shape == mz_x.shape
        assert len(peak_indices) == len(peaks)
        assert set(masks.keys()) == set(peaks)
        assert set(offsets.keys()) == set(peaks)
        assert starts.shape == (len(peaks),)
        assert stops.shape == (len(peaks),)
        assert offsets_arr.shape == (len(peaks),)

    def test_peak_indices_are_sorted(self):
        # LockmassEstimator sorts peaks before calling _prepare_lockmass; pass sorted peaks here
        mz_x = np.linspace(490, 510, 1000)
        peaks = np.array([495.0, 500.0, 505.0])  # already sorted
        _, peak_indices, _, _, starts, _stops, _ = _prepare_lockmass(mz_x, peaks)
        assert peak_indices[0] < peak_indices[1] < peak_indices[2]
        assert starts[0] < starts[1] < starts[2]

    def test_slice_arrays_are_contiguous_views(self):
        mz_x = np.linspace(490, 510, 1000)
        peaks = np.array([500.0])
        _, _, masks, _, starts, stops, _offsets_arr = _prepare_lockmass(mz_x, peaks)
        # Slice-based view should match boolean-mask indexed values
        s, e = int(starts[0]), int(stops[0])
        mask = next(iter(masks.values()))
        np.testing.assert_array_equal(mz_x[s:e], mz_x[mask])


# ---------------------------------------------------------------------------
# _nb_estimate_lockmass_maximum
# ---------------------------------------------------------------------------


class TestNbEstimateLockmassMaximum:
    def setup_method(self):
        self.mz_x, self.mz_y = _make_profile(peak_mz=500.0)
        self.peaks = np.array([500.0])
        _, _, _, _, self.starts, self.stops, self.offsets_arr = _prepare_lockmass(self.mz_x, self.peaks)

    def test_zero_shift_when_peak_centred(self):
        out = np.zeros(1, dtype=np.float32)
        _nb_estimate_lockmass_maximum(self.mz_y, self.starts, self.stops, self.offsets_arr, out)
        assert out[0] == pytest.approx(0.0, abs=2)

    def test_nonzero_shift_when_peak_offset(self):
        shifted_y = fast_roll(self.mz_y, 5)
        out = np.zeros(1, dtype=np.float32)
        _nb_estimate_lockmass_maximum(shifted_y, self.starts, self.stops, self.offsets_arr, out)
        assert out[0] != pytest.approx(0.0, abs=0.1)

    def test_output_array_reused(self):
        preallocated = np.zeros(1, dtype=np.float32)
        result = _nb_estimate_lockmass_maximum(self.mz_y, self.starts, self.stops, self.offsets_arr, preallocated)
        assert result is preallocated


# ---------------------------------------------------------------------------
# _nb_estimate_lockmass_shifts
# ---------------------------------------------------------------------------


class TestNbEstimateLockmassShifts:
    def setup_method(self):
        self.mz_x, self.mz_y = _make_profile(peak_mz=500.0, peak_intensity=2000.0)
        self.peaks = np.array([500.0])
        _, self.peak_indices, _, _, self.starts, self.stops, self.offsets_arr = _prepare_lockmass(self.mz_x, self.peaks)

    def test_zero_shift_when_peak_centred(self):
        out = np.zeros(1, dtype=np.float32)
        _nb_estimate_lockmass_shifts(
            self.mz_y,
            self.starts,
            self.stops,
            self.peak_indices,
            self.offsets_arr,
            False,
            out,
            500.0,
            250.0,
            0.25,
        )
        assert out[0] == pytest.approx(0.0, abs=2)

    def test_returns_zeros_when_below_threshold(self):
        low_y = self.mz_y * 0.001
        out = np.zeros(1, dtype=np.float32)
        _nb_estimate_lockmass_shifts(
            low_y,
            self.starts,
            self.stops,
            self.peak_indices,
            self.offsets_arr,
            False,
            out,
            LOCKMASS_THRESHOLD,
            LOCKMASS_THRESHOLD * 0.5,
            0.25,
        )
        assert np.all(out == 0.0)

    def test_weighted_true_accepted(self):
        out = np.zeros(1, dtype=np.float32)
        _nb_estimate_lockmass_shifts(
            self.mz_y,
            self.starts,
            self.stops,
            self.peak_indices,
            self.offsets_arr,
            True,
            out,
            500.0,
            250.0,
            0.25,
        )
        assert out.shape == (1,)


# ---------------------------------------------------------------------------
# MaximumIntensityLockmassEstimator
# ---------------------------------------------------------------------------


class TestMaximumIntensityLockmassEstimator:
    def setup_method(self):
        self.mz_x, self.mz_y = _make_profile(peak_mz=500.0, peak_intensity=2000.0)
        self.peaks = np.array([500.0])
        self.estimator = MaximumIntensityLockmassEstimator(self.mz_x, self.peaks)

    def test_estimate_returns_array(self):
        result = self.estimator.estimate(self.mz_y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_zero_shift_for_centred_peak(self):
        result = self.estimator.estimate(self.mz_y)
        assert result[0] == pytest.approx(0.0, abs=2)

    def test_weighted_param_accepted_without_error(self):
        # weighted is unused in this estimator but must not raise
        result = self.estimator.estimate(self.mz_y, weighted=True)
        assert result.shape == (1,)
        result2 = self.estimator.estimate(self.mz_y, weighted=False)
        np.testing.assert_array_equal(result, result2)

    def test_output_preallocated(self):
        out = np.zeros(1, dtype=np.float32)
        result = self.estimator.estimate(self.mz_y, out=out)
        assert result is out

    def test_apply_with_float_shift(self):
        shifted = self.estimator.apply(self.mz_y, 2.4)  # rounds to 2
        np.testing.assert_array_equal(shifted[2:], self.mz_y[:-2])

    def test_apply_with_numpy_int_shift(self):
        shifted = self.estimator.apply(self.mz_y, np.int64(3))
        np.testing.assert_array_equal(shifted[3:], self.mz_y[:-3])


# ---------------------------------------------------------------------------
# WeightedIntensityLockmassEstimator
# ---------------------------------------------------------------------------


class TestWeightedIntensityLockmassEstimator:
    def setup_method(self):
        self.mz_x, self.mz_y = _make_profile(peak_mz=500.0, peak_intensity=2000.0)
        self.peaks = np.array([500.0])
        self.estimator = WeightedIntensityLockmassEstimator(self.mz_x, self.peaks)

    def test_estimate_returns_array(self):
        result = self.estimator.estimate(self.mz_y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_zero_shift_for_centred_peak(self):
        result = self.estimator.estimate(self.mz_y)
        assert result[0] == pytest.approx(0.0, abs=2)

    def test_returns_zeros_when_below_threshold_and_warns(self):
        low_y = self.mz_y * 0.001
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = self.estimator.estimate(low_y)
        assert np.all(out == 0.0)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
        assert any("below the lockmass threshold" in str(w.message) for w in caught)

    def test_centroid_frac_stored(self):
        est = WeightedIntensityLockmassEstimator(self.mz_x, self.peaks, centroid_frac=0.5)
        assert est.centroid_frac == 0.5

    def test_output_preallocated(self):
        out = np.zeros(1, dtype=np.float32)
        result = self.estimator.estimate(self.mz_y, out=out)
        assert result is out

    def test_weighted_false_accepted(self):
        result = self.estimator.estimate(self.mz_y, weighted=False)
        assert result.shape == (1,)
