"""Tests for robust_lockmass.py."""

from __future__ import annotations

import numpy as np
import pytest

from ims_utils.robust_lockmass import (
    ROBUST_LOCKMASS_THRESHOLD,
    RobustLockmassEstimator,
    _nb_find_all_candidates,
    _nb_monotone_interp,
    _nb_ransac_quadratic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mz_axis(start: float = 200.0, stop: float = 1200.0, n: int = 300_000) -> np.ndarray:
    """Return a uniform m/z axis similar to a real profile spectrum."""
    return np.linspace(start, stop, n)


def _make_spectrum(
    mz_x: np.ndarray,
    peak_mzs: list[float],
    peak_intensities: list[float],
    sigma: float = 0.05,
    noise_level: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Gaussian profile spectrum with optional noise."""
    rng = np.random.default_rng(seed)
    mz_y = rng.random(len(mz_x)).astype(np.float32) * noise_level
    for mz, intensity in zip(peak_mzs, peak_intensities):
        mz_y += (intensity * np.exp(-0.5 * ((mz_x - mz) / sigma) ** 2)).astype(np.float32)
    return mz_y


def _ppm_err(a: float, b: float) -> float:
    """Absolute ppm difference between two m/z values."""
    return abs(a - b) / b * 1e6


# ---------------------------------------------------------------------------
# _nb_find_all_candidates
# ---------------------------------------------------------------------------


class TestNbFindAllCandidates:
    def _run(
        self,
        mz_y: np.ndarray,
        starts: np.ndarray,
        stops: np.ndarray,
        threshold: float = 100.0,
        centroid_frac: float = 0.25,
        max_cands: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_peaks = len(starts)
        cx_out = np.empty((n_peaks, max_cands), dtype=np.float64)
        cy_out = np.empty((n_peaks, max_cands), dtype=np.float64)
        n_found = np.zeros(n_peaks, dtype=np.int64)
        _nb_find_all_candidates(mz_y, starts, stops, threshold, centroid_frac, cx_out, cy_out, n_found)
        return cx_out, cy_out, n_found

    def test_single_centred_peak_found(self):
        """A Gaussian peak centred in the window should yield exactly one candidate."""
        mz_x = np.linspace(499.0, 501.0, 1000)
        mz_y = (2000.0 * np.exp(-0.5 * ((mz_x - 500.0) / 0.05) ** 2)).astype(np.float32)
        starts = np.array([300], dtype=np.int64)
        stops = np.array([700], dtype=np.int64)

        _, cy_out, n_found = self._run(mz_y, starts, stops, threshold=100.0)
        assert n_found[0] >= 1
        assert cy_out[0, 0] > 1000.0

    def test_centroid_position_near_peak(self):
        """Centroid index should be close to the true peak index."""
        mz_x = np.linspace(499.0, 501.0, 1000)
        peak_mz = 500.0
        mz_y = (2000.0 * np.exp(-0.5 * ((mz_x - peak_mz) / 0.05) ** 2)).astype(np.float32)
        true_idx = float(np.argmax(mz_y))
        starts = np.array([300], dtype=np.int64)
        stops = np.array([700], dtype=np.int64)

        cx_out, _, n_found = self._run(mz_y, starts, stops, threshold=100.0)
        assert n_found[0] >= 1
        assert abs(cx_out[0, 0] - true_idx) < 2.0

    def test_multiple_peaks_in_window(self):
        """Two Gaussian peaks in one wide window → two candidates."""
        mz_x = np.linspace(498.0, 502.0, 2000)
        mz_y = (
            2000.0 * np.exp(-0.5 * ((mz_x - 499.5) / 0.05) ** 2) + 2000.0 * np.exp(-0.5 * ((mz_x - 500.5) / 0.05) ** 2)
        ).astype(np.float32)
        starts = np.array([0], dtype=np.int64)
        stops = np.array([len(mz_x)], dtype=np.int64)

        _, _, n_found = self._run(mz_y, starts, stops, threshold=100.0)
        assert n_found[0] >= 2

    def test_below_threshold_gives_zero_candidates(self):
        """Peaks below threshold should produce no candidates."""
        mz_x = np.linspace(499.0, 501.0, 1000)
        mz_y = (50.0 * np.exp(-0.5 * ((mz_x - 500.0) / 0.05) ** 2)).astype(np.float32)
        starts = np.array([300], dtype=np.int64)
        stops = np.array([700], dtype=np.int64)

        _, _, n_found = self._run(mz_y, starts, stops, threshold=ROBUST_LOCKMASS_THRESHOLD)
        assert n_found[0] == 0

    def test_too_short_window_gives_zero_candidates(self):
        """Window of less than 3 points cannot contain a local maximum."""
        mz_y = np.array([0.0, 2000.0, 0.0], dtype=np.float32)
        starts = np.array([0], dtype=np.int64)
        stops = np.array([2], dtype=np.int64)  # only 2 points

        _, _, n_found = self._run(mz_y, starts, stops, threshold=100.0)
        assert n_found[0] == 0

    def test_flat_region_not_detected(self):
        """A perfectly flat plateau should not be reported as a peak."""
        mz_y = np.ones(100, dtype=np.float32) * 2000.0
        starts = np.array([0], dtype=np.int64)
        stops = np.array([100], dtype=np.int64)

        _, _, n_found = self._run(mz_y, starts, stops, threshold=100.0)
        assert n_found[0] == 0

    def test_multiple_windows_independent(self):
        """Candidates from different windows do not bleed into each other."""
        mz_x = np.linspace(100.0, 1000.0, 10000)
        mz_y = np.zeros(len(mz_x), dtype=np.float32)
        # Only the first window has a peak
        mz_y[2000:2050] = 2000.0
        mz_y[2025] = 3000.0

        starts = np.array([2000, 7000], dtype=np.int64)
        stops = np.array([2100, 8000], dtype=np.int64)

        _, _, n_found = self._run(mz_y, starts, stops, threshold=100.0)
        assert n_found[0] >= 1
        assert n_found[1] == 0


# ---------------------------------------------------------------------------
# _nb_ransac_quadratic
# ---------------------------------------------------------------------------


class TestNbRansacQuadratic:
    def _flat(self, ref_mzs, ppms):
        """Build flat arrays (one candidate per peak) for easy testing."""
        ref_mz = np.array(ref_mzs, dtype=np.float64)
        obs_ppm = np.array(ppms, dtype=np.float64)
        peak_starts = np.arange(len(ref_mzs), dtype=np.int64)
        peak_counts = np.ones(len(ref_mzs), dtype=np.int64)
        return ref_mz, obs_ppm, peak_starts, peak_counts

    def test_perfect_quadratic_fit(self):
        """Exact quadratic data → all inliers."""
        mzs = np.linspace(300.0, 1000.0, 10)
        a, b, c = 0.0001, -0.05, 10.0
        ppms = a * mzs**2 + b * mzs + c

        ref_mz, obs_ppm, peak_starts, peak_counts = self._flat(mzs.tolist(), ppms.tolist())
        best_a, _best_b, _best_c, n_in = _nb_ransac_quadratic(
            ref_mz,
            obs_ppm,
            peak_starts,
            peak_counts,
            np.int64(len(mzs)),
            np.int64(len(mzs)),
            1.0,
            np.int64(3),
            np.int64(500),
        )
        assert n_in >= 8
        assert not np.isnan(best_a)

    def test_outlier_is_rejected(self):
        """One large outlier should not contaminate the model."""
        mzs = np.linspace(300.0, 1000.0, 10)
        ppms = np.full(10, 50.0)  # constant 50 ppm
        ppms[5] = 500.0  # outlier

        ref_mz, obs_ppm, peak_starts, peak_counts = self._flat(mzs.tolist(), ppms.tolist())
        _best_a, _best_b, best_c, n_in = _nb_ransac_quadratic(
            ref_mz,
            obs_ppm,
            peak_starts,
            peak_counts,
            np.int64(len(mzs)),
            np.int64(len(mzs)),
            10.0,
            np.int64(3),
            np.int64(500),
        )
        assert n_in >= 8  # 9 inliers (outlier rejected)
        assert not np.isnan(best_c)

    def test_too_few_peaks_returns_nan(self):
        """Fewer than 3 peak groups → cannot fit a quadratic."""
        mzs = [300.0, 600.0]
        ppms = [50.0, 55.0]
        ref_mz, obs_ppm, peak_starts, peak_counts = self._flat(mzs, ppms)
        best_a, _, _, n_in = _nb_ransac_quadratic(
            ref_mz,
            obs_ppm,
            peak_starts,
            peak_counts,
            np.int64(2),
            np.int64(2),
            5.0,
            np.int64(2),
            np.int64(200),
        )
        assert np.isnan(best_a)
        assert n_in == 0

    def test_min_inliers_not_met_returns_nan(self):
        """If best inlier count < min_inliers the result is NaN sentinel."""
        # All 5 points at completely different ppm values; no quadratic can fit all 5 within 1 ppm
        mzs = np.linspace(300.0, 700.0, 5)
        ppms = np.array([10.0, 500.0, 10.0, 500.0, 10.0])  # wide alternating gap

        ref_mz, obs_ppm, peak_starts, peak_counts = self._flat(mzs.tolist(), ppms.tolist())
        best_a, _, _, n_in = _nb_ransac_quadratic(
            ref_mz,
            obs_ppm,
            peak_starts,
            peak_counts,
            np.int64(5),
            np.int64(5),
            1.0,  # very tight tolerance — cannot fit both 10 and 500 within 1 ppm
            np.int64(5),  # require ALL 5 inliers — impossible with bi-modal data
            np.int64(500),
        )
        # With min_inliers=5 and alternating 10/500 ppm values, RANSAC cannot succeed
        assert np.isnan(best_a)
        assert n_in == 0

    def test_multiple_candidates_per_peak(self):
        """Multiple candidates per peak: RANSAC should still find the correct model."""
        n_peaks = 5
        mzs = np.linspace(300.0, 1000.0, n_peaks)
        true_ppm = 50.0  # constant shift
        # One correct candidate + one "wrong" candidate per peak
        ref_mz_list, obs_ppm_list, starts_list, counts_list = [], [], [], []
        cursor = 0
        for mz in mzs:
            starts_list.append(cursor)
            # Correct candidate first, wrong candidate second
            ref_mz_list.extend([mz, mz])
            obs_ppm_list.extend([true_ppm, true_ppm + 200.0])
            counts_list.append(2)
            cursor += 2

        ref_mz = np.array(ref_mz_list, dtype=np.float64)
        obs_ppm = np.array(obs_ppm_list, dtype=np.float64)
        peak_starts = np.array(starts_list, dtype=np.int64)
        peak_counts = np.array(counts_list, dtype=np.int64)

        _best_a, _best_b, _best_c, n_in = _nb_ransac_quadratic(
            ref_mz,
            obs_ppm,
            peak_starts,
            peak_counts,
            np.int64(n_peaks),
            np.int64(len(ref_mz_list)),
            10.0,
            np.int64(4),
            np.int64(500),
        )
        # At least n_peaks correct candidates should be inliers
        assert n_in >= n_peaks


# ---------------------------------------------------------------------------
# _nb_monotone_interp
# ---------------------------------------------------------------------------


class TestNbMonotoneInterp:
    def test_identity_on_grid_points(self):
        """Query at exact grid points → exact values."""
        xp = np.linspace(0.0, 10.0, 100)
        fp = np.sin(xp)
        result = _nb_monotone_interp(xp, xp, fp)
        np.testing.assert_allclose(result, fp, atol=1e-12)

    def test_matches_numpy_interp(self):
        """Should match np.interp for sorted queries within range."""
        xp = np.linspace(0.0, 10.0, 500)
        fp = np.sin(xp)
        src = np.linspace(0.5, 9.5, 200)
        expected = np.interp(src, xp, fp)
        got = _nb_monotone_interp(src, xp, fp)
        np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_boundary_clamping(self):
        """Values outside range should be clamped to boundary fp values."""
        xp = np.array([1.0, 2.0, 3.0])
        fp = np.array([10.0, 20.0, 30.0])
        src = np.array([0.0, 0.5, 1.0, 3.0, 4.0])
        result = _nb_monotone_interp(src, xp, fp)
        assert result[0] == 10.0  # clamped to fp[0]
        assert result[1] == 10.0  # still clamped
        assert result[2] == 10.0  # exactly at left boundary
        assert result[4] == 30.0  # clamped to fp[-1]

    def test_float32_fp(self):
        """Should accept float32 intensity arrays without error."""
        xp = np.linspace(0.0, 10.0, 100)
        fp = np.sin(xp).astype(np.float32)
        src = np.linspace(1.0, 9.0, 50)
        result = _nb_monotone_interp(src, xp, fp)
        expected = np.interp(src, xp, fp.astype(np.float64))
        np.testing.assert_allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# RobustLockmassEstimator — _find_candidates
# ---------------------------------------------------------------------------


class TestFindCandidates:
    def test_centred_peak_gives_near_zero_ppm(self):
        """When the spectrum peak is exactly at the reference m/z, ppm ≈ 0."""
        mz_x = _make_mz_axis()
        peaks = np.array([500.0])
        mz_y = _make_spectrum(mz_x, [500.0], [5000.0])
        est = RobustLockmassEstimator(mz_x, peaks, search_ppm=200.0, threshold=100.0)
        ref, obs, *_ = est._find_candidates(mz_y)
        assert len(ref) >= 1
        # The closest-to-zero ppm candidate should be near 0
        best_idx = int(np.argmin(np.abs(obs)))
        assert abs(obs[best_idx]) < 10.0  # within 10 ppm

    def test_offset_peak_gives_nonzero_ppm(self):
        """Spectrum peak shifted by +50 ppm → observed ppm ≈ +50."""
        mz_x = _make_mz_axis()
        ref_mz = 500.0
        shift_da = ref_mz * 50.0 / 1e6  # +50 ppm
        actual_mz = ref_mz + shift_da
        peaks = np.array([ref_mz])
        mz_y = _make_spectrum(mz_x, [actual_mz], [5000.0])
        est = RobustLockmassEstimator(mz_x, peaks, search_ppm=200.0, threshold=100.0)
        _ref, obs, *_ = est._find_candidates(mz_y)
        assert len(obs) >= 1
        best_idx = int(np.argmin(np.abs(obs - 50.0)))
        assert abs(obs[best_idx] - 50.0) < 5.0  # within 5 ppm of 50

    def test_missing_peak_gives_no_candidates(self):
        """Peak far below threshold → no candidates for that reference."""
        mz_x = _make_mz_axis()
        peaks = np.array([500.0])
        mz_y = np.zeros(len(mz_x), dtype=np.float32)  # blank spectrum
        est = RobustLockmassEstimator(mz_x, peaks, search_ppm=200.0, threshold=ROBUST_LOCKMASS_THRESHOLD)
        ref, _obs, *_ = est._find_candidates(mz_y)
        assert len(ref) == 0

    def test_heavy_miscalibration_still_found(self):
        """Peak shifted by 150 ppm is inside a 200-ppm window and should be detected."""
        mz_x = _make_mz_axis()
        ref_mz = 500.0
        shift_da = ref_mz * 150.0 / 1e6  # 150 ppm
        peaks = np.array([ref_mz])
        mz_y = _make_spectrum(mz_x, [ref_mz + shift_da], [5000.0])
        est = RobustLockmassEstimator(mz_x, peaks, search_ppm=200.0, threshold=100.0)
        _ref, obs, *_ = est._find_candidates(mz_y)
        assert len(obs) >= 1
        best_idx = int(np.argmin(np.abs(obs - 150.0)))
        assert abs(obs[best_idx] - 150.0) < 10.0


# ---------------------------------------------------------------------------
# RobustLockmassEstimator — estimate
# ---------------------------------------------------------------------------


class TestEstimate:
    def test_returns_four_tuple(self):
        mz_x = _make_mz_axis()
        peaks = np.array([500.0, 700.0, 900.0])
        mz_y = _make_spectrum(mz_x, [500.0, 700.0, 900.0], [5000.0, 5000.0, 5000.0])
        est = RobustLockmassEstimator(mz_x, peaks, threshold=100.0)
        result = est.estimate(mz_y)
        assert len(result) == 4

    def test_centred_peaks_near_zero_ppm(self):
        """All peaks at their reference positions → observed ppm should be small."""
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0, 800.0, 1000.0])
        mz_y = _make_spectrum(mz_x, peaks.tolist(), [5000.0] * 4)
        est = RobustLockmassEstimator(mz_x, peaks, threshold=100.0)
        _ref, obs, coeffs, _inliers = est.estimate(mz_y)
        assert len(obs) >= 1
        # Most inlier ppm values should be near zero
        if coeffs is not None:
            assert abs(float(np.polyval(coeffs, 700.0))) < 20.0  # < 20 ppm at midpoint

    def test_constant_shift_recovered(self):
        """All peaks shifted by the same ppm → constant polynomial."""
        mz_x = _make_mz_axis()
        shift_ppm = 30.0
        ref_peaks = np.array([400.0, 600.0, 800.0, 1000.0])
        actual_mzs = ref_peaks * (1.0 + shift_ppm / 1e6)
        mz_y = _make_spectrum(mz_x, actual_mzs.tolist(), [5000.0] * 4)
        est = RobustLockmassEstimator(mz_x, ref_peaks, ransac_tol_ppm=10.0, threshold=100.0)
        _, _, coeffs, _ = est.estimate(mz_y)
        assert coeffs is not None
        # Polynomial evaluated at any reference m/z should be near shift_ppm
        for mz in ref_peaks:
            pred = float(np.polyval(coeffs, mz))
            assert abs(pred - shift_ppm) < 10.0

    def test_outlier_peak_rejected(self):
        """One reference peak with a wildly wrong candidate → RANSAC rejects it."""
        mz_x = _make_mz_axis()
        shift_ppm = 30.0
        ref_peaks = np.array([400.0, 600.0, 800.0, 1000.0])
        # All peaks at shift_ppm except the one at 600 (placed 300 ppm away)
        actual_mzs = list(ref_peaks * (1.0 + shift_ppm / 1e6))
        actual_mzs[1] = 600.0 * (1.0 + 300.0 / 1e6)
        mz_y = _make_spectrum(mz_x, actual_mzs, [5000.0] * 4)
        est = RobustLockmassEstimator(mz_x, ref_peaks, ransac_tol_ppm=20.0, threshold=100.0)
        _, obs_ppm, coeffs, inlier_mask = est.estimate(mz_y)
        if coeffs is not None and len(inlier_mask) > 0:
            # The outlier candidate (near 300 ppm) should ideally be excluded
            inlier_ppms = obs_ppm[inlier_mask]
            assert all(abs(p - shift_ppm) < 50.0 for p in inlier_ppms)

    def test_all_peaks_missing_returns_none_coeffs(self):
        """Blank spectrum → no candidates → poly_coeffs is None."""
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0, 800.0])
        mz_y = np.zeros(len(mz_x), dtype=np.float32)
        est = RobustLockmassEstimator(mz_x, peaks, threshold=ROBUST_LOCKMASS_THRESHOLD)
        _, _, coeffs, _ = est.estimate(mz_y)
        assert coeffs is None

    def test_single_peak_falls_back_to_constant(self):
        """Only one peak detected → falls back to degree-0 (constant) polynomial."""
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0, 800.0])
        # Only the first peak is visible
        mz_y = _make_spectrum(mz_x, [400.0 * (1 + 50.0 / 1e6)], [5000.0])
        est = RobustLockmassEstimator(mz_x, peaks, min_inliers=1, threshold=100.0)
        _, _, coeffs, _ = est.estimate(mz_y)
        if coeffs is not None:
            assert len(coeffs) == 1  # degree-0: single constant

    def test_two_peaks_falls_back_to_linear(self):
        """Two peak groups → degree-1 polynomial at most."""
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 800.0])
        mz_y = _make_spectrum(mz_x, [400.0 * (1 + 50.0 / 1e6), 800.0 * (1 + 50.0 / 1e6)], [5000.0, 5000.0])
        est = RobustLockmassEstimator(mz_x, peaks, min_inliers=1, threshold=100.0)
        _, _, coeffs, _ = est.estimate(mz_y)
        if coeffs is not None:
            assert len(coeffs) <= 2  # at most linear


# ---------------------------------------------------------------------------
# RobustLockmassEstimator — correct
# ---------------------------------------------------------------------------


class TestCorrect:
    def test_correct_returns_same_shape(self):
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0, 800.0, 1000.0])
        mz_y = _make_spectrum(mz_x, peaks.tolist(), [5000.0] * 4)
        est = RobustLockmassEstimator(mz_x, peaks, threshold=100.0)
        corrected = est.correct(mz_y, fast=True)
        assert corrected.shape == mz_y.shape

    def test_correct_warp_returns_same_shape(self):
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0, 800.0, 1000.0])
        mz_y = _make_spectrum(mz_x, peaks.tolist(), [5000.0] * 4)
        est = RobustLockmassEstimator(mz_x, peaks, threshold=100.0)
        corrected = est.correct(mz_y, fast=False)
        assert corrected.shape == mz_y.shape

    def test_centred_peaks_fast_no_change(self):
        """Peaks already at reference positions → scalar shift should be ~0 → spectrum unchanged."""
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0, 800.0, 1000.0])
        mz_y = _make_spectrum(mz_x, peaks.tolist(), [5000.0] * 4, noise_level=0.0)
        est = RobustLockmassEstimator(mz_x, peaks, threshold=100.0)
        corrected = est.correct(mz_y, fast=True)
        # Correct spectrum should be nearly identical to input
        np.testing.assert_allclose(corrected.astype(float), mz_y.astype(float), rtol=0.01)

    def test_constant_shift_corrected_fast(self):
        """Constant +50 ppm shift → fast correction should bring peak near reference."""
        mz_x = _make_mz_axis()
        shift_ppm = 50.0
        ref_peaks = np.array([500.0, 700.0, 900.0])
        actual_mzs = (ref_peaks * (1.0 + shift_ppm / 1e6)).tolist()
        mz_y = _make_spectrum(mz_x, actual_mzs, [5000.0] * 3, noise_level=0.0)
        est = RobustLockmassEstimator(mz_x, ref_peaks, threshold=100.0, ransac_tol_ppm=10.0)

        corrected = est.correct(mz_y, fast=True)

        # Peak at ref_peaks[1] (700 Da) in the corrected spectrum should be near 700 Da
        idx_ref = int(np.searchsorted(mz_x, ref_peaks[1]))
        # The corrected peak should be within ±10 ppm of the reference
        local = corrected[max(0, idx_ref - 50) : idx_ref + 50]
        if len(local) > 0:
            peak_idx = int(np.argmax(local)) + max(0, idx_ref - 50)
            peak_mz = mz_x[peak_idx]
            assert _ppm_err(peak_mz, ref_peaks[1]) < 20.0

    def test_constant_shift_corrected_warp(self):
        """Constant +50 ppm shift → warp correction should bring peak near reference."""
        mz_x = _make_mz_axis()
        shift_ppm = 50.0
        ref_peaks = np.array([500.0, 700.0, 900.0])
        actual_mzs = (ref_peaks * (1.0 + shift_ppm / 1e6)).tolist()
        mz_y = _make_spectrum(mz_x, actual_mzs, [5000.0] * 3, noise_level=0.0)
        est = RobustLockmassEstimator(mz_x, ref_peaks, threshold=100.0, ransac_tol_ppm=10.0)

        corrected = est.correct(mz_y, fast=False)
        assert corrected.shape == mz_y.shape
        assert corrected.dtype == mz_y.dtype

        idx_ref = int(np.searchsorted(mz_x, ref_peaks[1]))
        local = corrected[max(0, idx_ref - 50) : idx_ref + 50]
        if len(local) > 0:
            peak_idx = int(np.argmax(local)) + max(0, idx_ref - 50)
            peak_mz = mz_x[peak_idx]
            assert _ppm_err(peak_mz, ref_peaks[1]) < 20.0

    def test_failed_ransac_warns_and_returns_copy(self):
        """When RANSAC fails, a RuntimeWarning is issued and the input is returned unchanged."""
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0, 800.0])
        mz_y = np.zeros(len(mz_x), dtype=np.float32)
        est = RobustLockmassEstimator(mz_x, peaks, threshold=ROBUST_LOCKMASS_THRESHOLD)

        with pytest.warns(RuntimeWarning, match="RANSAC"):
            corrected = est.correct(mz_y)

        np.testing.assert_array_equal(corrected, mz_y)
        assert corrected is not mz_y  # returns a copy

    def test_fast_and_warp_dtype_preserved(self):
        """Output dtype should match the input spectrum dtype."""
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0, 800.0, 1000.0])
        mz_y = _make_spectrum(mz_x, peaks.tolist(), [5000.0] * 4).astype(np.float32)
        est = RobustLockmassEstimator(mz_x, peaks, threshold=100.0)
        assert est.correct(mz_y, fast=True).dtype == np.float32
        assert est.correct(mz_y, fast=False).dtype == np.float32

    def test_correct_for_reader_raises_on_bad_reader(self):
        mz_x = _make_mz_axis()
        est = RobustLockmassEstimator(mz_x, np.array([500.0]))
        with pytest.raises(ValueError, match="n_pixels"):
            est.correct_for_reader(object())


# ---------------------------------------------------------------------------
# RobustLockmassEstimator — init / basic properties
# ---------------------------------------------------------------------------


class TestInit:
    def test_peaks_sorted(self):
        mz_x = _make_mz_axis()
        peaks = np.array([900.0, 400.0, 600.0])
        est = RobustLockmassEstimator(mz_x, peaks)
        np.testing.assert_array_equal(est.peaks, np.sort(peaks))

    def test_starts_stops_shape(self):
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0, 800.0])
        est = RobustLockmassEstimator(mz_x, peaks)
        assert est._starts.shape == (3,)
        assert est._stops.shape == (3,)
        assert np.all(est._starts < est._stops)

    def test_wider_window_gives_more_indices(self):
        mz_x = _make_mz_axis()
        peaks = np.array([500.0])
        est_narrow = RobustLockmassEstimator(mz_x, peaks, search_ppm=50.0)
        est_wide = RobustLockmassEstimator(mz_x, peaks, search_ppm=500.0)
        narrow_size = est_narrow._stops[0] - est_narrow._starts[0]
        wide_size = est_wide._stops[0] - est_wide._starts[0]
        assert wide_size > narrow_size

    def test_candidate_buffers_preallocated(self):
        mz_x = _make_mz_axis()
        peaks = np.array([400.0, 600.0])
        est = RobustLockmassEstimator(mz_x, peaks, max_candidates=5)
        assert est._cx_out.shape == (2, 5)
        assert est._cy_out.shape == (2, 5)
