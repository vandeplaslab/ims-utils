"""Robust mass correction based on multi-peak lockmass with RANSAC polynomial fitting."""

from __future__ import annotations

import typing as ty
import warnings

import numba as nb
import numpy as np
from koyo.utilities import find_nearest_index, get_array_mask

from ims_utils.lockmass import LOCKMASS_THRESHOLD, fast_roll

if ty.TYPE_CHECKING:
    try:
        from imzy import BaseReader  # type: ignore[import]
    except ImportError:  # pragma: no cover
        BaseReader = None  # type: ignore[assignment]


# Default threshold for peak detection (inherited from lockmass module)
ROBUST_LOCKMASS_THRESHOLD: float = LOCKMASS_THRESHOLD

# Default maximum candidates collected per reference peak window
_MAX_CANDIDATES: int = 8


@nb.njit(fastmath=True, cache=True)
def _nb_find_all_candidates(
    mz_y: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    threshold: nb.float64,
    centroid_frac: nb.float64,
    cx_out: np.ndarray,
    cy_out: np.ndarray,
    n_found_out: np.ndarray,
) -> None:
    """Find all local-maximum peak candidates within each reference search window.

    For each window, every local maximum above the intensity threshold is recorded
    with its sub-pixel parabolic centroid position. Multiple candidates per window
    allow RANSAC to choose the correct match even when the spectrum is heavily
    miscalibrated.

    Parameters
    ----------
    mz_y : np.ndarray
        Full-spectrum intensity array (float32 or float64).
    starts, stops : int64[n_peaks]
        Half-open slice bounds of each search window in ``mz_y``.
    threshold : float
        Absolute minimum intensity for a candidate peak.
    centroid_frac : float
        Secondary threshold as a fraction of the window mean intensity.
    cx_out : float64[n_peaks, max_candidates]
        Output: sub-pixel centroid indices (global coordinates).
        Slots beyond ``n_found_out[j]`` are left uninitialized.
    cy_out : float64[n_peaks, max_candidates]
        Output: estimated peak intensity at each centroid.
    n_found_out : int64[n_peaks]
        Output: number of valid candidates stored per window.
    """
    n_peaks = len(starts)
    max_cands = cx_out.shape[1]

    for j in range(n_peaks):
        n_found_out[j] = 0
        s = starts[j]
        e = stops[j]
        n = e - s
        if n < 3:
            continue

        y = mz_y[s:e]

        # Window-level secondary intensity threshold
        win_sum = 0.0
        for k in range(n):
            win_sum += nb.float64(y[k])
        win_thresh = threshold
        secondary = (win_sum / n) * centroid_frac
        if secondary > win_thresh:
            win_thresh = secondary

        count = 0
        for i in range(1, n - 1):
            if count >= max_cands:
                break

            yi = nb.float64(y[i])
            y0 = nb.float64(y[i - 1])
            y2 = nb.float64(y[i + 1])

            if yi < win_thresh:
                continue

            # Plateau-aware local maximum: must be >= both neighbours,
            # with at least one strict inequality
            if yi < y0 or yi < y2:
                continue
            if yi == y0 and yi == y2:
                continue  # true flat region, not a peak

            # Parabolic centroid for uniform index spacing (step = 1)
            denom = y0 - 2.0 * yi + y2
            if denom >= 0.0:
                # Not a downward-opening parabola — skip
                continue

            cx_offset = 0.5 * (y0 - y2) / denom
            cx = nb.float64(s + i) + cx_offset
            cy = yi - 0.25 * (y0 - y2) * cx_offset

            if np.isnan(cx) or np.isnan(cy) or cy <= 0.0:
                continue

            cx_out[j, count] = cx
            cy_out[j, count] = cy
            count += 1

        n_found_out[j] = count


@nb.njit(fastmath=True, cache=True)
def _nb_monotone_interp(
    src: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    """Linear interpolation assuming both *src* and *xp* are monotonically increasing.

    An O(N) scan replaces the O(N log N) binary search in ``np.interp``, which is
    valid when both the source positions and the reference axis are sorted.

    Parameters
    ----------
    src : float64[N]
        Query positions (must be monotonically non-decreasing).
    xp : float64[N]
        Reference x-axis (must be monotonically increasing).
    fp : float64[N] or float32[N]
        Reference y-values corresponding to ``xp``.

    Returns
    -------
    np.ndarray (float64[N])
        Interpolated values at ``src``. Values outside ``[xp[0], xp[-1]]`` are
        clamped to the boundary values of ``fp``.
    """
    N = len(src)
    M = len(xp)
    out = np.empty(N, dtype=np.float64)
    j = 0

    for i in range(N):
        xi = src[i]

        # Advance j until xp[j+1] >= xi
        while j < M - 2 and xp[j + 1] < xi:
            j += 1

        if xi <= xp[0]:
            out[i] = nb.float64(fp[0])
        elif xi >= xp[M - 1]:
            out[i] = nb.float64(fp[M - 1])
        else:
            dxp = xp[j + 1] - xp[j]
            if dxp == 0.0:
                out[i] = nb.float64(fp[j])
            else:
                t = (xi - xp[j]) / dxp
                out[i] = nb.float64(fp[j]) + t * (nb.float64(fp[j + 1]) - nb.float64(fp[j]))

    return out


@nb.njit(fastmath=True, cache=True)
def _nb_ransac_quadratic(
    ref_mz: np.ndarray,
    obs_ppm: np.ndarray,
    peak_starts: np.ndarray,
    peak_counts: np.ndarray,
    n_peaks: nb.int64,
    n_total: nb.int64,
    tol_ppm: nb.float64,
    min_inliers: nb.int64,
    max_iter: nb.int64,
) -> tuple[nb.float64, nb.float64, nb.float64, nb.int64]:
    """RANSAC fit of a degree-2 polynomial ``shift_ppm(mz)`` using an analytic 3-point solve.

    Each RANSAC iteration samples **one candidate from each of 3 distinct reference
    peaks**, guaranteeing the polynomial is constrained at 3 different m/z values
    rather than fitting a degenerate model to 3 candidates from the same peak.

    Parameters
    ----------
    ref_mz : float64[n_total]
        Reference m/z for each candidate, grouped contiguously by peak.
    obs_ppm : float64[n_total]
        Observed ppm shift for each candidate.
    peak_starts : int64[n_peaks]
        Start index of each peak group in the flat arrays.
    peak_counts : int64[n_peaks]
        Number of candidates in each peak group (all >= 1).
    n_peaks : int
        Number of peaks with at least one candidate.
    n_total : int
        Total number of candidates.
    tol_ppm : float
        Inlier threshold in ppm.
    min_inliers : int
        Minimum inliers required to accept a model.
    max_iter : int
        Number of RANSAC iterations.

    Returns
    -------
    (a, b, c, n_best_inliers)
        Quadratic coefficients ``a*mz^2 + b*mz + c`` and inlier count.
        Returns ``(nan, nan, nan, 0)`` if fitting fails.
    """
    best_a = np.nan
    best_b = np.nan
    best_c = np.nan
    best_n = nb.int64(0)

    if n_peaks < 3 or n_total < 3:
        return best_a, best_b, best_c, best_n

    for _ in range(max_iter):
        # Sample 3 distinct peak group indices (rejection sampling — fast for small n_peaks)
        p0 = np.random.randint(0, n_peaks)
        p1 = np.random.randint(0, n_peaks)
        while p1 == p0:
            p1 = np.random.randint(0, n_peaks)
        p2 = np.random.randint(0, n_peaks)
        while p2 == p0 or p2 == p1:
            p2 = np.random.randint(0, n_peaks)

        # One random candidate from each selected peak
        c0 = peak_starts[p0] + np.random.randint(0, peak_counts[p0])
        c1 = peak_starts[p1] + np.random.randint(0, peak_counts[p1])
        c2 = peak_starts[p2] + np.random.randint(0, peak_counts[p2])

        x0, y0 = ref_mz[c0], obs_ppm[c0]
        x1, y1 = ref_mz[c1], obs_ppm[c1]
        x2, y2 = ref_mz[c2], obs_ppm[c2]

        # Analytic quadratic through 3 points via divided differences:
        #   a = [(y2-y1)/(x2-x1) - (y1-y0)/(x1-x0)] / (x2-x0)
        #   b = (y1-y0)/(x1-x0) - a*(x0+x1)
        #   c = y0 - a*x0^2 - b*x0
        d01 = x1 - x0
        d12 = x2 - x1
        d02 = x2 - x0
        if abs(d01) < 1e-6 or abs(d12) < 1e-6 or abs(d02) < 1e-6:
            continue  # coincident x-values → skip

        r01 = (y1 - y0) / d01
        r12 = (y2 - y1) / d12
        a = (r12 - r01) / d02
        b = r01 - a * (x0 + x1)
        c = y0 - a * x0 * x0 - b * x0

        # Count inliers across ALL candidates
        n_in = nb.int64(0)
        for k in range(n_total):
            xk = ref_mz[k]
            pred = a * xk * xk + b * xk + c
            residual = obs_ppm[k] - pred
            if residual < 0.0:
                residual = -residual
            if residual < tol_ppm:
                n_in += 1

        if n_in > best_n:
            best_n = n_in
            best_a = a
            best_b = b
            best_c = c

    if best_n < min_inliers:
        return np.nan, np.nan, np.nan, nb.int64(0)
    return best_a, best_b, best_c, best_n


class RobustLockmassEstimator:
    """Robust mass correction using multi-peak RANSAC polynomial fitting.

    Unlike the simple ``MaximumIntensityLockmassEstimator`` or
    ``WeightedIntensityLockmassEstimator``, which apply a uniform single-point
    shift, this estimator:

    1. Searches each reference peak window with a **ppm-scaled width** so it still
       finds peaks in badly miscalibrated spectra.
    2. Collects **all local maxima** (up to ``max_candidates``) within each window,
       not just the tallest one.
    3. Fits a degree-2 polynomial ``shift_ppm(mz)`` using **RANSAC** across all
       candidates, automatically rejecting missing or misidentified peaks as
       outliers.
    4. Applies the correction either as a fast scalar roll (``fast=True``) or as a
       full interpolated spectral warp (``fast=False``) that accounts for
       non-linear drift across the m/z axis.

    The ppm convention is: positive ``shift_ppm`` means the measured peak is at
    **higher** m/z than the reference, so the spectrum must shift **left** to
    correct.

    Parameters
    ----------
    mz_x : np.ndarray
        Mass-to-charge axis of the profile spectrum.
    peaks : np.ndarray
        Reference peak masses (same units as ``mz_x``).
    search_ppm : float
        Half-width of the search window around each reference peak in ppm.
        Larger values tolerate worse miscalibration at the cost of more false
        candidates.
    centroid_frac : float
        Fraction of the window mean intensity used as a secondary peak threshold.
    ransac_tol_ppm : float
        Inlier threshold for RANSAC in ppm.
    min_inliers : int
        Minimum inlier candidates required for the RANSAC fit to be accepted.
    max_candidates : int
        Maximum number of peak candidates collected per reference window.
    poly_degree : int
        Requested polynomial degree for the shift model (2 = quadratic).
        Automatically reduced when fewer candidates are available.
    max_iter : int
        Number of RANSAC iterations.
    threshold : float
        Absolute intensity threshold below which peaks are ignored.
    """

    def __init__(
        self,
        mz_x: np.ndarray,
        peaks: np.ndarray,
        search_ppm: float = 200.0,
        centroid_frac: float = 0.25,
        ransac_tol_ppm: float = 5.0,
        min_inliers: int = 2,
        max_candidates: int = _MAX_CANDIDATES,
        poly_degree: int = 2,
        max_iter: int = 200,
        threshold: float = ROBUST_LOCKMASS_THRESHOLD,
    ) -> None:
        self.mz_x = mz_x
        self.peaks = np.sort(np.asarray(peaks, dtype=np.float64))
        self.n_peaks = len(self.peaks)
        self.search_ppm = float(search_ppm)
        self.centroid_frac = float(centroid_frac)
        self.ransac_tol_ppm = float(ransac_tol_ppm)
        self.min_inliers = int(min_inliers)
        self.max_candidates = int(max_candidates)
        self.poly_degree = int(poly_degree)
        self.max_iter = int(max_iter)
        self.threshold = float(threshold)

        # Compute ppm-scaled search windows for each reference peak
        n_mz = len(mz_x)
        self._starts = np.empty(self.n_peaks, dtype=np.int64)
        self._stops = np.empty(self.n_peaks, dtype=np.int64)
        for j, p in enumerate(self.peaks):
            w = p * self.search_ppm / 1e6
            mask = get_array_mask(mz_x, p - w, p + w)
            where = np.where(mask)[0]
            if len(where) == 0:
                # Reference peak outside the m/z range: use the nearest single index
                idx = int(find_nearest_index(mz_x, p))
                self._starts[j] = max(0, idx - 1)
                self._stops[j] = min(n_mz, idx + 2)
            else:
                self._starts[j] = int(where[0])
                self._stops[j] = int(where[-1] + 1)

        self._ref_mz: np.ndarray = self.peaks  # already float64
        self._mz_x_f64: np.ndarray = mz_x.astype(np.float64)
        self._mean_spacing: float = float(np.mean(np.diff(mz_x)))

        # Preallocated candidate buffers — reused across calls to avoid per-call allocation
        self._cx_out = np.empty((self.n_peaks, self.max_candidates), dtype=np.float64)
        self._cy_out = np.empty((self.n_peaks, self.max_candidates), dtype=np.float64)
        self._n_found = np.zeros(self.n_peaks, dtype=np.int64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_candidates(self, mz_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the numba candidate finder and convert results to flat arrays.

        Returns
        -------
        ref_mz_flat : float64[N]
            Reference m/z for each candidate.
        obs_ppm_flat : float64[N]
            Observed ppm shift for each candidate.
        peak_starts : int64[P]
            Start index of each peak group in the flat arrays,
            where P = number of reference peaks that had at least one candidate.
        peak_counts : int64[P]
            Number of candidates per group.
        """
        _nb_find_all_candidates(
            mz_y,
            self._starts,
            self._stops,
            self.threshold,
            self.centroid_frac,
            self._cx_out,
            self._cy_out,
            self._n_found,
        )

        mz_x64 = self._mz_x_f64
        n_mz = len(mz_x64)

        ref_mz_list: list[float] = []
        obs_ppm_list: list[float] = []
        peak_starts_list: list[int] = []
        peak_counts_list: list[int] = []

        for j in range(self.n_peaks):
            n = int(self._n_found[j])
            if n == 0:
                continue
            ref_mz = float(self._ref_mz[j])
            group_start = len(ref_mz_list)

            for k in range(n):
                cx = float(self._cx_out[j, k])
                # Convert sub-pixel index → m/z by linear interpolation on mz_x
                ci = int(cx)
                ci = max(0, min(n_mz - 2, ci))
                frac = cx - float(ci)
                obs_mz = mz_x64[ci] + frac * (mz_x64[ci + 1] - mz_x64[ci])
                ppm = (obs_mz - ref_mz) / ref_mz * 1.0e6
                ref_mz_list.append(ref_mz)
                obs_ppm_list.append(ppm)

            added = len(ref_mz_list) - group_start
            if added > 0:
                peak_starts_list.append(group_start)
                peak_counts_list.append(added)

        if not ref_mz_list:
            return (
                np.empty(0, np.float64),
                np.empty(0, np.float64),
                np.empty(0, np.int64),
                np.empty(0, np.int64),
            )

        return (
            np.array(ref_mz_list, dtype=np.float64),
            np.array(obs_ppm_list, dtype=np.float64),
            np.array(peak_starts_list, dtype=np.int64),
            np.array(peak_counts_list, dtype=np.int64),
        )

    def _fit_polynomial(
        self,
        ref_mz_flat: np.ndarray,
        obs_ppm_flat: np.ndarray,
        peak_starts: np.ndarray,
        peak_counts: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Fit a robust polynomial shift model to the candidate pool.

        Returns
        -------
        poly_coeffs : np.ndarray or None
            Polynomial coefficients (highest degree first, as from ``np.polyfit``),
            or ``None`` if fitting fails.
        inlier_mask : bool[N]
            Boolean mask over the candidate arrays indicating RANSAC inliers.
        """
        n_total = len(ref_mz_flat)
        n_groups = len(peak_starts)

        if n_total == 0:
            return None, np.zeros(0, dtype=bool)

        effective_degree = min(self.poly_degree, n_groups - 1, n_total - 1)

        # --- Degree 2: numba RANSAC quadratic ---
        if effective_degree >= 2:
            a, b, c, n_inliers = _nb_ransac_quadratic(
                ref_mz_flat,
                obs_ppm_flat,
                peak_starts,
                peak_counts,
                np.int64(n_groups),
                np.int64(n_total),
                self.ransac_tol_ppm,
                np.int64(self.min_inliers),
                np.int64(self.max_iter),
            )
            if n_inliers >= self.min_inliers and not np.isnan(a):
                inlier_mask = np.abs(a * ref_mz_flat**2 + b * ref_mz_flat + c - obs_ppm_flat) < self.ransac_tol_ppm
                n_in = int(inlier_mask.sum())
                poly_coeffs = np.polyfit(
                    ref_mz_flat[inlier_mask],
                    obs_ppm_flat[inlier_mask],
                    min(2, n_in - 1),
                )
                return poly_coeffs, inlier_mask
            effective_degree = 1

        # --- Degree 1: linear RANSAC or direct fit ---
        if effective_degree >= 1:
            if n_total >= 4:
                best_coeffs: np.ndarray | None = None
                best_n = 0
                n_iter = min(self.max_iter, n_total * (n_total - 1))
                for _ in range(n_iter):
                    i0 = int(np.random.randint(0, n_total))
                    i1 = int(np.random.randint(0, n_total - 1))
                    if i1 >= i0:
                        i1 += 1
                    dx = ref_mz_flat[i1] - ref_mz_flat[i0]
                    if abs(dx) < 1e-6:
                        continue
                    slope = (obs_ppm_flat[i1] - obs_ppm_flat[i0]) / dx
                    intercept = obs_ppm_flat[i0] - slope * ref_mz_flat[i0]
                    resid = np.abs(obs_ppm_flat - (slope * ref_mz_flat + intercept))
                    n_in = int((resid < self.ransac_tol_ppm).sum())
                    if n_in > best_n:
                        best_n = n_in
                        best_coeffs = np.array([slope, intercept])
                if best_coeffs is not None and best_n >= self.min_inliers:
                    inlier_mask = np.abs(obs_ppm_flat - np.polyval(best_coeffs, ref_mz_flat)) < self.ransac_tol_ppm
                    poly_coeffs = np.polyfit(ref_mz_flat[inlier_mask], obs_ppm_flat[inlier_mask], 1)
                    return poly_coeffs, inlier_mask
            else:
                poly_coeffs = np.polyfit(ref_mz_flat, obs_ppm_flat, 1)
                inlier_mask = np.abs(obs_ppm_flat - np.polyval(poly_coeffs, ref_mz_flat)) < self.ransac_tol_ppm
                if int(inlier_mask.sum()) >= self.min_inliers:
                    return poly_coeffs, inlier_mask
            effective_degree = 0

        # --- Degree 0: constant (robust median) ---
        median_ppm = float(np.median(obs_ppm_flat))
        inlier_mask = np.abs(obs_ppm_flat - median_ppm) < self.ransac_tol_ppm
        n_in = int(inlier_mask.sum())
        if n_in >= self.min_inliers:
            poly_coeffs = np.array([float(np.median(obs_ppm_flat[inlier_mask]))])
            return poly_coeffs, inlier_mask

        return None, np.zeros(n_total, dtype=bool)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, mz_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """Estimate per-candidate ppm shifts and fit a robust polynomial model.

        Parameters
        ----------
        mz_y : np.ndarray
            Intensity values of the spectrum.

        Returns
        -------
        ref_mz_flat : np.ndarray (float64)
            Reference m/z for each candidate found.
        obs_ppm_flat : np.ndarray (float64)
            Observed ppm shift for each candidate.
        poly_coeffs : np.ndarray or None
            Fitted polynomial coefficients (highest degree first, as from
            ``np.polyfit``), or ``None`` if RANSAC fitting failed.
        inlier_mask : np.ndarray (bool)
            Boolean mask indicating which candidates are RANSAC inliers.
        """
        ref_mz_flat, obs_ppm_flat, peak_starts, peak_counts = self._find_candidates(mz_y)
        poly_coeffs, inlier_mask = self._fit_polynomial(ref_mz_flat, obs_ppm_flat, peak_starts, peak_counts)
        return ref_mz_flat, obs_ppm_flat, poly_coeffs, inlier_mask

    def correct(self, mz_y: np.ndarray, fast: bool = True) -> np.ndarray:
        """Estimate shifts, fit a RANSAC polynomial model, and apply the correction.

        Parameters
        ----------
        mz_y : np.ndarray
            Intensity values of the spectrum.
        fast : bool
            If ``True`` (default), evaluate the polynomial at the geometric mean
            of the reference peaks and apply a single scalar shift via
            ``fast_roll``. Fastest option; equivalent to the original lockmass
            approach but with a robust polynomial-fitted shift.

            If ``False``, evaluate the polynomial at every m/z point and apply a
            continuous spectral warp via monotone linear interpolation. More
            accurate for spectra spanning a wide m/z range with significant
            non-linear drift.

        Returns
        -------
        np.ndarray
            Corrected spectrum on the original m/z axis. A copy of the input
            spectrum is returned when RANSAC fitting fails.
        """
        _, _, poly_coeffs, _ = self.estimate(mz_y)

        if poly_coeffs is None:
            warnings.warn(
                "RobustLockmassEstimator: RANSAC fitting failed "
                "(too few candidates or inliers); returning spectrum unchanged.",
                RuntimeWarning,
                stacklevel=2,
            )
            return mz_y.copy()

        if fast:
            mz_mean = float(np.mean(self._ref_mz))
            shift_ppm = float(np.polyval(poly_coeffs, mz_mean))
            da_shift = shift_ppm * mz_mean / 1.0e6
            index_shift = round(da_shift / self._mean_spacing)
            return fast_roll(mz_y, -index_shift)

        # Continuous warp: evaluate shift at every m/z point
        da_shift_arr = np.polyval(poly_coeffs, self._mz_x_f64) * self._mz_x_f64 / 1.0e6
        src = self._mz_x_f64 + da_shift_arr  # source positions in the measured spectrum
        corrected = _nb_monotone_interp(src, self._mz_x_f64, mz_y.astype(np.float64))
        return corrected.astype(mz_y.dtype)

    def correct_for_reader(self, reader: ty.Any, fast: bool = True, silent: bool = False) -> list[np.ndarray]:
        """Apply mass correction to all spectra in a reader.

        Parameters
        ----------
        reader : BaseReader
            Reader object with a ``spectra_iter(silent=...)`` method and
            ``n_pixels`` attribute.
        fast : bool
            Passed to ``correct``.
        silent : bool
            Whether to suppress the reader's progress output.

        Returns
        -------
        list of np.ndarray
            Corrected spectra, one per pixel, in iteration order.
        """
        if not hasattr(reader, "n_pixels"):
            raise ValueError("reader must have n_pixels attribute")
        if not hasattr(reader, "spectra_iter"):
            raise ValueError("reader must have spectra_iter attribute")
        return [self.correct(mz_y, fast=fast) for _, mz_y in reader.spectra_iter(silent=silent)]
