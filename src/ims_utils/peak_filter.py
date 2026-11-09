"""Filtering of peaks."""

from __future__ import annotations

import typing as ty

import numpy as np
import numpy.typing as npt
import pandas as pd
from koyo.timer import MeasureTimer
from loguru import logger

Polarity = ty.Literal["positive", "negative"]
GroupResult = list[tuple[list[int], list[float]]]


class PeakFilter:
    """Select peaks by m/z, matrix, Kendrick mass, mass defect, or isotope grouping.

    Filter methods return indices to keep, not indices to remove. The
    :attr:`indices` property intersects the keep sets from every filter that has
    been run; :attr:`removed_indices` returns the complement. Results are cached
    until :meth:`reset_filters` is called. The supplied m/z and intensity arrays
    are retained as inputs and are not changed by selecting peaks.

    Parameters
    ----------
    mzs : np.ndarray, optional
        One-dimensional m/z values, normally sorted ascending.
    intensities : np.ndarray, optional
        Intensities aligned with ``mzs``.
    polarity : Polarity, optional
        Ion polarity used when selecting matrix peak lists.
    """

    polarity: Polarity
    mzs: np.ndarray
    intensities: np.ndarray
    _groups: GroupResult | None = None

    _indices_by_mz: npt.NDArray | None = None
    _indices_by_mz_merge: npt.NDArray | None = None
    _indices_by_matrix: npt.NDArray | None = None
    _indices_by_kendrick_mass: npt.NDArray | None = None
    _indices_by_mass_defect: npt.NDArray | None = None

    def __init__(
        self,
        mzs: npt.NDArray | None = None,
        intensities: npt.NDArray | None = None,
        polarity: Polarity | None = None,
    ) -> None:
        """Initialize."""
        if mzs is not None:
            self.mzs = mzs
        if intensities is not None:
            self.intensities = intensities
        if polarity is not None:
            self.polarity = polarity

    def reset_filters(self) -> None:
        """Clear cached keep-index sets and isotope groups.

        Subsequent filter calls start from the full input peak list.
        """
        self._groups = None
        self._indices_by_mz = None
        self._indices_by_mz_merge = None
        self._indices_by_matrix = None
        self._indices_by_kendrick_mass = None
        self._indices_by_mass_defect = None
        self._groups_by_mz = None
        self._groups_by_mz_merge = None

    def filter_by_mz(  # type: ignore[override]
        self, mz_tol: float = 5e-3, mz_ppm: float = 0, max_in_group: int = 6, **_kwargs: ty.Any
    ) -> npt.NDArray:
        """Keep the most intense peak in each group at the requested m/z spacing.

        Parameters
        ----------
        mz_tol : float
            Absolute m/z tolerance in Da.
        mz_ppm : float
            Relative tolerance in ppm; when nonzero it replaces ``mz_tol``.
        max_in_group : int
            Maximum number of members in a group.

        Returns
        -------
        npt.NDArray
            Original indices of selected peaks. The indices are cached and
            combined with keep indices from other filters.
        """
        from ims_utils.utilities import find_groups_within_bounds

        # Validate inputs
        assert mz_tol is not None and mz_tol > 0, "M/z tolerance must be greater than 0"
        assert max_in_group is not None and max_in_group > 0, "Max in group must be greater than 0"
        if not any([mz_ppm, mz_tol]):
            raise ValueError("Either `mz_ppm` or `mz_tol` must be specified")

        with MeasureTimer() as timer:
            groups = find_groups_within_bounds(
                self.mzs, tolerance=mz_tol, ppm=mz_ppm, keep_singletons=True, max_in_group=max_in_group
            )
            self._indices_by_mz = filter_groups(groups, self.intensities)
            self._groups_by_mz = groups
            n = len(self.mzs) - len(self._indices_by_mz)
        logger.trace(
            f"[MZ] Found {len(self._indices_by_mz)} groups (reduction of {n}) in {timer()} (tol={mz_tol}; ppm={mz_ppm})"
        )
        return self._indices_by_mz

    def filter_by_mz_merge(
        self, mz_tol: float = 5e-3, mz_ppm: float = 0, max_in_group: int = 6, **_kwargs: ty.Any
    ) -> npt.NDArray:
        """Keep the most intense member of groups sharing an m/z value.

        The grouping uses zero target distance and the supplied absolute Da or
        ppm tolerance. Returns and caches original input indices to keep.
        """
        from ims_utils.utilities import find_groups_within_bounds

        # Validate inputs
        assert mz_tol is not None and mz_tol > 0, "M/z tolerance must be greater than 0"
        assert max_in_group is not None and max_in_group > 0, "Max in group must be greater than 0"
        if not any([mz_ppm, mz_tol]):
            raise ValueError("Either `mz_ppm` or `mz_tol` must be specified")
        with MeasureTimer() as timer:
            groups = find_groups_within_bounds(
                self.mzs, tolerance=mz_tol, ppm=mz_ppm, keep_singletons=True, distance=0, max_in_group=max_in_group
            )
            self._indices_by_mz_merge = filter_groups(groups, self.intensities)
            self._groups_by_mz_merge = groups
            n = len(self.mzs) - len(self._indices_by_mz_merge)
        logger.trace(
            f"[MERGE] Found {len(self._indices_by_mz_merge)} groups (reduction of {n}) in {timer()} (tol={mz_tol};"
            f" ppm={mz_ppm})"
        )
        return self._indices_by_mz_merge

    def filter_by_matrix(
        self,
        matrix: ty.Literal["dmaca", "mapca", "chca", "nedc"] | str,
        polarity: ty.Literal["auto", "positive", "negative"] | str = "auto",
        mz_tol: float = 5e-3,
        mz_ppm: float = 0,
        **_kwargs: ty.Any,
    ) -> npt.NDArray:
        """Remove peaks matching a reference matrix peak list.

        Returns and caches original input indices to keep. ``polarity="auto"``
        uses the polarity supplied to this object.
        """
        from koyo.utilities import find_nearest_index

        from ims_utils.assets import find_matrix
        from ims_utils.spectrum import ppm_error

        assert mz_tol is not None and mz_tol > 0, "M/z tolerance must be greater than 0"
        if not any([mz_ppm, mz_tol]):
            raise ValueError("Either `mz_ppm` or `mz_tol` must be specified")

        with MeasureTimer() as timer:
            matrix = matrix.lower()
            if ";" in matrix:
                matrix, polarity = matrix.split(";")

            polarity = polarity.lower()
            if polarity == "auto":
                polarity = self.polarity

            filename = find_matrix(matrix, polarity)
            if not filename or not filename.exists():
                raise FileNotFoundError(f"Matrix file not found: {filename}. Matrix={matrix}; Polarity={polarity}")
            peaklist = pd.read_csv(filename, sep=",")

            # find indices
            matrix_mzs = peaklist.mzs.values
            matrix_indices = find_nearest_index(self.mzs, matrix_mzs)
            if mz_ppm:
                ppm = abs(ppm_error(matrix_mzs, self.mzs[matrix_indices]))
                indices_to_remove = matrix_indices[ppm < mz_ppm]
            else:
                tol = np.abs(matrix_mzs - self.mzs[matrix_indices])
                indices_to_remove = matrix_indices[tol < mz_tol]
            self._indices_by_matrix = np.setdiff1d(np.arange(len(self.mzs)), indices_to_remove)
            n_idx = len(self._indices_by_matrix)  # type: ignore[arg-type]
            n = len(self.mzs) - n_idx
        logger.trace(
            f"[MATRIX] Found {n_idx:,} groups (reduction of {n:,}) in {timer()} (matrix={matrix}; polarity={polarity};"
            f" tol={mz_tol}; ppm={mz_ppm})"
        )
        return self._indices_by_matrix  # type: ignore[return-value]

    def filter_by_kendrick_mass(
        self, ref: str | float, filters: list[tuple[float, float, float, float]], **_kwargs: ty.Any
    ) -> npt.NDArray:
        """Exclude peaks falling inside the supplied Kendrick mass defect regions.

        Each region is ``(min_mz, max_mz, max_kmd, min_kmd)``. Returns and
        caches original input indices that remain after excluding those regions.
        """
        from ims_utils.kendrick import calculate_kendrick_mass_defect, filter_by_rules

        with MeasureTimer() as timer:
            _km, kmd = calculate_kendrick_mass_defect(self.mzs, ref)
            df = pd.DataFrame({"mzs": self.mzs, "kmd": kmd})
            filtered = filter_by_rules(df, filters)
            self._indices_by_kendrick_mass = filtered.index.to_numpy()
            n_idx = len(self._indices_by_kendrick_mass)  # type: ignore[arg-type]
            n = len(self.mzs) - n_idx
        logger.trace(
            f"[KENDRICK] Found {n_idx:,} groups (reduction of {n:,}) in {timer()} (ref={ref}; filters={filters})"
        )
        return self._indices_by_kendrick_mass  # type: ignore[return-value]

    def filter_by_mass_defect(self, min_defect: float = 0.0, max_defect: float = 1.0, **_kwargs: ty.Any) -> npt.NDArray:
        """Remove peaks whose fractional m/z lies in the inclusive defect range.

        Returns and caches original input indices outside the requested range.
        """
        with MeasureTimer() as timer:
            mass_defect, _nominal = np.modf(self.mzs)
            # mask = np.logical_and(min_defect <= mass_defect, mass_defect <= max_defect)
            mask = np.logical_and(mass_defect >= min_defect, mass_defect <= max_defect)
            self._indices_by_mass_defect = np.arange(len(self.mzs))[~mask]
            n_idx = len(self._indices_by_mass_defect)  # type: ignore[arg-type]
            n = len(self.mzs) - n_idx
        logger.trace(
            f"[MASS DEFECT] Found {n_idx:,} groups (reduction of {n:,}) in {timer()} (min={min_defect};"
            f" max={max_defect})"
        )
        return self._indices_by_mass_defect  # type: ignore[return-value]

    @property
    def removed_mzs(self) -> npt.NDArray:
        """Return m/z values corresponding to the current removed indices."""
        if self.indices is None:
            raise ValueError("No indices to filter")
        return self.mzs[self.removed_indices]  # type: ignore[no-any-return]

    @property
    def remaining_mzs(self) -> npt.NDArray:
        """Return m/z values corresponding to the current kept indices."""
        if self.indices is None:
            raise ValueError("No indices to filter")
        return self.mzs[self.indices]  # type: ignore[no-any-return]

    def find_index(self, mz: float) -> int:
        """Find index of a given m/z."""
        from koyo.utilities import find_nearest_index

        from ims_utils.spectrum import ppm_error

        index = find_nearest_index(self.mzs, mz)
        mz_ = self.mzs[index]
        ppm = abs(ppm_error(mz, mz_))
        if ppm:
            logger.warning(f"Found m/z {mz_} is not close to {mz} (ppm={ppm})")
        return index  # type: ignore[no-any-return]

    @property
    def indices(self) -> npt.NDArray:
        """Return original indices kept by the intersection of active filters."""
        if not any(
            [
                self._indices_by_mz is not None,
                self._indices_by_mz_merge is not None,
                self._indices_by_matrix is not None,
                self._indices_by_kendrick_mass is not None,
                self._indices_by_mass_defect is not None,
            ]
        ):
            raise ValueError("No indices to filter")

        all_indices = np.arange(len(self.mzs))
        indices_to_remove = []
        for indices_ in [
            self._indices_by_mz,
            self._indices_by_mz_merge,
            self._indices_by_matrix,
            self._indices_by_kendrick_mass,
            self._indices_by_mass_defect,
        ]:
            if indices_ is not None:
                indices_to_remove.append(np.setdiff1d(all_indices, indices_))
        indices_to_remove = np.unique(np.concatenate(indices_to_remove))  # type: ignore[no-any-return]
        return np.setdiff1d(all_indices, indices_to_remove)

    @property
    def removed_indices(self) -> npt.NDArray:
        """Return original indices excluded by the active filters."""
        if self.indices is None:
            raise ValueError("No indices to filter")
        return self.removed_indices_against(self.indices)

    def removed_indices_against(self, indices: npt.NDArray) -> npt.NDArray:
        """Return original indices absent from the supplied keep-index array."""
        if indices is None:
            raise ValueError("No indices to filter")
        return np.setdiff1d(np.arange(len(self.mzs)), indices)

    def is_removed(self, mz: float) -> bool:
        """Check if centroid is removed."""
        from koyo.utilities import find_nearest_value

        from ims_utils.spectrum import ppm_error

        found_mz = find_nearest_value(self.mzs, mz)
        ppm = abs(ppm_error(mz, found_mz))
        return np.isin(found_mz, self.removed_mzs) and ppm < 5  # type: ignore[return-value]

    def group_into_isotope_groups(
        self, mz_tol: float = 5e-3, mz_ppm: float = 0, max_in_group: int = 6, **_kwargs: ty.Any
    ) -> GroupResult:
        """Create tentative isotope groups, based purely on m/z. Only consider the 'remaining m/zs'."""
        from ims_utils.utilities import find_groups_within_bounds

        # Validate inputs
        assert mz_tol is not None and mz_tol > 0, "M/z tolerance must be greater than 0"
        assert max_in_group is not None and max_in_group > 0, "Max in group must be greater than 0"
        if not any([mz_ppm, mz_tol]):
            raise ValueError("Either `mz_ppm` or `mz_tol` must be specified")

        # isotope groups
        groups = find_groups_within_bounds(
            self.remaining_mzs, tolerance=mz_tol, ppm=mz_ppm, keep_singletons=True, max_in_group=max_in_group
        )
        # they need to be mapped to the original indices
        groups_ = []
        for _, mzs in groups:
            groups_.append(([self.find_index(mz) for mz in mzs], mzs))
        self._groups = groups_
        return groups_

    def get_groups(self, sort_by: ty.Literal["mz", "count"] = "mz", skip_singletons: bool = True) -> GroupResult:
        """Return groups of centroids."""
        groups = self._groups
        if groups is None:
            groups = self.group_into_isotope_groups()

        if sort_by == "count":
            groups = sorted(groups, key=lambda x: len(x[0]), reverse=True)
        else:
            groups = sorted(groups, key=lambda x: x[0])
        if skip_singletons:
            groups = [group for group in groups if len(group[0]) > 1]
        return groups

    def get_mzs_and_intensities_for(self, indices: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Return m/z and intensities for given indices."""
        if indices is None:
            raise ValueError("No indices to filter")
        return self.mzs[indices], self.intensities[indices]


def get_weights_for_indices(weights: npt.NDArray, indices: list[int]) -> npt.NDArray:
    """Return pairwise weights along a sequence of peak indices.

    The first value is the first peak's self-weight; later values compare each
    peak to its predecessor. Empty indices produce an empty float array.
    """
    res = np.zeros(len(indices))
    if len(indices) == 0:
        return res
    last_index = indices[0]
    for i, index in enumerate(indices):
        res[i] = weights[last_index, index]
        last_index = index
    return np.asarray(res)


def get_weights_for_indicies(weights: npt.NDArray, indices: list[int]) -> npt.NDArray:
    """Compatibility spelling for ``get_weights_for_indices``."""
    return get_weights_for_indices(weights, indices)


def filter_groups(groups: list[tuple[list[int], list[float]]], intensities: npt.NDArray) -> npt.NDArray:
    """Filter groups of centroids."""
    from ims_utils.utilities import find_highest

    indices_to_keep = []
    for indices, _ in groups:
        index = find_highest(indices, intensities)
        indices_to_keep.append(index)
    return np.unique(indices_to_keep)


def filter_groups_first(groups: list[tuple[list[int], list[float]]]) -> npt.NDArray:
    """Filter groups of centroids."""
    indices_to_keep = []
    for indices, _ in groups:
        index = indices[0]
        indices_to_keep.append(index)
    return np.unique(indices_to_keep)


def get_info(indices: list[int], intensities: npt.NDArray) -> tuple[int, list[str]]:
    """Return information about each item in a group."""
    from ims_utils.utilities import find_highest

    index = find_highest(indices, intensities)
    res = []
    for index_ in indices:
        res.append("selected" if index == index_ else "removed")
    return indices.index(index), res
