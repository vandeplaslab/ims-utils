"""Various utility functions."""

from __future__ import annotations

import numpy as np

from ims_utils.spectrum import ppm_to_delta_mass


def find_highest(indices: list[int], intensities: np.ndarray) -> int:
    """Find the highest intensity within a group."""
    if len(indices) == 0:
        raise ValueError("At least one index is required.")
    return int(indices[np.argmax(intensities[indices])])


def find_groups_within_bounds(
    arr: np.ndarray,
    distance: float = 1.00235,
    tolerance: float = 1e-4,
    ppm: float | None = None,
    keep_singletons: bool = False,
    max_in_group: int = 6,
) -> list[tuple[list[int], list[float]]]:
    """
    Find disjoint groups separated by a target distance in sorted m/z values.

    Each match is compared to the last group member. Earlier groups have priority
    when candidates could belong to multiple groups.

    Parameters
    ----------
    arr : np.ndarray
        Ascending one-dimensional m/z array.
    distance : float
        Target difference between adjacent group members in m/z units.
    tolerance : float
        Absolute tolerance around the target distance.
    ppm : float, optional
        If nonzero, replace absolute tolerance with ppm of the group's first m/z.
    keep_singletons : bool
        Include groups containing only one peak.
    max_in_group : int
        Maximum number of peaks per group.

    Returns
    -------
    list[tuple[list[int], list[float]]]
        Original indices and corresponding m/z values for each group.
    """
    if ppm:
        tolerance = 0

    min_distance = distance - tolerance
    max_distance = distance + tolerance

    groups = []
    n = len(arr)
    visited = set()  # Keep track of indices already included in groups
    min_in_group = 1 if not keep_singletons else 0

    for i in range(n):
        if i in visited:
            continue  # Skip already grouped elements

        group_indices = [i]
        group_values = [arr[i]]
        visited.add(i)

        if ppm:
            tolerance = ppm_to_delta_mass(arr[i], ppm)
            min_distance = distance - tolerance
            max_distance = distance + tolerance

        # Check subsequent elements for matches relative to the last group member
        last_index = i
        for j in range(i + 1, n):
            if j in visited:
                continue
            diff = arr[j] - arr[last_index]
            if min_distance <= diff <= max_distance and len(group_indices) < max_in_group:
                group_indices.append(j)
                group_values.append(arr[j])
                visited.add(j)
                last_index = j  # Update the reference point for further matches
            elif diff > max_distance:
                break  # Exit early since array is sorted

        if len(group_indices) > min_in_group:  # Only include groups with more than one element
            groups.append((group_indices, group_values))
    return groups


def find_groups_within_bounds_weighted(
    arr: np.ndarray,
    weights: np.ndarray,
    distance: float = 1.00235,
    tolerance: float = 1e-4,
    ppm: float | None = None,
    weight_threshold: float = 0.8,
    keep_singletons: bool = False,
    compare_to_last: bool = True,
    max_in_group: int = 6,
) -> list[tuple[list[int], list[float]]]:
    """
    Find disjoint distance-based groups gated by pairwise weights.

    Distance is always measured from the last member. Low-weight candidates are
    skipped because later candidates within the mass window may still qualify.

    Parameters
    ----------
    arr : np.ndarray
        Ascending one-dimensional m/z array.
    weights : np.ndarray
        Square pairwise weight matrix aligned with ``arr``.
    distance : float
        Target difference between adjacent members in m/z units.
    tolerance : float
        Absolute tolerance around the distance.
    ppm : float, optional
        If nonzero, replace absolute tolerance with ppm of the group's first m/z.
    weight_threshold : float
        Minimum accepted weight, inclusive.
    keep_singletons : bool
        Include groups containing only one peak.
    compare_to_last : bool
        Compare weights to the last member if True, otherwise to the first.
    max_in_group : int
        Maximum number of peaks per group.

    Returns
    -------
    list[tuple[list[int], list[float]]]
        Original indices and corresponding m/z values for each group.
    """
    if ppm:
        tolerance = 0

    min_distance = distance - tolerance
    max_distance = distance + tolerance

    groups = []
    n = len(arr)
    visited = set()  # Keep track of indices already included in groups
    min_in_group = 1 if not keep_singletons else 0

    for i in range(n):
        if i in visited:
            continue  # Skip already grouped elements

        group_indices = [i]
        group_values = [arr[i]]
        visited.add(i)

        if ppm:
            tolerance = ppm_to_delta_mass(arr[i], ppm)
            min_distance = distance - tolerance
            max_distance = distance + tolerance

        # Check subsequent elements for matches relative to the last group member
        last_index = i
        reference_index = i  # Initially compare to the first entry
        for j in range(i + 1, n):
            if j in visited:
                continue
            diff = arr[j] - arr[last_index]
            if (
                min_distance <= diff <= max_distance
                and weights[reference_index, j] >= weight_threshold
                and len(group_indices) < max_in_group
            ):
                group_indices.append(j)
                group_values.append(arr[j])
                visited.add(j)
                last_index = j  # Update the reference point for further matches
                if compare_to_last:
                    reference_index = j  # Update the reference point to the last group member
            elif diff > max_distance:
                break  # Exit early since array is sorted

        if len(group_indices) > min_in_group:  # Only include groups with more than one element
            groups.append((group_indices, group_values))
    return groups
