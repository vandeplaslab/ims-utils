"""Various utility functions."""

from __future__ import annotations

import numpy as np
from koyo.spectrum import ppm_to_delta_mass


def find_highest(indices: list[int], intensities: np.ndarray) -> int:
    """Find the highest intensity within a group."""
    indices = np.asarray(indices)  # type: ignore[assignment]
    if len(indices) == 1:
        index = indices[0]
    else:
        ys = intensities[indices]
        index = indices[np.argmax(ys)]
    return index


def find_groups_within_bounds(
    arr: np.ndarray,
    distance: float = 1.00235,
    tolerance: float = 1e-4,
    ppm: float | None = None,
    keep_singletons: bool = False,
    max_in_group: int = 6,
) -> list[tuple[list[int], list[float]]]:
    """
    Find groups of values within specified bounds. For each group, subsequent matches are
    checked against the last entry in the group.

    Parameters
    ----------
        arr (np.ndarray): The sorted input array.
        distance (float): Default distance
        tolerance (float): Tolerance around the distance.
        keep_singletons (bool): Whether to include singletons in the output.
        max_in_group (int): Maximum number of elements in a group.

    Returns
    -------
        List[Tuple[List[int], List[float]]]: A list of groups, where each group contains:
            - A list of indices of the matching data points.
            - A list of corresponding values from the array.
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
    Find groups of values using distance bounds and weights. For each group, subsequent matches
    are checked against the last entry in the group.

    Parameters
    ----------
        arr (np.ndarray): The sorted input array.
        weights (np.ndarray): A 2D array of weights between all pairs of values.
        distance (float): Default distance.
        tolerance (float): Tolerance around the distance.
        weight_threshold (float): Minimum weight required to link points in the same group.
        keep_singletons (bool): Whether to include singletons in the output.
        compare_to_last (bool): Whether to compare subsequent matches to the last entry in the group.
        max_in_group (int): Maximum number of elements in a group.

    Returns
    -------
        List[Tuple[List[int], List[float]]]: A list of groups, where each group contains:
            - A list of indices of the matching data points.
            - A list of corresponding values from the array.
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
            elif diff > max_distance or weights[reference_index, j] < weight_threshold:
                break  # Exit early since array is sorted

        if len(group_indices) > min_in_group:  # Only include groups with more than one element
            groups.append((group_indices, group_values))
    return groups
