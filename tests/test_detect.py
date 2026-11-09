import numpy as np
import pytest

from ims_utils.detect import decreasing_step_size, get_best_snr


def test_step_sizes_follow_exponential_schedule() -> None:
    actual = list(decreasing_step_size(2.0, range(3), multiplier=0.5))
    np.testing.assert_allclose(actual, 2.0 * np.exp(-0.5 * np.arange(3)))
    assert list(decreasing_step_size(2.0, [])) == []


@pytest.mark.parametrize("data", [[], [(1, 2)]])
def test_best_snr_requires_nonempty_rows(data: list) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        get_best_snr(data)


def test_best_snr_falls_back_to_highest_score() -> None:
    assert get_best_snr([(3, 0.2, 10)]) == (3, 0.2, 10)
    assert get_best_snr([(3, 0.2, 10), (7, 0.1, 20)]) == (7, 0.1, 20)


def test_best_snr_uses_knee_coordinate_instead_of_list_index(monkeypatch: pytest.MonkeyPatch) -> None:
    from ims_utils._vendored import kneed

    class Knee:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.knee = 30

    monkeypatch.setattr(kneed, "KneeLocator", Knee)
    data = [(10, 0.5, 1), (20, 0.4, 4), (30, 0.3, 7), (40, 0.2, 8), (50, 0.1, 10)]
    assert get_best_snr(data) == data[2]
