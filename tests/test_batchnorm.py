import numpy as np
import pytest

from ms_utils.batchnorm import HAS_RECOMBAT, ReComBat, combat, combine_arrays, create_batches

ARRAY_2D = np.random.randint(0, 255, (100, 10)) * 1.0
DATA_2D = {"a": ARRAY_2D, "b": ARRAY_2D * 2.0, "c": ARRAY_2D * 0.5}


def test_create_batches():
    batches = create_batches(DATA_2D)
    assert len(batches) == 300
    assert batches.unique().shape[0] == 3


def test_combine_arrays():
    combined = combine_arrays(DATA_2D)
    assert combined.shape == (300, 10)


@pytest.mark.skipif(not HAS_RECOMBAT, reason="ReComBat not installed")
def test_recombat():
    data = combine_arrays(DATA_2D)
    batches = create_batches(DATA_2D)

    rebatch = ReComBat()
    assert not rebatch.is_fitted
    rebatch.fit(data, batches)
    assert rebatch.is_fitted
    ret = rebatch.transform(data, batches)
    assert ret.shape == (300, 10)


def test_combat():
    res = combat(combine_arrays(DATA_2D), create_batches(DATA_2D))
    assert res.shape == (300, 10)

    res = combat(combine_arrays(DATA_2D), create_batches(DATA_2D).values)
    assert res.shape == (300, 10)
