import numpy as np
import pytest

from ims_utils.batchnorm import HAS_COMBAT, HAS_RECOMBAT, ReComBat, combat, combine_arrays, create_batches

ARRAY_2D = np.random.default_rng(0).integers(1, 255, (100, 10)) * 1.0
DATA_2D = {"a": ARRAY_2D, "b": ARRAY_2D * 2.0, "c": ARRAY_2D * 0.5}


@pytest.mark.skipif(not HAS_COMBAT, reason="ComBat not installed")
def test_create_batches():
    batches = create_batches(DATA_2D)
    assert len(batches) == 300
    assert batches.unique().shape[0] == 3


@pytest.mark.skipif(not HAS_COMBAT, reason="ComBat not installed")
def test_combine_arrays():
    combined = combine_arrays(DATA_2D)
    assert combined.shape == (300, 10)


@pytest.mark.skipif(not HAS_COMBAT, reason="ComBat not installed")
def test_combat():
    res = combat(combine_arrays(DATA_2D), create_batches(DATA_2D))
    assert res.shape == (300, 10)

    res = combat(combine_arrays(DATA_2D), create_batches(DATA_2D).values)
    assert res.shape == (300, 10)


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


@pytest.mark.skipif(not HAS_COMBAT, reason="ComBat not installed")
def test_combat_reduces_known_batch_offset_without_modifying_input() -> None:
    reference = np.random.default_rng(42).normal(10.0, 2.0, (40, 8))
    data = np.vstack([reference, reference * 1.5 + 5.0])
    original = data.copy()
    batches = np.repeat(["a", "b"], 40)
    corrected = combat(data, batches)
    assert np.isfinite(corrected).all()
    initial_difference = np.abs(data[:40].mean(0) - data[40:].mean(0))
    corrected_difference = np.abs(corrected[:40].mean(0) - corrected[40:].mean(0))
    # Empirical Bayes shrinkage leaves a residual rather than forcing equal means.
    assert np.all(corrected_difference < 0.1 * initial_difference)
    np.testing.assert_array_equal(data, original)


@pytest.mark.skipif(not HAS_RECOMBAT, reason="ReComBat not installed")
def test_parametric_convergence_handles_negative_gamma_and_zero_initial_values() -> None:
    from ims_utils.batchnorm.utilities import parametric_update

    initial_gamma = np.array([-1.0, 0.0])
    initial_delta = np.ones(2)
    observed = np.array([[-4.0, -2.0], [-3.0, -1.0], [-2.0, 0.0], [-3.0, -1.0]])
    gamma_hat = observed.mean(axis=0)
    with np.errstate(all="raise"):
        gamma, delta = parametric_update(
            initial_gamma,
            initial_delta,
            4,
            0.8,
            gamma_hat,
            -1.5,
            1.0,
            2.0,
            observed,
            conv_criterion=1e-8,
        )
    expected_gamma = (4 * 0.8 * gamma_hat + delta * -1.5) / (4 * 0.8 + delta)
    expected_delta = (1.0 + 0.5 * np.sum((observed - gamma) ** 2, axis=0)) / 3.0
    np.testing.assert_allclose(gamma, expected_gamma, rtol=1e-7)
    np.testing.assert_allclose(delta, expected_delta, rtol=1e-7)


@pytest.mark.skipif(not HAS_COMBAT, reason="ComBat not installed")
def test_combat_solver_converges_from_zero_gamma() -> None:
    from ims_utils.batchnorm.combat import _it_sol

    observations = np.array([[-4.0, -3.0, -2.0, -3.0], [-2.0, -1.0, 0.0, -1.0]])
    with np.errstate(all="raise"):
        gamma, delta = _it_sol(observations, np.zeros(2), np.ones(2), -1.5, 0.8, 2.0, 1.0, conv=1e-8)
    np.testing.assert_allclose(gamma, delta * -1.5 / (0.8 * 4 + delta), rtol=1e-7)
    np.testing.assert_allclose(delta, (0.5 * ((observations - gamma[:, None]) ** 2).sum(axis=1) + 1) / 3, rtol=1e-7)
