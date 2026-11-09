import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix

from ims_utils.smooth import gaussian, smooth_spectra


def test_gaussian_preserves_constant_and_smooths_impulse() -> None:
    np.testing.assert_allclose(gaussian(np.ones(20), 2), np.ones(20))
    impulse = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(gaussian(impulse, 1), gaussian_filter(impulse, 1))


def test_smoothing_rows_does_not_mix_spectra_and_supports_sparse() -> None:
    spectra = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    expected = np.vstack([gaussian_filter(row, 1) for row in spectra])
    dense = spectra.copy()
    assert smooth_spectra(dense) is dense
    np.testing.assert_allclose(dense, expected)
    np.testing.assert_allclose(smooth_spectra(csr_matrix(spectra)), expected)
