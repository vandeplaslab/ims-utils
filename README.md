# ims-utils

[![License](https://img.shields.io/pypi/l/ims-utils.svg?color=green)](https://github.com/vandeplaslab/ims-utils/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ims-utils.svg?color=green)](https://pypi.org/project/ims-utils)
[![Python Version](https://img.shields.io/pypi/pyversions/ims-utils.svg?color=green)](https://python.org)
[![CI](https://github.com/vandeplaslab/ims-utils/actions/workflows/ci.yml/badge.svg)](https://github.com/vandeplaslab/ims-utils/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/vandeplaslab/ims-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/vandeplaslab/ims-utils)

Small utilities for imaging mass spectrometry and spectrum processing. The package requires Python 3.10 or newer; mass-to-charge values are in Da (m/z), and intensity arrays are numeric.

## Install

Install the core package with uv:

```bash
uv add ims-utils
```

Optional features are installed with extras. `html` adds Plotly, while `batch` adds pandas, scikit-learn, mpire, and patsy for batch normalization. For example:

```bash
uv add 'ims-utils[html,batch]'
```

The declared `oms` extra is restricted to Python versions below 3.10, so it cannot install `pyopenms` alongside this package's Python >=3.10 requirement. Install a compatible `pyopenms` build separately if you need OpenMS centroiding. `centroid_to_profile_ppm` optionally uses the undeclared `ms_peak_picker` package. Other optional utilities have undeclared imports: `fiducials` needs `opencv-python` and `natsort`, and all `detect` centroid helpers need `pyopenms`.

The `peak_filter` and `kendrick` modules use pandas and are available with the `batch` extra.

## Examples

### Filter and resample a spectrum

`find_between` returns indices within inclusive m/z bounds. `interpolate_ppm` interpolates intensities onto a requested m/z axis. `resample_ppm` instead sums intensities into ppm bins.

```python
import numpy as np
from ims_utils.filters import PpmResampling
from ims_utils.spectrum import find_between, get_ppm_axis, interpolate_ppm, resample_ppm

mz = np.linspace(100.0, 200.0, 1001)
intensity = np.exp(-0.5 * ((mz - 150.0) / 0.2) ** 2)
indices = find_between(mz, 145.0, 155.0)
target_mz = get_ppm_axis(145.0, 155.0, 10.0)
interpolated_mz, interpolated_intensity = interpolate_ppm(target_mz, mz[indices], intensity[indices])
summed_intensity = resample_ppm(target_mz, mz[indices], intensity[indices])
ppm_mz, ppm_intensity = PpmResampling(10.0, 145.0, 155.0)(mz[indices], intensity[indices])
```

### Convert centroids to a profile

Centroid arrays must have matching lengths and be sorted by m/z unless `sort=True`. Choose exactly one resolution control. Returned values are the profile m/z axis, summed intensity, and per-centroid Gaussian sigma, respectively.

```python
import numpy as np
from ims_utils.profile import centroid_to_profile

centroid_mz = np.array([100.0, 100.5, 101.0])
centroid_intensity = np.array([10.0, 20.0, 8.0])
profile_mz, profile_intensity, sigma = centroid_to_profile(
    centroid_mz, centroid_intensity, resolving_power=10_000
)
```

### Estimate and apply classic lockmass correction

Classic estimators return one index shift per reference peak; they do not change the mass axis. `fast_roll` applies a shift in intensity-array indices. The measured peak's index offset is negated when applying the correction.

```python
import numpy as np
from ims_utils.lockmass import WeightedIntensityLockmassEstimator

mz = np.linspace(100.0, 200.0, 10_001)
references = np.array([120.0, 160.0, 190.0])
intensity = sum(1000.0 * np.exp(-0.5 * ((mz - peak) / 0.03) ** 2) for peak in references)
estimator = WeightedIntensityLockmassEstimator(mz, references)
index_shifts = estimator.estimate(intensity)
corrected = estimator.apply(intensity, -float(np.mean(index_shifts)))
```

### Robust lockmass correction

The robust estimator reports observed peak offsets as ppm in its estimate tuple, and fits polynomial coefficients in descending order. `correct` applies the correction and returns a new intensity array on the same m/z axis. With `fast=True`, one scalar correction uses the arithmetic mean of reference masses; `fast=False` applies a continuous interpolated warp.

```python
import numpy as np
from ims_utils.robust_lockmass import RobustLockmassEstimator

mz = np.linspace(100.0, 200.0, 10_001)
references = np.array([120.0, 140.0, 160.0, 180.0])
intensity = sum(1000.0 * np.exp(-0.5 * ((mz - peak * (1 + 8e-6)) / 0.03) ** 2) for peak in references)
estimator = RobustLockmassEstimator(mz, references)
reference_mz, observed_ppm, coefficients, inliers = estimator.estimate(intensity)
corrected = estimator.correct(intensity, fast=False)
```

In the estimate tuple, `reference_mz` contains the reference mass repeated for each detected candidate; `observed_ppm` contains the measured candidate's offset from that reference.

### Inter-normalize datasets

The mean-fold-change inter-normalizer accepts one samples-by-features array per named dataset. Apply each returned scale by multiplication.

```python
import numpy as np
from ims_utils.internorm import calculate_mfc_inter_normalization

datasets = {
    "sample_a": np.array([[10.0, 20.0], [12.0, 18.0]]),
    "sample_b": np.array([[20.0, 40.0], [24.0, 36.0]]),
}
names, scales = calculate_mfc_inter_normalization(datasets)
normalized = {name: datasets[name] * scale for name, scale in zip(names, scales)}
```

### Batch normalization

ComBat and ReComBat expect samples by features and one batch label per sample. They return corrected values in that orientation. Install `ims-utils[batch]` first.

```python
import numpy as np
from ims_utils.batchnorm import combat

samples_by_features = np.array([
    [1.0, 3.0, 5.0, 2.0, 4.0], [1.2, 2.8, 4.7, 2.4, 4.3],
    [4.0, 6.0, 2.0, 5.0, 3.0], [4.2, 5.8, 2.3, 5.4, 2.7],
    [1.1, 3.2, 5.2, 1.8, 3.8], [4.1, 6.2, 1.8, 5.2, 3.2],
])
batch_labels = np.array(["A", "A", "B", "B", "A", "B"])
corrected = combat(samples_by_features, batch_labels)
```

## Development

Clone the repository and install development dependencies with uv:

```bash
git clone https://github.com/vandeplaslab/ims-utils.git
cd ims-utils
uv sync --group dev
```

Run the test suite with `uv run pytest`.

Contributions and documentation improvements are welcome.
