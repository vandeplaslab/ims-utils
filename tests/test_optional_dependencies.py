from __future__ import annotations

import subprocess
import sys
from textwrap import dedent

import numpy as np
import pytest


def test_core_imports_and_batch_fallback_without_optional_dependencies() -> None:
    script = dedent("""
        import importlib.abc
        import sys

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                optional = {'pandas', 'patsy', 'sklearn', 'plotly', 'pyopenms', 'cv2', 'ms_peak_picker'}
                if fullname.split('.')[0] in optional:
                    raise ModuleNotFoundError(f'Blocked optional module: {fullname}', name=fullname)
                return None

        sys.meta_path.insert(0, BlockOptional())
        import ims_utils
        import ims_utils.assets
        import ims_utils.baseline
        import ims_utils.coordinates
        import ims_utils.deisotope
        import ims_utils.detect
        import ims_utils.fiducials
        import ims_utils.filters
        import ims_utils.html
        import ims_utils.internorm
        import ims_utils.lockmass
        import ims_utils.profile
        import ims_utils.resample
        import ims_utils.robust_lockmass
        import ims_utils.smooth
        import ims_utils.spectrum
        import ims_utils.utilities
        from ims_utils import batchnorm
        assert not batchnorm.HAS_COMBAT
        assert not batchnorm.HAS_RECOMBAT
        assert batchnorm.combat(None, None) is None
        assert batchnorm.ReComBat() is None
        assert ims_utils.assets.get_matrix_tables()
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "ims-utils[batch]" in result.stdout


def test_profile_ppm_reports_missing_optional_package(monkeypatch: pytest.MonkeyPatch) -> None:
    from ims_utils import profile

    monkeypatch.setattr(profile, "HAS_MPP", False)
    with pytest.raises(ImportError, match="ms_peak_picker"):
        profile.centroid_to_profile_ppm(np.array([100.0]), np.array([1.0]), 99.0, 101.0)


def test_openms_wrapper_reports_missing_optional_package(monkeypatch: pytest.MonkeyPatch) -> None:
    from ims_utils.deisotope import oms_deisotope

    monkeypatch.setitem(sys.modules, "pyopenms", None)
    with pytest.raises(ImportError):
        oms_deisotope(np.array([100.0]), np.array([1.0]), (1, 2))
