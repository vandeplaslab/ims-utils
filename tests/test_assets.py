from importlib.metadata import version

import ims_utils
from ims_utils.assets import find_matrix, get_matrix_tables


def test_version_uses_distribution_name() -> None:
    assert ims_utils.__version__ == version("ims-utils")


def test_matrix_assets_are_discoverable() -> None:
    tables = get_matrix_tables()
    assert {path.name for path in tables} == {"dmaca;negative.csv", "mapca;positive.csv", "mapca;negative.csv"}
    assert all(path.is_file() for path in tables)
    assert find_matrix("MAPCA", "POSITIVE").name == "mapca;positive.csv"
    assert find_matrix("missing", "positive") is None
