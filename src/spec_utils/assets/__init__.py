"""Assets API."""

from __future__ import annotations

from pathlib import Path

from koyo.utilities import get_module_path

HERE = Path(get_module_path("spec_utils.assets", "__init__.py")).parent.resolve()


def get_matrix_tables() -> list[Path]:
    """Return list of matrix tables."""
    path = HERE / "matrix"
    return list(path.glob("*.csv"))


def find_matrix(matrix: str, polarity: str) -> Path | None:
    """Find matrix table."""
    path = HERE / "matrix"
    matrix = matrix.lower()
    polarity = polarity.lower()
    filename = path / f"{matrix};{polarity}.csv"
    if filename.exists():
        return filename
    return None
