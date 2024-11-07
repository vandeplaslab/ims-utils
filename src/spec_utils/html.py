"""HTML utilities."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

if ty.TYPE_CHECKING:
    import plotly.graph_objects as go


def write_html(fig, filename: PathLike) -> None:
    """Write HTML to file."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(filename), include_plotlyjs="cdn")


def make_spectrum(mz_x: np.ndarray, mz_y: np.ndarray, name: str) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()

    # Add the mass spectrum line
    fig.add_trace(go.Scatter(x=mz_x, y=mz_y, mode="lines", name=name))

    # Update layout
    fig.update_layout(
        title="Mass Spectrum",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        template="plotly_white",
        width=1200,  # Adjust width here
        height=800,  # Adjust height here
    )
    return fig


def make_spectrum_with_scatter(
    mz_x: np.ndarray, mz_y: np.ndarray, name: str, mz_x_scatter: np.ndarray, mz_y_scatter: np.ndarray, name_scatter: str
) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()

    # Add the mass spectrum line
    fig.add_trace(go.Scatter(x=mz_x, y=mz_y, mode="lines", name=name))
    fig.add_trace(go.Scatter(x=mz_x_scatter, y=mz_y_scatter, mode="markers", name=name_scatter))

    # Update layout
    fig.update_layout(
        title="Mass Spectrum",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        template="plotly_white",
        width=1200,  # Adjust width here
        height=800,  # Adjust height here
    )
    return fig


def make_butterfly_spectrum(
    mz_x: np.ndarray, mz_y_top: np.ndarray, mz_y_bottom: np.ndarray, name_top: str, name_bottom: str
) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()

    # Add the mass spectrum line
    fig.add_trace(go.Scatter(x=mz_x, y=mz_y_top / mz_y_top.max(), mode="lines", name=name_top, line=dict(color="blue")))
    fig.add_trace(
        go.Scatter(x=mz_x, y=-mz_y_bottom / mz_y_bottom.max(), mode="lines", name=name_bottom, line=dict(color="red"))
    )

    # Update layout
    fig.update_layout(
        title="Mass Spectrum",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        template="plotly_white",
        width=1200,  # Adjust width here
        height=800,  # Adjust height here
    )
    return fig


def make_difference_spectrum(
    mz_x: np.ndarray, mz_y_top: np.ndarray, mz_y_bottom: np.ndarray, name_top: str, name_bottom: str
) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    import plotly.graph_objects as go

    mz_y_top = mz_y_top / mz_y_top.max()
    mz_y_bottom = mz_y_bottom / mz_y_bottom.max()

    # make difference plot and mask values above and below
    mz_y_diff = mz_y_top - mz_y_bottom
    mz_y_top = mz_y_diff.copy()
    mz_y_top[mz_y_top < 0] = 0
    mz_y_bottom = mz_y_diff.copy()
    mz_y_bottom[mz_y_bottom > 0] = 0

    # Create a figure
    fig = go.Figure()

    # Add the mass spectrum line
    fig.add_trace(go.Scatter(x=mz_x, y=mz_y_top, mode="lines", name=name_top, line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=mz_x, y=mz_y_bottom, mode="lines", name=name_bottom, line=dict(color="red")))

    # Update layout
    fig.update_layout(
        title="Mass Spectrum",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        template="plotly_white",
        width=1200,  # Adjust width here
        height=800,  # Adjust height here
    )
    return fig
