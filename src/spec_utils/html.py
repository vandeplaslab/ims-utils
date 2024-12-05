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


def make_peaks_spectrum(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    peaks: np.ndarray,
    window: float = 0.1,
    normalize: bool = False,
    n_cols: int = 3,
    style="plotly_white",
    width: int = 400,
    height: int = 350,
) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    import plotly.graph_objects as go
    from koyo.utilities import get_array_window
    from plotly.subplots import make_subplots

    peaks = np.asarray(peaks)
    n_peaks = len(peaks)
    n_rows = (n_peaks + n_cols - 1) // n_cols
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    if len(spectra) > len(colors):
        colors = [f"hsl({360 * i / len(spectra)}, 100%, 50%)" for i in range(len(spectra))]

    # Create a figure
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"Peak {peaks[i]:.3f}" for i in range(n_peaks)])
    for i, peak in enumerate(peaks):
        row, col = (i // n_cols + 1, i % n_cols + 1)
        for j, (name, (mz_x, mz_y)) in enumerate(spectra.items()):
            mz_x_window, mz_y_window = get_array_window(mz_x, peak - window, peak + window, mz_y)
            if normalize:
                mz_y_window = mz_y_window / mz_y_window.max()
            color = colors[j]
            fig.add_trace(
                go.Scatter(
                    x=mz_x_window,
                    y=mz_y_window,
                    mode="lines",
                    name=name,
                    line={"color": color},
                    showlegend=i == 0,
                    legendgroup=name,
                ),
                row=row,
                col=col,
            )
        fig.add_vline(x=peak, line_width=1, line_dash="dash", line_color="black", row=row, col=col)

    # Update layout
    fig.update_layout(
        title="Mass Spectrum",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        template=style,
        width=50 + width * n_cols,  # Adjust width here
        height=height * n_rows,  # Adjust height here
        hoverlabel={"namelength": -1},
    )
    return fig


def make_overlay_spectrum(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    px: np.ndarray | None = None,
    py: np.ndarray | None = None,
    normalize: bool = True,
) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()

    # Add the mass spectrum line
    y_max = 0
    for name, (mz_x, mz_y) in spectra.items():
        if normalize:
            mz_y = mz_y / mz_y.max()
        fig.add_trace(go.Scatter(x=mz_x, y=mz_y, mode="lines", name=name))
        y_max = max(y_max, mz_y.max())

    if px is not None:
        if py is None:
            for x in px:
                fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="black")
        else:
            fig.add_trace(go.Scatter(x=px, y=py, mode="markers", name="peaks", marker={"color": "black", "size": 10}))
    # Update x/y min/max
    x_min = min(mz_x.min() for mz_x, _ in spectra.values())
    x_max = max(mz_x.max() for mz_x, _ in spectra.values())
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[0, y_max * 1.05])

    # Update layout
    fig.update_layout(
        title="Mass Spectrum",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        template="plotly_white",
        width=1200,  # Adjust width here
        height=700,  # Adjust height here
        hoverlabel={"namelength": -1},
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
    fig.add_trace(go.Scatter(x=mz_x, y=mz_y_top / mz_y_top.max(), mode="lines", name=name_top, line={"color": "blue"}))
    fig.add_trace(
        go.Scatter(x=mz_x, y=-mz_y_bottom / mz_y_bottom.max(), mode="lines", name=name_bottom, line={"color": "red"})
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
    fig.add_trace(go.Scatter(x=mz_x, y=mz_y_top, mode="lines", name=name_top, line={"color": "blue"}))
    fig.add_trace(go.Scatter(x=mz_x, y=mz_y_bottom, mode="lines", name=name_bottom, line={"color": "red"}))

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
