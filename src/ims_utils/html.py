"""HTML utilities."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from koyo.utilities import find_nearest_index, get_array_window
from tqdm import tqdm

if ty.TYPE_CHECKING:
    import plotly.graph_objects as go


def _get_colors(n: int) -> list[str]:
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    if n > len(colors):
        colors = [f"hsl({360 * i / n}, 100%, 50%)" for i in range(n)]
    return colors


def write_html(fig: go.Figure, filename: PathLike) -> None:
    """Write HTML to file."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(filename), include_plotlyjs="cdn")


def show_html(filename: PathLike) -> None:
    """Show HTML in browser."""
    import webbrowser

    filename = Path(filename)
    webbrowser.open(filename.as_uri())


def make_plot(x: np.ndarray, y: np.ndarray, name: str, x_label: str = "", y_label: str = "", title: str = "") -> go.Figure:
    """Export Plotly line plot as HTML document."""
    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()

    # Add the mass spectrum line
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="simple_white",
        width=1200,  # Adjust width here
        height=800,  # Adjust height here
    )
    return fig


def add_scatter(fig: go.Figure, x: np.ndarray, y: np.ndarray, name: str = "scatter", color: str = "blue", visible: bool = True) -> go.Figure:
    """Add scatter plot."""
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=name, line={"color": color}, visible=True if visible else "legendonly"))
    return fig


def add_line(fig: go.Figure, x: np.ndarray, y: np.ndarray, name: str = "line", color: str = "blue", visible: bool = True) -> go.Figure:
    """Add line plot."""
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line={"color": color}, visible=True if visible else "legendonly"))
    return fig


def make_spectrum(mz_x: np.ndarray, mz_y: np.ndarray, name: str) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    return make_plot(mz_x, mz_y, name, x_label="m/z", y_label="Intensity", title="Mass Spectrum")


def make_peaks_spectrum(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    peaks: np.ndarray,
    window: float = 0.1,
    normalize: bool = False,
    n_cols: int = 3,
    style="simple_white",
    width: int = 400,
    height: int = 350,
    titles: list[str] | None = None,
) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    import plotly.graph_objects as go
    from koyo.utilities import get_array_window
    from plotly.subplots import make_subplots

    # get minimum and maximum m/z values
    mz_min = min(mz_x.min() for mz_x, _ in spectra.values())
    mz_max = max(mz_x.max() for mz_x, _ in spectra.values())

    # validate peaks are within the m/z range and if not, exclude them
    peaks_, excluded_peaks_ = [], []
    for peak in peaks:
        if mz_min <= peak <= mz_max:
            peaks_.append(peak)
        else:
            excluded_peaks_.append(peak)
    peaks = np.asarray(peaks_)

    peaks = np.asarray(peaks)
    n_peaks = len(peaks)
    n_rows = (n_peaks + n_cols - 1) // n_cols
    colors = _get_colors(len(spectra))

    # Create a figure
    if titles is None:
        titles = [f"Peak {peaks[i]:.3f}" for i in range(n_peaks)]
    assert len(titles) == n_peaks, f"Expected {n_peaks} titles, got {len(titles)}"

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)
    for i, peak in enumerate(tqdm(peaks, desc="Adding peaks...")):
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
        xaxis={"title": "m/z"},
        yaxis={"title": "Intensity"},
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
    as_vline: bool = True,
    mz_min: float | None = None,
    mz_max: float | None = None,
) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()

    # Add the mass spectrum line
    y_max = 0
    ys_ = {}
    for name, (mz_x, mz_y) in spectra.items():
        if normalize:
            mz_y = mz_y / mz_y.max()
        if mz_min is not None or mz_max is not None:
            x, y = get_array_window(mz_x, mz_min or mz_x.min(), mz_max or mz_x.max(), mz_y)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        else:
            fig.add_trace(go.Scatter(x=mz_x, y=mz_y, mode="lines", name=name))
        y_max = max(y_max, mz_y.max())
        if not as_vline and px is not None:
            indices = find_nearest_index(mz_x, px)
            ys_[name] = mz_y[indices]

    if px is not None:
        if as_vline or py is not None:
            for x in px:
                fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="black")
        else:
            if py is None:
                for name, py in ys_.items():
                    fig.add_trace(
                        go.Scatter(
                            x=px,
                            y=py,
                            mode="markers",
                            name=f"{name} (peaks)",
                            marker={"color": "black", "size": 10, "symbol": "diamond-tall", "opacity": 0.5},
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=px,
                        y=py,
                        mode="markers",
                        name="peaks",
                        marker={"color": "black", "size": 10, "symbol": "diamond-tall", "opacity": 0.5},
                    )
                )
    # Update x/y min/max
    x_min = mz_min or min(mz_x.min() for mz_x, _ in spectra.values())
    x_max = mz_max or max(mz_x.max() for mz_x, _ in spectra.values())

    # Update layout
    fig.update_layout(
        title="Mass Spectrum",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        template="simple_white",
        width=1200,  # Adjust width here
        height=700,  # Adjust height here
        hoverlabel={"namelength": -1},
        xaxis={"range": [x_min, x_max]},
        yaxis={"range": [0, y_max * 1.05]},
    )
    return fig


def make_spectrum_with_scatter(
    mz_x: np.ndarray,
    mz_y: np.ndarray,
    peaks: dict[str, tuple[np.ndarray, np.ndarray]],
    name: str = "Spectrum",
    marker_size: int = 5,
    marker_opacity: float = 0.75,
) -> go.Figure:
    """Export Plotly mass spectrum as HTML document."""
    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()

    colors = _get_colors(len(peaks))

    # Add the mass spectrum line
    y_max = mz_y.max()
    fig.add_trace(go.Scatter(x=mz_x, y=mz_y, mode="lines", name=name, line={"color": "black"}))
    for scatter_name, (mz_x_scatter, mz_y_scatter) in peaks.items():
        y_max = max(y_max, mz_y_scatter.max())
        fig.add_trace(
            go.Scatter(
                x=mz_x_scatter,
                y=mz_y_scatter,
                mode="markers",
                name=scatter_name,
                marker={"color": colors.pop(0), "size": marker_size, "opacity": marker_opacity},
            )
        )

    # Update layout
    fig.update_layout(
        title="Mass Spectrum",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        template="simple_white",
        width=1200,  #
        height=800,  # Adjust height here
        yaxis={"range": [0, y_max * 1.05]},
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
        template="simple_white",
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
        template="simple_white",
        width=1200,  # Adjust width here
        height=800,  # Adjust height here
        xaxis={"range": [mz_x[0], mz_x[-1]]},
    )
    return fig
