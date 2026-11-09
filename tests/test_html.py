from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")

from ims_utils.html import (
    add_line,
    add_scatter,
    make_butterfly_spectrum,
    make_difference_spectrum,
    make_overlay_spectrum,
    make_peaks_spectrum,
    make_plot,
    make_spectrum,
    make_spectrum_with_scatter,
    show_html,
    write_html,
)


def test_line_and_marker_helpers_update_figure() -> None:
    axis, intensity = np.arange(3.0), np.array([0.0, 2.0, 0.0])
    figure = make_plot(axis, intensity, "a", "m/z", "Intensity", "Example")
    assert add_line(figure, axis, intensity * 2, visible=False) is figure
    assert add_scatter(figure, axis, intensity, name="peaks") is figure
    assert len(figure.data) == 3
    assert figure.data[1].visible == "legendonly"
    np.testing.assert_array_equal(figure.data[1].y, [0.0, 4.0, 0.0])
    assert figure.layout.xaxis.title.text == "m/z"
    np.testing.assert_array_equal(make_spectrum(axis, intensity, "a").data[0].y, intensity)


def test_overlay_honors_explicit_peak_intensities_and_zero_bounds() -> None:
    axis, intensity = np.array([-1.0, 0.0, 1.0, 2.0]), np.array([1.0, 2.0, 4.0, 8.0])
    figure = make_overlay_spectrum(
        {"a": (axis, intensity)}, px=np.array([1.0]), py=np.array([7.0]), normalize=False, as_vline=False, mz_min=0.0
    )
    assert figure.data[-1].mode == "markers"
    np.testing.assert_array_equal(figure.data[-1].y, [7.0])
    assert figure.layout.xaxis.range[0] == 0.0
    assert not figure.layout.shapes
    inferred = make_overlay_spectrum({"a": (axis, intensity)}, px=np.array([1.0]), as_vline=False)
    np.testing.assert_allclose(inferred.data[-1].y, [0.5])
    vertical = make_overlay_spectrum({"a": (axis, intensity)}, px=np.array([1.0]))
    assert vertical.layout.shapes[0].x0 == 1.0


def test_butterfly_and_difference_have_expected_signs() -> None:
    axis = np.arange(3.0)
    top, bottom = np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0, 2.0])
    butterfly = make_butterfly_spectrum(axis, top, bottom, "top", "bottom")
    np.testing.assert_allclose(butterfly.data[0].y, [0.0, 1.0, 0.5])
    np.testing.assert_allclose(butterfly.data[1].y, [0.0, -0.5, -1.0])
    difference = make_difference_spectrum(axis, top, bottom, "top", "bottom")
    np.testing.assert_allclose(difference.data[0].y, [0.0, 0.5, 0.0])
    np.testing.assert_allclose(difference.data[1].y, [0.0, 0.0, -0.5])
    np.testing.assert_array_equal(top, [0.0, 2.0, 1.0])


def test_zero_signal_normalization_is_finite() -> None:
    axis, zeros = np.arange(5.0), np.zeros(5)
    with np.errstate(all="raise"):
        figures = [
            make_overlay_spectrum({"a": (axis, zeros)}),
            make_butterfly_spectrum(axis, zeros, zeros, "a", "b"),
            make_difference_spectrum(axis, zeros, zeros, "a", "b"),
            make_peaks_spectrum({"a": (axis, zeros)}, np.array([2.0]), window=1.0, normalize=True),
        ]
    for figure in figures:
        for trace in figure.data:
            np.testing.assert_array_equal(trace.y, np.zeros(len(trace.y)))


def test_peak_subplot_and_scatter_coordinates() -> None:
    axis, intensity = np.arange(100.0, 105.0), np.array([0.0, 1.0, 5.0, 1.0, 0.0])
    figure = make_peaks_spectrum({"a": (axis, intensity)}, np.array([102.0]), window=1.0, n_cols=1)
    assert figure.layout.shapes[0].x0 == 102.0
    assert max(figure.data[0].y) == 5.0
    scatter = make_spectrum_with_scatter(axis, intensity, {"peaks": (np.array([102.0]), np.array([5.0]))})
    np.testing.assert_array_equal(scatter.data[1].x, [102.0])
    np.testing.assert_array_equal(scatter.data[1].y, [5.0])


def test_html_write_and_relative_browser_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "nested" / "spectrum.html"
    write_html(make_spectrum(np.arange(3.0), np.ones(3), "a"), destination)
    assert "cdn.plot.ly" in destination.read_text()
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    monkeypatch.chdir(tmp_path)
    show_html("nested/spectrum.html")
    assert opened == [destination.as_uri()]
