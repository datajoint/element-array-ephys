from __future__ import annotations

from modulefinder import Module
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from .. import probe


def plot_waveform(waveform: np.ndarray, sampling_rate: float) -> go.Figure:
    """Plot unit waveform.

    Args:
        waveform (np.ndarray): Amplitude of a spike waveform in μV.
        sampling_rate (float): Sampling rate in kHz.

    Returns:
        go.Figure: Plotly figure object for showing the amplitude of a waveform (y-axis in μV) over time (x-axis).
    """
    waveform_df = pd.DataFrame(data={"waveform": waveform})
    waveform_df["timestamp"] = waveform_df.index / sampling_rate

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=waveform_df["timestamp"],
            y=waveform_df["waveform"],
            mode="lines",
            line=dict(color="rgb(0, 160, 223)", width=2),  # DataJoint Blue
            hovertemplate="%{y:.2f} μV<br>" + "%{x:.2f} ms<extra></extra>",
        )
    )
    fig.update_layout(
        title="Avg. waveform",
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (μV)",
        template="simple_white",
        width=350,
        height=350,
    )
    return fig


def plot_auto_correlogram(
    spike_times: np.ndarray, bin_size: float = 0.001, window_size: int = 1
) -> go.Figure:
    """Plot the auto-correlogram of a unit.

    Args:
        spike_times (np.ndarray): Spike timestamps in seconds
        bin_size (float, optional): Size of the time bin (lag) in seconds. Defaults to 0.001.
        window_size (int, optional): Size of the correlogram window in seconds. Defaults to 1 (± 500ms)

    Returns:
        go.Figure: Plotly figure object for showing
        counts (y-axis) over time lags (x-axis).
    """
    from .corr import acorr

    correlogram = acorr(
        spike_times=spike_times, bin_size=bin_size, window_size=window_size
    )
    df = pd.DataFrame(
        data={"correlogram": correlogram},
        index=pd.RangeIndex(
            start=-(window_size * 1e3) / 2,
            stop=(window_size * 1e3) / 2 + bin_size * 1e3,
            step=bin_size * 1e3,
        ),
    )
    df["lags"] = df.index  # in ms

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["lags"],
            y=df["correlogram"],
            mode="lines",
            line=dict(color="black", width=1),
            hovertemplate="%{y}<br>" + "%{x:.2f} ms<extra></extra>",
        )
    )
    fig.update_layout(
        title="Auto Correlogram",
        xaxis_title="Lags (ms)",
        yaxis_title="Count",
        template="simple_white",
        width=350,
        height=350,
        yaxis_range=[0, None],
    )
    return fig


def plot_depth_waveforms(
    ephys: Module,
    unit_key: dict[str, Any],
    y_range: float = 60,
) -> go.Figure:
    """Plot the peak waveform (in red) and waveforms from its neighboring sites on a spatial coordinate.

    Args:
        ephys (Module): Imported ephys module object.
        unit_key (dict[str, Any]): Key dictionary from ephys.CuratedClustering.Unit table.
        y_range (float, optional): Vertical range to show waveforms relative to the peak waveform in μm. Defaults to 60.

    Returns:
        go.Figure: Plotly figure object.
    """

    sampling_rate = (ephys.EphysRecording & unit_key).fetch1(
        "sampling_rate"
    ) / 1e3  # in kHz

    probe_type, peak_electrode = (ephys.CuratedClustering.Unit & unit_key).fetch1(
        "probe_type", "electrode"
    )  # electrode where the peak waveform was found

    electrodes, coord_y = (
        probe.ProbeType.Electrode & f"probe_type='{probe_type}'"
    ).fetch("electrode", "y_coord")

    peak_electrode_shank = (
        probe.ProbeType.Electrode
        & f"probe_type='{probe_type}'"
        & f"electrode={peak_electrode}"
    ).fetch1("shank")

    peak_coord_y = coord_y[electrodes == peak_electrode][0]

    coord_ylim_low = (
        coord_y.min()
        if (peak_coord_y - y_range) <= coord_y.min()
        else peak_coord_y - y_range
    )
    coord_ylim_high = (
        coord_y.max()
        if (peak_coord_y + y_range) >= coord_y.max()
        else peak_coord_y + y_range
    )

    tbl = (
        (probe.ProbeType.Electrode)
        & f"probe_type = '{probe_type}'"
        & f"y_coord BETWEEN {coord_ylim_low} AND {coord_ylim_high}"
        & f"shank={peak_electrode_shank}"
    )
    electrodes_to_plot, x_coords, y_coords = tbl.fetch(
        "electrode", "x_coord", "y_coord"
    )

    coords = np.array([x_coords, y_coords]).T  # x, y coordinates

    waveforms = (
        ephys.WaveformSet.Waveform
        & unit_key
        & f"electrode IN {tuple(electrodes_to_plot)}"
    ).fetch("waveform_mean")
    waveforms = np.stack(waveforms)  # all mean waveforms of a given neuron

    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])

    # Spacing between recording sites (in um)
    x_inc = np.diff(np.sort((coords[coords[:, 1] == coords[0, 1]][:, 0]))).mean() / 2
    y_inc = np.diff(np.sort((coords[coords[:, 0] == coords[0, 0]][:, 1]))).mean() / 2
    time = np.arange(waveforms.shape[1]) / sampling_rate

    x_scale_factor = x_inc / (time[-1] + 1 / sampling_rate)  # correspond to 1 ms
    time_scaled = time * x_scale_factor

    wf_amps = waveforms.max(axis=1) - waveforms.min(axis=1)
    max_amp = wf_amps.max()
    y_scale_factor = y_inc / max_amp

    unique_x_loc = np.sort(np.unique(coords[:, 0]))
    xtick_label = [str(int(x)) for x in unique_x_loc]
    xtick_loc = time_scaled[len(time_scaled) // 2 + 1] + unique_x_loc

    # Plot figure
    fig = go.Figure()
    for electrode, wf, coord in zip(electrodes_to_plot, waveforms, coords):
        wf_scaled = wf * y_scale_factor
        wf_scaled -= wf_scaled.mean()
        color = "red" if electrode == peak_electrode else "rgb(51, 76.5, 204)"

        fig.add_trace(
            go.Scatter(
                x=time_scaled + coord[0],
                y=wf_scaled + coord[1],
                mode="lines",
                line=dict(color=color, width=1.5),
                hovertemplate=f"electrode {electrode}<br>"
                + f"x ={coord[0]: .0f} μm<br>"
                + f"y ={coord[1]: .0f} μm<extra></extra>",
            )
        )
        fig.update_layout(
            title="Depth Waveforms",
            xaxis_title="Electrode position (μm)",
            yaxis_title="Distance from the probe tip (μm)",
            template="simple_white",
            width=400,
            height=600,
            xaxis_range=[x_min - x_inc / 2, x_max + x_inc * 1.2],
            yaxis_range=[y_min - y_inc * 2, y_max + y_inc * 2],
        )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickvals=xtick_loc, ticktext=xtick_label)

    # Add a scale bar
    x0 = xtick_loc[0] - (x_scale_factor * 1.5)
    y0 = y_min - (y_inc * 1.5)

    fig.add_trace(
        go.Scatter(
            x=[x0, x0 + x_scale_factor],
            y=[y0, y0],
            mode="lines",
            line=dict(color="black", width=2),
            hovertemplate="1 ms<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x0, x0],
            y=[y0, y0 + y_inc],
            mode="lines",
            line=dict(color="black", width=2),
            hovertemplate=f"{max_amp: .2f} μV<extra></extra>",
        )
    )
    return fig
