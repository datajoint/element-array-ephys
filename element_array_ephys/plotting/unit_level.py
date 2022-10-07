import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import element_array_ephys.ephys_acute as ephys
from element_array_ephys import probe


def plot_waveform(
    waveform: np.array, sampling_rate: float, fig=None, ax=None
) -> matplotlib.figure.Figure:

    waveform_df = pd.DataFrame(data={"waveform": waveform})
    waveform_df["timestamp"] = waveform_df.index / sampling_rate
    waveform_df.set_index("timestamp", inplace=True)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    ax.plot(waveform_df, "k")
    ax.set(xlabel="Time (ms)", ylabel="Voltage ($\mu$V)", title="Avg. waveform")
    sns.despine()

    return fig


def plot_correlogram(
    spike_times, bin_size=0.001, window_size=1, fig=None, ax=None
) -> matplotlib.figure.Figure:

    from brainbox.singlecell import acorr

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

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    df["lags"] = df.index  # in ms

    ax.plot(df["lags"], df["correlogram"], color="royalblue", linewidth=0.5)
    ymax = round(correlogram.max() / 10) * 10
    ax.set_ylim(0, ymax)
    # ax.axvline(x=0, color="k", linewidth=0.5, ls="--")
    ax.set(xlabel="Lags (ms)", ylabel="Count", title="Auto Correlogram")
    sns.despine()

    return fig


def plot_depth_waveforms(
    probe_type: str,
    unit_id: int,
    sampling_rate: float,
    y_range: float = 50,
    fig=None,
    ax=None,
):

    peak_electrode = (ephys.CuratedClustering.Unit & f"unit={unit_id}").fetch1(
        "electrode"
    )  # electrode where the peak waveform was found

    peak_coord_y = (
        probe.ProbeType.Electrode()
        & f"probe_type='{probe_type}'"
        & f"electrode={peak_electrode}"
    ).fetch1("y_coord")

    coord_y = (probe.ProbeType.Electrode).fetch(
        "y_coord"
    )  # y-coordinate for all electrodes
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
    )
    electrodes_to_plot = tbl.fetch("electrode")

    coords = np.array(tbl.fetch("x_coord", "y_coord")).T  # x, y coordinates

    waveforms = (
        ephys.WaveformSet.Waveform
        & f"unit = {unit_id}"
        & f"electrode IN {tuple(electrodes_to_plot)}"
    ).fetch("waveform_mean")
    waveforms = np.stack(waveforms)  # all mean waveforms of a given neuron

    if ax is None:
        fig, ax = plt.subplots(1, 1, frameon=True, figsize=[1.5, 2], dpi=200)

    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])

    # Spacing between channels (in um)
    x_inc = np.abs(np.diff(coords[:, 0])).min()
    y_inc = np.unique((np.abs(np.diff(coords[:, 1])))).max()

    time = np.arange(waveforms.shape[1]) * (1 / sampling_rate)

    x_scale_factor = x_inc / (time + (1 / sampling_rate))[-1]
    time_scaled = time * x_scale_factor

    wf_amps = waveforms.max(axis=1) - waveforms.min(axis=1)
    max_amp = wf_amps.max()
    y_scale_factor = y_inc / max_amp

    unique_x_loc = np.sort(np.unique(coords[:, 0]))
    xtick_label = list(map(str, map(int, unique_x_loc)))
    xtick_loc = time_scaled[int(len(time_scaled) / 2) + 1] + unique_x_loc

    # Plot the mean waveform for each electrode
    for electrode, wf, coord in zip(electrodes_to_plot, waveforms, coords):

        wf_scaled = wf * y_scale_factor
        wf_scaled -= wf_scaled.mean()

        color = "r" if electrode == peak_electrode else [0.2, 0.3, 0.8] * 255
        ax.plot(
            time_scaled + coord[0], wf_scaled + coord[1], color=color, linewidth=0.5
        )

    ax.set(xlabel="($\mu$m)", ylabel="Distance from the probe tip ($\mu$m)")
    ax.set_ylim([y_min - y_inc * 2, y_max + y_inc * 2])
    ax.xaxis.get_label().set_fontsize(8)
    ax.yaxis.get_label().set_fontsize(8)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_label)
    sns.despine()
    sns.set_style("white")

    return fig
