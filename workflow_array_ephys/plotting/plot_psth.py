import matplotlib.pyplot as plt
import numpy as np


def _plot_spike_raster(
    aligned_spikes, trial_ids=None, vlines=[0], ax=None, title="", xlim=None
):
    if not ax:
        fig, ax = plt.subplots(1, 1)

    raster = np.concatenate(aligned_spikes)
    if trial_ids is None:
        trial_ids = range(len(aligned_spikes))

    trial_ids = np.concatenate(
        [[t] * len(s) for t, s in zip(trial_ids, aligned_spikes)]
    ).astype(int)

    assert len(raster) == len(trial_ids)

    ax.plot(raster, trial_ids, "ro", markersize=4)

    for x in vlines:
        ax.axvline(x=x, linestyle="--", color="k")

    ax.set_ylabel("Trial (#)")
    if xlim:
        ax.set_xlim(xlim)
    ax.set_title(title)


def _plot_psth(psth, psth_edges, bin_size, vlines=[0], ax=None, title="", xlim=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)

    ax.bar(psth_edges, psth, width=bin_size, edgecolor="black", align="edge")

    for x in vlines:
        ax.axvline(x=x, linestyle="--", color="k")

    ax.set_ylabel("spikes/s")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
