"""Code adapted from International Brain Laboratory, T. (2021). ibllib [Computer software]. https://github.com/int-brain-lab/ibllib
"""

import numpy as np


def _index_of(arr: np.ndarray, lookup: np.ndarray):
    """Replace scalars in an array by their indices in a lookup table."""

    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


def _increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account.
    """

    bbins = np.bincount(indices)
    arr[: len(bbins)] += bbins
    return arr


def _diff_shifted(arr, steps=1):
    return arr[steps:] - arr[: len(arr) - steps]


def _create_correlograms_array(n_clusters, winsize_bins):
    return np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1), dtype=np.int32)


def _symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays."""

    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # We symmetrize c[i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    correlograms[..., 0] = np.maximum(correlograms[..., 0], correlograms[..., 0].T)

    sym = correlograms[..., 1:][..., ::-1]
    sym = np.transpose(sym, (1, 0, 2))

    return np.dstack((sym, correlograms))


def xcorr(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    bin_size: float,
    window_size: int,
) -> np.ndarray:
    """Compute all pairwise cross-correlograms among the clusters appearing in `spike_clusters`.

    Args:
        spike_times (np.ndarray): Spike times in seconds.
        spike_clusters (np.ndarray): Spike-cluster mapping.
        bin_size (float): Size of the time bin in seconds.
        window_size (int): Size of the correlogram window in seconds.

    Returns:
         np.ndarray: cross-correlogram array
    """
    assert np.all(np.diff(spike_times) >= 0), "The spike times must be increasing."
    assert spike_times.ndim == 1
    assert spike_times.shape == spike_clusters.shape

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds

    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds
    winsize_bins = 2 * int(0.5 * window_size / bin_size) + 1

    # Take the cluster order into account.
    clusters = np.unique(spike_clusters)
    n_clusters = len(clusters)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _index_of(spike_clusters, clusters)

    # Shift between the two copies of the spike trains.
    shift = 1

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_times, dtype=bool)

    correlograms = _create_correlograms_array(n_clusters, winsize_bins)

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Interval between spike i and spike i+shift.
        spike_diff = _diff_shifted(spike_times, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = np.round(spike_diff / bin_size).astype(np.int64)

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b > (winsize_bins / 2)] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = np.ravel_multi_index(
            (spike_clusters_i[:-shift][m], spike_clusters_i[+shift:][m], d),
            correlograms.shape,
        )

        # Increment the matching spikes in the correlograms array.
        _increment(correlograms.ravel(), indices)

        shift += 1

    return _symmetrize_correlograms(correlograms)


def acorr(spike_times: np.ndarray, bin_size: float, window_size: int) -> np.ndarray:
    """Compute the auto-correlogram of a unit.

    Args:
        spike_times (np.ndarray): Spike times in seconds.
        bin_size (float, optional): Size of the time bin in seconds.
        window_size (int, optional): Size of the correlogram window in seconds.

    Returns:
        np.ndarray: auto-correlogram array (winsize_samples,)
    """
    xc = xcorr(
        spike_times,
        np.zeros_like(spike_times, dtype=np.int32),
        bin_size=bin_size,
        window_size=window_size,
    )
    return xc[0, 0, :]
