from functools import partial

import numpy as np


def _num_clusters_histogram(distances, freq_threshold, n_bins_start):
    if distances.size == 1:
        return 1

    if not freq_threshold:
        threshold_func = _zero_bins
    else:
        threshold_func = partial(_bins_below_threshold, freq_threshold)
    zero_bins = False
    i = 0
    while not zero_bins:
        hist, edges = np.histogram(distances, bins=n_bins_start + i)
        zero_bins_indices = threshold_func(hist)
        zero_bins = zero_bins_indices.size
        i += 1
    first_gap = zero_bins_indices[0]
    left_bin_edge_first_gap = edges[first_gap]
    gap_idx = (distances <= left_bin_edge_first_gap).sum()
    num_clust = distances.size + 1 - gap_idx
    return num_clust


def _zero_bins(hist):
    return np.flatnonzero(~hist.astype(bool))


def _bins_below_threshold(freq_threshold, hist):
    return np.flatnonzero(hist < freq_threshold)


def _num_clusters_simple(distances, min_gap_size):
    # Differences between subsequent elements (padding by the first
    # distance)
    diff = np.ediff1d(distances, to_begin=distances[0])
    gap_indices = np.flatnonzero(diff >= min_gap_size)
    if gap_indices.size:
        num_clust = distances.size + 1 - gap_indices[0]
        return num_clust
    # No big enough gaps -> one cluster
    return 1
