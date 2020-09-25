from functools import partial
from math import floor

import numpy as np


def _num_clusters_histogram(distances, freq_threshold, n_bins_start, max_frac):
    if distances.size == 1:
        return 1

    if not freq_threshold:
        threshold_func = _zero_bins
    else:
        threshold_func = partial(_bins_below_threshold, freq_threshold)

    zero_bins = False
    i = 0
    if max_frac == 1.:
        while not zero_bins:
            hist, edges = np.histogram(distances, bins=n_bins_start + i)
            zero_bins_indices = threshold_func(hist)
            zero_bins = zero_bins_indices.size
            i += 1
        first_gap = zero_bins_indices[0]
        left_bin_edge_first_gap = edges[first_gap]
        gap_idx = (distances <= left_bin_edge_first_gap).sum()
        num_clust = distances.size + 1 - gap_idx
    else:
        max_num_clust = max_frac * (distances.size + 1)
        over_max_num = True
        while over_max_num:
            while (not zero_bins) and over_max_num:
                hist, edges = np.histogram(distances, bins=n_bins_start + i)
                zero_bins_indices = threshold_func(hist)
                zero_bins = zero_bins_indices.size
                i += 1
            first_gap = zero_bins_indices[0]
            left_bin_edge_first_gap = edges[first_gap]
            gap_idx = np.sum(distances <= left_bin_edge_first_gap)
            num_clust = distances.size + 1 - gap_idx
            if num_clust > max_num_clust:
                num_clust = max_num_clust
                break
            else:
                over_max_num = False

    return floor(num_clust)


def _zero_bins(hist):
    return np.flatnonzero(~hist.astype(bool))


def _bins_below_threshold(freq_threshold, hist):
    return np.flatnonzero(hist < freq_threshold)


def _num_clusters_simple(distances, min_gap_size, max_frac):
    # Differences between subsequent elements (padding by the first distance)
    diff = np.ediff1d(distances, to_begin=distances[0])
    gap_indices = np.flatnonzero(diff >= min_gap_size)
    if gap_indices.size:
        num_clust = distances.size + 1 - gap_indices[0]
        if max_frac is None:
            return num_clust
        max_num_clust = max_frac * (distances.size + 1)
        if num_clust > max_num_clust:
            num_clust = max_num_clust
        return floor(num_clust)
    # No big enough gaps -> one cluster
    return 1
