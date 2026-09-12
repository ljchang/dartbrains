"""Cluster-extent FWE needs spatially smooth data. SimulateGrid's noise is iid,
so the chapter must smooth it. Validate calibration with and without smoothing,
using a permutation p-value (handles integer ties better than a percentile cut).
"""
import numpy as np
from scipy.ndimage import label, gaussian_filter
from nltools.algorithms.inference import one_sample_permutation_test

W, N_SUB, N_PERM, N_SIM, ALPHA = 30, 20, 500, 100, 0.05

def largest_cluster(mask2d):
    lab, n = label(mask2d)
    return np.bincount(lab.ravel())[1:].max() if n else 0

def cluster_fwe_p(obs_map, null, w, forming):
    """p = P(max null cluster size >= largest observed cluster)."""
    null_max = np.array([largest_cluster((np.abs(p) > forming).reshape(w, w)) for p in null])
    obs_size = largest_cluster((np.abs(obs_map) > forming).reshape(w, w))
    if obs_size == 0:
        return 1.0, 0
    return (np.sum(null_max >= obs_size) + 1) / (len(null_max) + 1), obs_size

for fwhm in [0, 2.0]:
    rng = np.random.default_rng(0)
    hits, sizes = 0, []
    for s in range(N_SIM):
        d = rng.standard_normal((N_SUB, W, W))
        if fwhm:
            d = np.stack([gaussian_filter(x, sigma=fwhm) for x in d])
            d /= d.std()                              # renormalize after smoothing
        flat = d.reshape(N_SUB, -1)
        res = one_sample_permutation_test(flat, n_permute=N_PERM, return_null=True,
                                          random_state=s, parallel="cpu")
        null = np.asarray(res["null_dist"])
        forming = np.percentile(np.abs(null), 99)
        p, size = cluster_fwe_p(res["mean"], null, W, forming)
        hits += p < ALPHA
        sizes.append(size)
    r = hits / N_SIM
    print(f"smoothing sigma={fwhm:<4} cluster-extent FWE rate = {r:5.1%} "
          f"(+/-{1.96*(r*(1-r)/N_SIM)**0.5:.1%})   median observed cluster = {np.median(sizes):.0f} vox")
