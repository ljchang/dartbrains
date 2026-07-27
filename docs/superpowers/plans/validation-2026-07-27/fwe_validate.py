"""Under the global null the family-wise false-positive rate must be ~alpha.

Classical corrections are evaluated on PARAMETRIC t-test p-values; the FWE
procedures use the permutation null. Mixing them is a mistake: a permutation
p-value cannot go below ~1/n_permute, so Bonferroni over 900 voxels is
unreachable with 500 permutations.
"""
import numpy as np
from scipy.ndimage import label
from scipy.stats import ttest_1samp
from nltools.algorithms.inference import one_sample_permutation_test
from nltools.stats import fdr

W, N_SUB, N_PERM, N_SIM, ALPHA = 30, 20, 500, 100, 0.05
NV = W * W

def max_stat_threshold(null, alpha=ALPHA):
    return np.percentile(np.abs(null).max(axis=1), 100 * (1 - alpha))

def largest_cluster(mask2d):
    lab, n = label(mask2d)
    return np.bincount(lab.ravel())[1:].max() if n else 0

def cluster_extent_threshold(null, forming, w, alpha=ALPHA):
    sizes = [largest_cluster((np.abs(p) > forming).reshape(w, w)) for p in null]
    return np.percentile(sizes, 100 * (1 - alpha))

rng = np.random.default_rng(0)
hits = dict(unc_001=0, bonferroni=0, fdr_05=0, perm_maxstat=0, perm_cluster=0)
for s in range(N_SIM):
    data = rng.standard_normal((N_SUB, NV))                # pure noise

    _, p_param = ttest_1samp(data, popmean=0, axis=0)       # parametric family
    hits["unc_001"]    += (p_param < 0.001).any()
    hits["bonferroni"] += (p_param < ALPHA / NV).any()
    q = fdr(p_param, q=ALPHA)
    hits["fdr_05"]     += bool(q > 0 and (p_param <= q).any())

    res = one_sample_permutation_test(data, n_permute=N_PERM, return_null=True,
                                      random_state=s, parallel="cpu")
    obs, null = np.abs(res["mean"]), np.asarray(res["null_dist"])   # permutation family
    hits["perm_maxstat"] += (obs > max_stat_threshold(null)).any()

    forming = np.percentile(np.abs(null), 99)
    k_crit = cluster_extent_threshold(null, forming, W)
    lab_o, n_o = label((obs > forming).reshape(W, W))
    sizes_o = np.bincount(lab_o.ravel())[1:] if n_o else np.array([0])
    hits["perm_cluster"] += (sizes_o > k_crit).any()

print(f"Family-wise false-positive rate, {N_SIM} null simulations, "
      f"{W}x{W}={NV} voxels, alpha={ALPHA}\n")
for k, v in hits.items():
    r = v / N_SIM
    se = (r * (1 - r) / N_SIM) ** 0.5
    print(f"  {k:14s} {r:6.1%}  +/-{1.96*se:.1%}   ({v}/{N_SIM} maps with >=1 false positive)")
print(f"\n  (permutation p-value floor with n_permute={N_PERM}: {1/(N_PERM+1):.4f}; "
      f"Bonferroni target {ALPHA/NV:.2e} is unreachable)")
