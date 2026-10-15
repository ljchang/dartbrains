"""Benchmark possible thresholding accelerators; not a production inference API.

Run with NumPy/SciPy/nltools and numba>=0.64 (Python 3.13 / NumPy 2.4):
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/thresholding_jit.py

Identical draws/signs are used for all methods. Report compilation separately
from warm calls. No fastmath, changed precision, GPU, or reduced simulation budget.
Numba is intentionally not a required course dependency. Results go to stdout.
"""
from __future__ import annotations

import json
import platform
import time
from statistics import median

import numba
import numpy as np
import scipy
from numba import njit
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_1samp
from dartbrains_tools.simulations import maximum_cluster_sizes, cluster_fwe_p
from dartbrains_tools._simulation_numba import maximum_cluster_sizes as maximum_clusters
from nltools.algorithms.inference import one_sample_permutation_test
# Private helper used only in this benchmark to match the installed API's draws.
from nltools.algorithms.inference.random import _generate_sign_flips


@njit(cache=True)
def maximum_t(data):
    n, voxels = data.shape
    result = 0.0
    for v in range(voxels):
        mean = 0.0
        for i in range(n):
            mean += data[i, v]
        mean /= n
        ss = 0.0
        for i in range(n):
            ss += (data[i, v] - mean) ** 2
        statistic = abs(mean) / np.sqrt(ss / (n - 1) / n)
        result = max(result, statistic)
    return result


@njit(cache=True)
def signed_means(data, signs):
    n, voxels = data.shape
    result = np.zeros((len(signs), voxels))
    for p in range(len(signs)):
        for i in range(n):
            for v in range(voxels):
                result[p, v] += signs[p, i] * data[i, v]
        for v in range(voxels):
            result[p, v] /= n
    return result




def scipy_clusters(null, width, forming):
    result = []
    for values in null:
        labels, count = label((np.abs(values) > forming).reshape(width, width))
        result.append(np.bincount(labels.ravel())[1:].max() if count else 0)
    return np.asarray(result)


def numpy_maximum_t(data):
    means = data.mean(axis=0)
    return np.max(np.abs(means) / np.sqrt(data.var(axis=0, ddof=1) / len(data)))


def sweep(kernel):
    rng = np.random.default_rng(0)
    return np.array([kernel(rng.standard_normal((20, 10_000))) for _ in range(100)])


def timed(fn, repeat=5):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return median(times)


def calibration(backend):
    """The chapter's complete two-condition calibration, including permutations."""
    results = []
    for sigma in (0, 2.0):
        rng = np.random.default_rng(2)
        hits, sizes = 0, []
        for i in range(100):
            data = rng.standard_normal((20, 30, 30))
            if sigma:
                data = np.stack([gaussian_filter(x, sigma=sigma) for x in data])
                data /= data.std()
            res = one_sample_permutation_test(data.reshape(20, -1), n_permute=500,
                                             return_null=True, random_state=i, n_jobs=1)
            null = np.asarray(res['null_dist'])
            p, size, _ = cluster_fwe_p(res['mean'], null, 30, np.percentile(np.abs(null), 99), backend=backend)
            hits += p < .05
            sizes.append(size)
        results.append([hits / 100, float(np.median(sizes))])
    return results


def main():
    data = np.random.default_rng(0).standard_normal((20, 900))
    large = np.random.default_rng(0).standard_normal((20, 10_000))
    signs = _generate_sign_flips(500, 20, random_state=0)
    null = signs.astype(float) @ data / len(data)
    forming = np.percentile(np.abs(null), 99)
    first_calls = {}
    for name, fn in [("max_t", lambda: maximum_t(large)), ("signed_means", lambda: signed_means(data, signs)), ("clusters", lambda: maximum_clusters(null, 30, forming))]:
        start = time.perf_counter()
        fn()
        first_calls[name] = time.perf_counter() - start

    reference = one_sample_permutation_test(data, n_permute=500, n_jobs=1, random_state=0, return_null=True)
    np.testing.assert_allclose(null, reference['null_dist'], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(signed_means(data, signs), reference['null_dist'], rtol=1e-13, atol=1e-14)
    scipy_max = lambda x: np.abs(ttest_1samp(x, 0, axis=0).statistic).max()
    reference_maxima = sweep(scipy_max)
    for kernel in [numpy_maximum_t, maximum_t]:
        candidate = sweep(kernel)
        np.testing.assert_allclose(candidate, reference_maxima, rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(candidate[:, None] >= np.arange(3, 7.01, .1), reference_maxima[:, None] >= np.arange(3, 7.01, .1))
    for values in [null, np.zeros((5, 900)), np.ones((5, 900)), np.eye(900)[:5]]:
        np.testing.assert_array_equal(maximum_clusters(values, 30, forming), scipy_clusters(values, 30, forming))

    report = {
        'versions': dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__, machine=platform.machine()),
        'first_call_seconds_compilation_or_disk_cache': first_calls,
        'warm_median_seconds': {
            'sweep_scipy': timed(lambda: sweep(scipy_max)),
            'sweep_numpy_formula': timed(lambda: sweep(numpy_maximum_t)),
            'sweep_jit': timed(lambda: sweep(maximum_t)),
            'random_draws_only': timed(lambda: sweep(lambda x: x[0, 0])),
            'max_t_kernel_scipy': timed(lambda: scipy_max(large)),
            'max_t_kernel_jit': timed(lambda: maximum_t(large)),
            'permutation_nltools_including_sign_generation': timed(lambda: one_sample_permutation_test(data, n_permute=500, n_jobs=1, random_state=0, return_null=True)),
            'sign_generation_only': timed(lambda: _generate_sign_flips(500, 20, random_state=0)),
            'signed_means_numpy_pregenerated_signs': timed(lambda: signs.astype(float) @ data / len(data)),
            'signed_means_jit_pregenerated_signs': timed(lambda: signed_means(data, signs)),
            'clusters_scipy_500_maps': timed(lambda: scipy_clusters(null, 30, forming)),
            'clusters_jit_500_maps': timed(lambda: maximum_clusters(null, 30, forming)),
            'clusters_library_scipy_500_maps': timed(lambda: maximum_cluster_sizes(null, 30, forming, backend='scipy')),
            'clusters_library_numba_500_maps': timed(lambda: maximum_cluster_sizes(null, 30, forming, backend='numba')),
            'cluster_p_scipy': timed(lambda: cluster_fwe_p(data.mean(axis=0), null, 30, forming, backend='scipy')),
            'cluster_p_numba': timed(lambda: cluster_fwe_p(data.mean(axis=0), null, 30, forming, backend='numba')),
        },
        'validation': 'max t matches within 1e-13; all tested threshold decisions and cluster sizes match exactly; signed means match installed nltools within 1e-13',
        'scope': 'Finite float64 synthetic inputs only; matrix-mean timing excludes p-value calculation, sign generation, and smoothing. Not a drop-in general inference implementation.',
    }
    calibration_times = {}
    for backend in ('scipy', 'numba'):
        start = time.perf_counter()
        result = calibration(backend)
        calibration_times[backend] = dict(seconds=time.perf_counter() - start, result=result)
    assert calibration_times['scipy']['result'] == calibration_times['numba']['result']
    report['full_calibration_single_run_after_warmup'] = calibration_times
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
