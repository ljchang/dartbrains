# Thresholding performance investigation

`thresholding_jit.py` compares SciPy, direct NumPy, and Numba on identical seeded
data. `thresholding_jit_results.json` records a local Apple Silicon run with
Python 3.13.12, NumPy 2.4.4, SciPy 1.17.1, and Numba 0.68.0. Numerical tests also
passed with Numba 0.64.0 and without Numba. Microbenchmarks use five-repeat medians;
the complete calibration is one run per backend after warmup.

Observed results:

| Work | Baseline | Candidate |
| --- | ---: | ---: |
| 100 maximum-t experiments | SciPy 0.198 s | NumPy 0.084 s; JIT 0.078 s |
| Cluster p-value, 500 maps of 30 × 30 voxels | SciPy 6.84 ms | Numba 0.274 ms |
| Complete two-condition cluster calibration | SciPy 10.66 s | Numba 9.36 s |

The full calibration produced identical FWER estimates (0.01 unsmoothed, 0.03
smoothed) and median cluster sizes (1 and 5). Most time remains in generating
permutations. NumPy matrix multiplication was about 11 times faster than the
experimental JIT loop for signed means, so that loop was not adopted. Compilation
or disk-cache load is reported separately; warm speedups exclude it. No GPU
implementation was benchmarked, and the results do not measure Nilearn TFCE.

## Architecture and release boundary

The adjacent `dartbrains-tools` checkout contains a provisional simulation module
used by this benchmark and the local chapter. Its seeded Gaussian experiments and
finite, four-connected 2D examples are teaching simulations. General max-statistic,
cluster, and TFCE correction belongs in `nltools`, potentially delegating the
spatial inference to Nilearn. Neither the installed `nltools==0.6.0.dev2` nor the
inspected source checkout exposes those correction APIs; the chapter uses Nilearn
for production spatial inference.

The local course environment uses an editable install of `../dartbrains-tools`.
The currently locked registry release does not contain the provisional module.
Do not deploy this chapter before settling that API boundary and releasing the
supporting tools package; then raise its dependency floor, refresh the course
lockfile, synchronize notebook dependency metadata, and verify a clean install.
No package, assignments, or book deployment was published by this investigation.

To reproduce from these adjacent checkouts, install the local tools package with
its `simulation` extra into the course environment, then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/thresholding_jit.py
```
