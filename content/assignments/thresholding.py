# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "scipy", "matplotlib", "nltools @ git+https://github.com/cosanlab/nltools@master", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "8787ad5f,cfdbdc3c,8af6cf8a,e015f812,65f164fe,304b4ede,5d776121,961aaec3,67deb9bf,49d7e4df,a6b875e0,113232a0,1a5960ed,859b98a1,8f765f0f,a7b3bdc6,c5b4c8c6,1f5034a7,e3431dcb,2bbce784,3b2cadd7,46cc9e0c,2e70bb61,f8eb9e26,d1a5dd83,7e01de6b,51157800,0bb7884c,b01c8d44,7d160fcd"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "thresholding"
# grader-assignment-id = "d228d348-c31c-4a58-b963-35ef7f9205aa"
# grader-assignment-version = "8d337f1b-44e2-49c5-b7ae-80d4eae46c33"
# grader-version = "1"
# ///
"""DartBrains assignment: Thresholding Group Analyses."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import contextlib
    import io

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.ndimage import label, gaussian_filter

    from nltools import SimulateGrid
    from nltools.algorithms.inference import one_sample_permutation_test

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return (
        SimulateGrid,
        contextlib,
        gaussian_filter,
        g,
        io,
        label,
        mo,
        np,
        one_sample_permutation_test,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: Thresholding Group Analyses

        These are the **Exercises** at the end of the
        [Thresholding Group Analyses chapter](https://dartbrains.org/content/Thresholding_Group_Analyses.html).

        Everything here is a simulation, so you know the ground truth — which is the whole
        point. In real data you can never measure your own false positive rate. Here you can,
        and that is the only way to develop intuitions about what a correction is actually
        buying you.

        Two practical notes:

        - **Every question pins its seeds and parameters.** Simulation answers move around
          from run to run, so the questions fix the amplitudes, the seeds and the grid size.
          Use exactly the values given or your numbers will not match the checks.
        - **The whole notebook runs in well under a minute.** If a cell seems to hang, you have
          probably called `plot_grid_simulation()` inside a loop, which renders a figure per
          iteration. Call `fit()` and `threshold_simulation()` directly instead.

        Sign in once at the top.
        """
    )
    return


@app.cell
def _(g):
    g.signin_button()
    return


@app.cell
def _():
    # === MOGRADER: MARKS ===
    _marks = {"th-q01": 5, "th-q02": 5, "th-q03": 5, "th-q04": 5, "th-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup

        Run this cell as it is.

        `quiet()` silences the progress bars that `one_sample_permutation_test` prints, so the
        loops below do not bury the notebook in output. `max_stat_threshold`, `largest_cluster`
        and `cluster_fwe_p` are copied unchanged from the chapter.

        `run_cluster_calibration` is the chapter's function with one addition: a
        `forming_percentile` argument, so you can change the cluster-forming threshold by
        passing a number rather than by editing the function body.
        """
    )
    return


@app.cell
def _(contextlib, gaussian_filter, io, label, np, one_sample_permutation_test):
    @contextlib.contextmanager
    def quiet():
        """Suppress the progress bars printed by nltools' permutation routines."""
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield

    def max_stat_threshold(null, alpha=0.05):
        """Critical value from the null distribution of the maximum statistic.

        ``null`` has shape (n_permutations, n_voxels). Taking the max across voxels
        collapses each permutation to a single number: the most extreme value anywhere
        in that permuted map.
        """
        return np.percentile(np.abs(null).max(axis=1), 100 * (1 - alpha))

    def largest_cluster(mask2d):
        """Size in voxels of the largest contiguous blob in a boolean grid."""
        labelled, n_found = label(mask2d)
        return np.bincount(labelled.ravel())[1:].max() if n_found else 0

    def cluster_fwe_p(observed, null, width, forming):
        """p = P(largest null cluster >= largest observed cluster)."""
        null_max = np.array(
            [largest_cluster((np.abs(perm) > forming).reshape(width, width)) for perm in null]
        )
        observed_size = largest_cluster((np.abs(observed) > forming).reshape(width, width))
        if observed_size == 0:
            return 1.0, 0, null_max
        p_value = (np.sum(null_max >= observed_size) + 1) / (len(null_max) + 1)
        return p_value, observed_size, null_max

    def run_cluster_calibration(
        smoothing_sigma,
        forming_percentile=99,
        n_sims=100,
        width=30,
        n_subjects=20,
        n_permute=500,
        alpha=0.05,
    ):
        """Family-wise error rate of the cluster test under the complete null.

        Identical to the chapter's version except that the cluster-forming threshold is
        now a parameter (``forming_percentile``) rather than a hard-coded 99.

        Returns (fwe_rate, median_largest_cluster).
        """
        rng = np.random.default_rng(2)
        hits, observed_sizes = 0, []
        with quiet():
            for i in range(n_sims):
                data = rng.standard_normal((n_subjects, width, width))
                if smoothing_sigma:
                    data = np.stack([gaussian_filter(x, sigma=smoothing_sigma) for x in data])
                    data = data / data.std()  # renormalize after smoothing
                res = one_sample_permutation_test(
                    data.reshape(n_subjects, -1),
                    n_permute=n_permute,
                    return_null=True,
                    random_state=i,
                    n_jobs=1,
                )
                null = np.asarray(res["null_dist"])
                forming = np.percentile(np.abs(null), forming_percentile)
                p_value, size, _ = cluster_fwe_p(res["mean"], null, width, forming)
                hits += p_value < alpha
                observed_sizes.append(size)
        return hits / n_sims, float(np.median(observed_sizes))

    return (
        cluster_fwe_p,
        largest_cluster,
        max_stat_threshold,
        quiet,
        run_cluster_calibration,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. How big does the signal have to be to survive Bonferroni?

        A Bonferroni correction on a 100 x 100 grid means testing each voxel at
        $\alpha = 0.05 / 10{,}000$. That is a brutally conservative threshold. The question is
        what it costs you in sensitivity: **how large does the signal have to be before you
        recover all of it?**

        Sweep the signal amplitude and record how much of the true signal survives.

        For each amplitude in `AMPLITUDES`, build a
        `SimulateGrid(signal_amplitude=amp, signal_width=10, grid_width=100, n_subjects=20,
        random_state=0)`, call `.fit()`, then `.threshold_simulation(threshold=BONFERRONI,
        threshold_type="p")`, and read off `.tp_percent` — the proportion of true signal
        voxels recovered.

        Produce two things:

        - `recovery` — a `pandas.Series` indexed by amplitude, holding `tp_percent`.
        - `min_amplitude` — the smallest amplitude in the sweep that recovers **100%** of the
          signal.

        *Do not call `plot_grid_simulation()` here — it runs 100 extra simulations and draws a
        figure each time. `fit()` then `threshold_simulation()` is all you need, and the whole
        sweep takes a couple of seconds.*
        """
    )
    return


@app.cell
def _(np):
    GRID_WIDTH = 100
    SIGNAL_WIDTH = 10
    N_SUBJECTS = 20
    BONFERRONI = 0.05 / (GRID_WIDTH**2)
    AMPLITUDES = np.round(np.arange(0.0, 3.01, 0.25), 2)

    print(f"Bonferroni threshold: p < {BONFERRONI}")
    print(f"Amplitudes to sweep:  {AMPLITUDES}")
    return AMPLITUDES, BONFERRONI, GRID_WIDTH, N_SUBJECTS, SIGNAL_WIDTH


@app.cell
def _(
    AMPLITUDES,
    BONFERRONI,
    GRID_WIDTH,
    N_SUBJECTS,
    SIGNAL_WIDTH,
    SimulateGrid,
    pd,
):
    recovery = ...
    min_amplitude = ...
    # YOUR CODE HERE
    pass
    recovery
    return min_amplitude, recovery


@app.cell
def _(mo, plt, recovery):
    mo.stop(recovery is ..., mo.md("*The recovery curve appears once Q1 is answered.*"))

    _f, _a = plt.subplots(figsize=(8, 4))
    _a.plot(recovery.index, recovery.values, marker="o")
    _a.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    _a.set_xlabel("Signal amplitude")
    _a.set_ylabel("Proportion of true signal recovered")
    _a.set_title("Sensitivity of a Bonferroni-corrected test")
    plt.gcf()
    return


@app.cell
def _(g, min_amplitude, mo, recovery):
    mo.stop(
        recovery is ... or min_amplitude is ...,
        mo.md("**Complete the code cell above first.** This check runs once `recovery` and `min_amplitude` are defined."),
    )

    _values = [float(v) for v in recovery.values]
    g.check(
        "th-q01: How big does the signal have to be to survive Bonferroni?",
        [
            (
                len(recovery) == 13,
                "recovery should have one entry per amplitude in AMPLITUDES (13 of them)",
                1,
            ),
            (
                _values[0] == 0.0,
                "at zero amplitude there is no signal, so nothing should be recovered",
                1,
            ),
            (
                2.0 <= float(min_amplitude) <= 2.5,
                "the signal has to be roughly twice the noise sd before Bonferroni recovers all of it",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("th-q01")
    return


@app.cell
def _(g):
    g.feedback("th-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. What does FDR buy you over Bonferroni?

        Bonferroni controls the probability of making **even one** false positive. FDR controls
        the *proportion* of your positives that are false. That is a weaker guarantee, and it
        should buy you sensitivity.

        Measure the trade at a signal amplitude of **1.5** — a level where Q1 showed Bonferroni
        struggling. Build two simulations with identical settings (`signal_width=10`,
        `grid_width=100`, `n_subjects=20`, `random_state=0`, `signal_amplitude=1.5`) and
        threshold them two ways:

        - Bonferroni: `threshold_simulation(threshold=BONFERRONI, threshold_type="p")`
        - FDR: `threshold_simulation(threshold=0.05, threshold_type="q", correction="fdr")`

        Collect the results in a dictionary called `comparison` with exactly these four keys:

        ```python
        comparison = {
            "bonferroni_tp": ...,   # .tp_percent under Bonferroni
            "bonferroni_fp": ...,   # .fp_percent under Bonferroni
            "fdr_tp": ...,          # .tp_percent under FDR
            "fdr_fp": ...,          # .fp_percent under FDR
        }
        ```

        *`tp_percent` is the proportion of true signal voxels recovered; `fp_percent` is the
        proportion of the grid that is a false positive. Note the asymmetry in what each number
        is a proportion **of** — that is why the false positive numbers look so small.*
        """
    )
    return


@app.cell
def _(BONFERRONI, GRID_WIDTH, N_SUBJECTS, SIGNAL_WIDTH, SimulateGrid):
    comparison = ...
    # YOUR CODE HERE
    pass
    comparison
    return (comparison,)


@app.cell
def _(comparison, g, mo):
    mo.stop(
        comparison is ...,
        mo.md("**Complete the code cell above first.** This check runs once `comparison` is defined."),
    )

    _keys = {"bonferroni_tp", "bonferroni_fp", "fdr_tp", "fdr_fp"}
    g.check(
        "th-q02: What does FDR buy you over Bonferroni?",
        [
            (
                isinstance(comparison, dict) and set(comparison) == _keys,
                f"comparison should be a dict with exactly the keys {sorted(_keys)}",
                1,
            ),
            (
                abs(float(comparison["bonferroni_tp"]) - 0.63) < 0.06,
                "Bonferroni should recover about 63% of the signal at this amplitude",
                2,
            ),
            (
                float(comparison["fdr_tp"]) > float(comparison["bonferroni_tp"]),
                "FDR should be the more sensitive of the two",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("th-q02")
    return


@app.cell
def _(g):
    g.feedback("th-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. How many permutations do you need?

        The chapter's max-statistic section used 1,000 permutations. Was that enough? There are
        two separate things to worry about, and they have different answers.

        The first is a hard floor: with $n$ permutations the smallest p-value you can possibly
        observe is $1/(n+1)$. No amount of real signal gets you below it.

        The second is stability — how much your threshold moves if you just change the random
        seed. That is the one that matters in practice, and the only way to see it is to run the
        same thing several times.

        For each `n_permute` in `(100, 500, 5000)`, run `one_sample_permutation_test` on `DATA`
        **five times**, with `random_state` 0 through 4, and each time compute
        `max_stat_threshold(null)` where `null = np.asarray(result["null_dist"])`.

        Build a `pandas.DataFrame` called `permutation_stability`, indexed by `n_permute`, with
        two columns:

        - `min_p` — the smallest attainable p-value, $1/(n+1)$.
        - `threshold_spread` — `max(thresholds) - min(thresholds)` across the five seeds.

        *Wrap the loop in `with quiet():` to suppress the progress bars. Pass `n_jobs=1` and
        `return_null=True` to `one_sample_permutation_test`.*
        """
    )
    return


@app.cell
def _(np):
    DATA = np.random.default_rng(0).standard_normal((20, 900))  # 20 subjects, a 30x30 grid of pure noise
    PERMUTATION_COUNTS = (100, 500, 5000)
    DATA.shape
    return DATA, PERMUTATION_COUNTS


@app.cell
def _(
    DATA,
    PERMUTATION_COUNTS,
    max_stat_threshold,
    np,
    one_sample_permutation_test,
    pd,
    quiet,
):
    permutation_stability = ...
    # YOUR CODE HERE
    pass
    permutation_stability
    return (permutation_stability,)


@app.cell
def _(g, mo, permutation_stability):
    mo.stop(
        permutation_stability is ...,
        mo.md("**Complete the code cell above first.** This check runs once `permutation_stability` is defined."),
    )

    _spread = [float(v) for v in permutation_stability["threshold_spread"]]
    _minp = [float(v) for v in permutation_stability["min_p"]]
    g.check(
        "th-q03: How many permutations do you need?",
        [
            (
                len(permutation_stability) == 3
                and {"min_p", "threshold_spread"} <= set(permutation_stability.columns),
                "the frame needs one row per permutation count and both columns",
                1,
            ),
            (
                abs(_minp[0] - 1 / 101) < 1e-9 and abs(_minp[2] - 1 / 5001) < 1e-9,
                "min_p is 1/(n+1): about 0.0099 at 100 permutations and 0.0002 at 5000",
                2,
            ),
            (
                _spread[2] < _spread[0],
                "the threshold should be more stable across seeds with more permutations",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("th-q03")
    return


@app.cell
def _(g):
    g.feedback("th-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. The cluster-forming threshold is a researcher degree of freedom

        The chapter's cluster simulation used the 99th percentile of the null as its
        cluster-forming threshold. Nothing forced that choice — and Woo et al. (2014) argue
        that this freedom is a problem.

        Run `run_cluster_calibration(2.0, forming_percentile=p)` for `p` in `(95, 99)` — smoothed
        data, complete null, so *every* cluster it finds is a false positive by construction.
        The function returns `(fwe_rate, median_largest_cluster)`.

        Collect the results in a `pandas.DataFrame` called `forming_comparison`, indexed by the
        forming percentile, with columns `fwe_rate` and `median_cluster`.

        Then answer the question the chapter asks, by assigning a boolean to
        `fwe_stays_nominal`: **does the family-wise error rate stay at roughly 5%?**

        *Each call takes about 7 seconds. Be careful here — the intuitive answer to the last
        part is not the one the simulation gives, and the check is looking for what actually
        happens rather than what you might expect.*
        """
    )
    return


@app.cell
def _(pd, run_cluster_calibration):
    forming_comparison = ...
    fwe_stays_nominal = ...
    # YOUR CODE HERE
    pass
    forming_comparison
    return forming_comparison, fwe_stays_nominal


@app.cell
def _(forming_comparison, fwe_stays_nominal, g, mo):
    mo.stop(
        forming_comparison is ... or fwe_stays_nominal is ...,
        mo.md("**Complete the code cell above first.** This check runs once `forming_comparison` and `fwe_stays_nominal` are defined."),
    )

    _rates = [float(v) for v in forming_comparison["fwe_rate"]]
    _clusters = [float(v) for v in forming_comparison["median_cluster"]]
    g.check(
        "th-q04: The cluster-forming threshold is a researcher degree of freedom",
        [
            (
                len(forming_comparison) == 2
                and {"fwe_rate", "median_cluster"} <= set(forming_comparison.columns),
                "the frame needs a row per forming percentile and both columns",
                1,
            ),
            (
                all(r <= 0.10 for r in _rates),
                "neither forming threshold should blow the error rate far past nominal",
                1,
            ),
            (
                bool(fwe_stays_nominal) is True,
                "the error rate does hold -- nonparametric cluster inference calibrates itself "
                "to whatever forming threshold you chose",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("th-q04")
    return


@app.cell
def _(g):
    g.feedback("th-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. What the liberal threshold actually costs you

        Q4 found that the error rate holds at either forming threshold. So is the choice
        harmless? No — and this question measures the actual cost.

        Under the complete null there is no signal to be specific *about*. So now put a real
        signal in: a graded Gaussian blob at the centre of the grid, on top of smoothed noise.
        Because you built the signal, you know exactly which voxels are real, and you can ask
        of the significant cluster: **how much of what I am reporting is actually signal?**

        Two quantities, for each forming threshold:

        - **precision** — of the voxels in the significant cluster, the fraction that are truly
          signal. Low precision means you are reporting noise as though it were a finding.
        - **recall** — of the truly-signal voxels, the fraction your cluster caught.

        `specificity_at()` below does one simulation and returns both, averaged over 20 seeds.
        Read it, then call it for forming percentiles 95 and 99 and assemble a `pandas.DataFrame`
        called `specificity` indexed by percentile, with columns `mean_size`, `precision` and
        `recall`.

        *This takes a few seconds per threshold. The two numbers together are the Woo et al.
        argument in concrete form — look at how they move in opposite directions.*
        """
    )
    return


@app.cell
def _(gaussian_filter, label, np, one_sample_permutation_test, quiet):
    _W, _NS = 30, 20
    _yy, _xx = np.mgrid[0:_W, 0:_W]
    _centre = (_W - 1) / 2
    BLOB = np.exp(-((_xx - _centre) ** 2 + (_yy - _centre) ** 2) / (2 * 3.5**2))
    TRUTH = BLOB > 0.5  # the voxels we will count as "really" signal

    def _biggest_blob(mask):
        labelled, n_found = label(mask)
        if not n_found:
            return np.zeros_like(mask, dtype=bool)
        return labelled == (np.bincount(labelled.ravel())[1:].argmax() + 1)

    def specificity_at(forming_percentile, n_sims=20, amplitude=1.0, n_permute=500):
        """Mean cluster size, precision and recall at one cluster-forming threshold.

        Smoothed noise plus a graded Gaussian signal at the centre. For each simulation we
        take the largest supra-threshold cluster and compare it against the known signal
        footprint ``TRUTH``.
        """
        sizes, precisions, recalls = [], [], []
        with quiet():
            for seed in range(n_sims):
                rng = np.random.default_rng(seed)
                data = rng.standard_normal((_NS, _W, _W))
                data = np.stack([gaussian_filter(x, sigma=2.0) for x in data])
                data = data / data.std()
                data = data + amplitude * BLOB[None, :, :]

                res = one_sample_permutation_test(
                    data.reshape(_NS, -1),
                    n_permute=n_permute,
                    return_null=True,
                    random_state=0,
                    n_jobs=1,
                )
                null = np.asarray(res["null_dist"])
                observed = np.asarray(res["mean"]).reshape(_W, _W)

                forming = np.percentile(np.abs(null), forming_percentile)
                blob = _biggest_blob(np.abs(observed) > forming)
                inside = (blob & TRUTH).sum()

                sizes.append(blob.sum())
                precisions.append(inside / max(blob.sum(), 1))
                recalls.append(inside / TRUTH.sum())
        return float(np.mean(sizes)), float(np.mean(precisions)), float(np.mean(recalls))

    print(f"True signal footprint: {TRUTH.sum()} voxels of {_W * _W}")
    return TRUTH, specificity_at


@app.cell
def _(pd, specificity_at):
    specificity = ...
    # YOUR CODE HERE
    pass
    specificity
    return (specificity,)


@app.cell
def _(g, mo, specificity):
    mo.stop(
        specificity is ...,
        mo.md("**Complete the code cell above first.** This check runs once `specificity` is defined."),
    )

    _liberal = specificity.loc[95]
    _strict = specificity.loc[99]
    g.check(
        "th-q05: What the liberal threshold actually costs you",
        [
            (
                len(specificity) == 2
                and {"mean_size", "precision", "recall"} <= set(specificity.columns),
                "the frame needs a row per forming percentile and all three columns",
                1,
            ),
            (
                float(_liberal["precision"]) < float(_strict["precision"]),
                "the liberal threshold should be the less precise of the two -- more of what "
                "it reports is noise",
                2,
            ),
            (
                float(_liberal["recall"]) > float(_strict["recall"]),
                "the liberal threshold should catch more of the true signal",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("th-q05")
    return


@app.cell
def _(g):
    g.feedback("th-q05")
    return


if __name__ == "__main__":
    app.run()
