# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "nltools @ git+https://github.com/cosanlab/nltools@master", "dartbrains-tools>=0.1.8", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "50c97552,35eaae33,8af6cf8a,b3f8529a,4c938411,c1c06b88,c47f831c,912f6d1c,6fb7c219,c873209e,8f224269,a373c696,2db25adc,107da8fb,391b3e45,c32a7bfa,b6c455f1,3e1fa387,8abe3290,104ae5bf,595ee5d1,de49eddb,c3eb2298,843008c6,380de0df,61645d98"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "neuroimaging-data"
# grader-assignment-id = "18a54570-5d16-4a10-a86f-ba17396ed504"
# grader-assignment-version = "fcdd661b-4a69-4b34-88ed-80db293ff736"
# grader-version = "1"
# ///
"""DartBrains assignment: Introduction to Neuroimaging Data."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from nltools.data import BrainData

    from dartbrains_tools.data import localizer

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return BrainData, g, localizer, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: Introduction to Neuroimaging Data

        These are the **Exercises** at the end of the
        [Introduction to Neuroimaging Data chapter](https://dartbrains.org/content/Introduction_to_Neuroimaging_Data.html).

        All five questions are about **looking at your data before you analyse it**. Where is the
        signal good enough to measure anything? Which volumes are you going to regret keeping?
        These are the questions that decide whether a later result means anything, and they are
        the easiest ones to skip.

        **This downloads about 57 MB** (one participant's preprocessed run), so run it in
        [molab](https://molab.marimo.io) rather than in the browser.

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
    _marks = {"nd-q01": 5, "nd-q02": 5, "nd-q03": 5, "nd-q04": 5, "nd-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup

        Run this cell as it is.
        """
    )
    return


@app.cell
def _(localizer):
    SUBJECT = "S01"
    print(f"{len(localizer.get_subjects())} preprocessed subjects: {localizer.get_subjects()}")
    print(f"TR = {localizer.TR}s")
    return (SUBJECT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. Load a run

        Load `SUBJECT`'s preprocessed functional run into a `BrainData` called `bold`, then put
        its dimensions into `run_shape` as a `(n_volumes, n_voxels)` tuple.

        A `BrainData` built from a 4D image is a **matrix**: one row per volume, one column per
        voxel. Almost everything below is either a column-wise summary (a per-voxel map) or a
        row-wise one (a per-volume number), so it is worth being clear which axis is which
        before you start.

        *`localizer.get_file(subject, "derivatives", "bold")` gives you the path.*
        """
    )
    return


@app.cell
def _(BrainData, SUBJECT, localizer):
    bold = ...
    run_shape = ...
    # YOUR CODE HERE
    pass
    run_shape
    return bold, run_shape


@app.cell
def _(g, mo, run_shape):
    mo.stop(
        run_shape is ...,
        mo.md("**Complete the code cell above first.** This check runs once `run_shape` is defined."),
    )

    _s = tuple(run_shape) if run_shape is not ... else ()
    g.check(
        "nd-q01: Load a run",
        [
            (len(_s) == 2 and _s[0] == 128, "the run should have 128 volumes", 2),
            (len(_s) == 2 and _s[1] == 238955, "each volume should have 238955 voxels", 2),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("nd-q01")
    return


@app.cell
def _(g):
    g.feedback("nd-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. Where can you actually measure anything?

        Temporal signal-to-noise ratio asks, for each voxel, how large its signal is relative to
        how much it wobbles over time:

        $$\text{tSNR}_i = \frac{\text{mean}(\text{voxel}_i)}{\text{sd}(\text{voxel}_i)}$$

        Both are taken **over time**, so this is a per-voxel map: one number per column of
        `bold`. A voxel with low tSNR is one where a real effect would be buried in its own
        noise — no amount of later modelling recovers it.

        Compute the map as a numpy array called `tsnr`, then summarise it in `tsnr_stats`:

        ```python
        tsnr_stats = {"mean": ..., "median": ..., "max": ...}
        ```

        *`np.asarray(bold.data)` gives you the `(128, 238955)` matrix. Mind the axis: you want
        one number per voxel, not per volume.*
        """
    )
    return


@app.cell
def _(bold, np):
    tsnr = ...
    tsnr_stats = ...
    # YOUR CODE HERE
    pass
    tsnr_stats
    return tsnr, tsnr_stats


@app.cell
def _(g, mo, np, tsnr, tsnr_stats):
    mo.stop(
        tsnr is ... or tsnr_stats is ...,
        mo.md("**Complete the code cell above first.** This check runs once `tsnr` is defined."),
    )

    g.check(
        "nd-q02: Where can you actually measure anything?",
        [
            (
                np.asarray(tsnr).shape == (238955,),
                "tsnr should be one number per voxel (238955), not per volume",
                2,
            ),
            (
                abs(float(tsnr_stats.get("mean", 0)) - 61.932) < 0.5,
                "mean tSNR across the brain should be about 61.9",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("nd-q02")
    return


@app.cell
def _(g):
    g.feedback("nd-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. Volumes that do not belong

        Now the other axis. Average each **volume** across all its voxels to get one global
        intensity per time point, z-score that series, and call any volume with $|z| > 2$ an
        outlier.

        Put the outlier volume indices in a sorted list called `global_outliers`.

        *One number per volume this time, so the other axis from Q2.*
        """
    )
    return


@app.cell
def _(bold, np):
    global_outliers = ...
    # YOUR CODE HERE
    pass
    global_outliers
    return (global_outliers,)


@app.cell
def _(g, global_outliers, mo):
    mo.stop(
        global_outliers is ...,
        mo.md("**Complete the code cell above first.** This check runs once `global_outliers` is defined."),
    )

    _o = list(global_outliers) if global_outliers is not ... else []
    g.check(
        "nd-q03: Volumes that do not belong",
        [
            (len(_o) == 4, "four volumes should exceed |z| > 2", 2),
            (
                _o == [31, 99, 100, 101],
                "the outliers should be volumes 31, 99, 100 and 101",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("nd-q03")
    return


@app.cell
def _(g):
    g.feedback("nd-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. The same question asked of head motion

        fMRIPrep already estimated how far the head moved between each pair of volumes, as
        **framewise displacement**. Apply the same rule to it — z-score, flag $|z| > 2$ — and put
        the result in `motion_outliers`.

        **One catch, and it is the whole question.** Framewise displacement is a difference
        between consecutive volumes, so it is undefined for the first one and fMRIPrep writes
        `NaN` there. If you replace that `NaN` with `0`, you have claimed the head was
        unusually still at volume 0, and since the mean FD is about 0.084 a zero sits more than
        two standard deviations below it — so volume 0 is reported as an outlier. It is not one.
        It is an artifact of how you filled in a value that was never measured.

        Exclude the undefined volume from the statistics instead.

        *`localizer.load_confounds(subject)` returns a DataFrame with a
        `framewise_displacement` column. A boolean mask from `np.isnan` is the tidiest way to
        leave the first volume out of the mean and standard deviation.*
        """
    )
    return


@app.cell
def _(SUBJECT, localizer, np):
    motion_outliers = ...
    # YOUR CODE HERE
    pass
    motion_outliers
    return (motion_outliers,)


@app.cell
def _(g, mo, motion_outliers):
    mo.stop(
        motion_outliers is ...,
        mo.md("**Complete the code cell above first.** This check runs once `motion_outliers` is defined."),
    )

    _o = list(motion_outliers) if motion_outliers is not ... else []
    g.check(
        "nd-q04: The same question asked of head motion",
        [
            (
                0 not in _o,
                "volume 0 is not an outlier -- its framewise displacement was never defined",
                2,
            ),
            (_o == [27, 95, 104, 105], "the outliers should be volumes 27, 95, 104 and 105", 2),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("nd-q04")
    return


@app.cell
def _(g):
    g.feedback("nd-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. Do the two measures agree?

        You now have two lists of suspect volumes, found with the same rule applied to two
        different signals. The obvious assumption is that they are two views of the same
        problem — a head movement disturbs the image, so the volumes with high motion should be
        the volumes with odd global intensity.

        Check whether that is true here. Build:

        ```python
        agreement = {"overlap": ..., "union_size": ..., "correlation": ...}
        ```

        - `overlap` — sorted list of volumes flagged by **both** measures.
        - `union_size` — how many distinct volumes are flagged by **either**.
        - `correlation` — Pearson correlation between $|z|$ of the global signal and framewise
          displacement, over the volumes where FD is defined.

        *Reuse the two series you already built. Exclude the undefined first volume from the
        correlation as well.*
        """
    )
    return


@app.cell
def _(SUBJECT, bold, global_outliers, localizer, motion_outliers, np):
    agreement = ...
    # YOUR CODE HERE
    pass
    agreement
    return (agreement,)


@app.cell
def _(agreement, g, mo):
    mo.stop(
        agreement is ...,
        mo.md("**Complete the code cell above first.** This check runs once `agreement` is defined."),
    )

    g.check(
        "nd-q05: Do the two measures agree?",
        [
            (
                isinstance(agreement, dict)
                and {"overlap", "union_size", "correlation"} <= set(agreement),
                "agreement needs the keys overlap, union_size and correlation",
                1,
            ),
            (
                list(agreement.get("overlap", [None])) == [],
                "not one volume is flagged by both measures",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("nd-q05")
    return


@app.cell
def _(g):
    g.feedback("nd-q05")
    return


if __name__ == "__main__":
    app.run()
