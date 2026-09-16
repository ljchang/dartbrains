# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "scipy", "matplotlib", "nltools @ git+https://github.com/cosanlab/nltools@master", "dartbrains-tools>=0.1.8", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "b57e9ceb,61975c6d,8af6cf8a,278d1cda,81493290,918c995a,e6157a99,2af47edb,55d5abd3,926d4c46,955d0e95,90dedcbf,92ce3057,13a0c6a6,e3483f08,d799d2b5,26570eba,63dcd5ea,15a1fb3a,0be90e07,345f031d,e328fa6a,94125c79,a21a4760,bef05f7b,28aff7f3,7a6e9748"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "group-analysis"
# grader-assignment-id = "8b1234c4-bb84-488d-a983-c23eb23d4859"
# grader-assignment-version = "47565eb1-5f83-4b71-883f-c772c507161e"
# grader-version = "1"
# ///
"""DartBrains assignment: Group Analysis."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind

    from nltools.data import BrainData
    from nltools.algorithms import fdr, threshold

    from dartbrains_tools.data import localizer

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return BrainData, fdr, g, localizer, mo, np, pd, plt, threshold, ttest_ind


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: Group Analysis

        These are the **Exercises** at the end of the
        [Group Analysis chapter](https://dartbrains.org/content/Group_Analysis.html).

        The chapter ran a first-level GLM for every participant and wrote out one beta image
        per condition. Those betas are already computed and published, so this assignment
        starts where the chapter's exercises start: taking subject-level contrasts to the
        group level.

        **This assignment downloads data.** Each beta image is about 2 MB and you will need
        four conditions across 20 participants, so roughly 150 MB in total. `localizer` caches
        every file after the first download, so re-running a cell is fast — but the first run
        of each question takes a minute or so. Run this in
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
    _marks = {"ga-q01": 5, "ga-q02": 5, "ga-q03": 5, "ga-q04": 5, "ga-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup

        Run this cell as it is. `SUBJECTS` is the 20 participants with published betas, and
        `load_condition` fetches one condition's beta image for all of them and stacks it into
        a single `BrainData` — the same `BrainData([...])` pattern the chapter uses.

        The ten condition names are in `localizer.CONDITIONS`. Each is a crossing of modality
        (`audio_` / `video_`) with task (`sentence`, `computation`, `left_hand`, `right_hand`),
        plus the two checkerboards.
        """
    )
    return


@app.cell
def _(BrainData, localizer):
    SUBJECTS = localizer.get_subjects()

    def load_condition(condition):
        """Stack one condition's beta image across all 20 participants.

        Returns a BrainData of shape (20, 238955) -- one row per participant.
        """
        return BrainData([localizer.get_file(s, "betas", condition) for s in SUBJECTS])

    print(f"{len(SUBJECTS)} participants: {SUBJECTS}")
    print(f"conditions: {localizer.CONDITIONS}")
    return SUBJECTS, load_condition


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. A subject-level contrast

        The chapter's first exercise asks which regions are more involved in **visual** than
        **auditory** processing. The cleanest way to ask that is to hold the task constant and
        vary only the modality: compare `video_sentence` against `audio_sentence`. Both
        conditions are a sentence; only the channel it arrives through differs.

        Build `con`, the subject-level contrast — one contrast image per participant, computed
        as `video_sentence` minus `audio_sentence`.

        *`load_condition` gives you a `BrainData` per condition, and `BrainData` supports
        arithmetic directly, so this is a subtraction rather than a loop. The result should
        still have 20 rows.*
        """
    )
    return


@app.cell
def _(load_condition):
    con = ...
    # YOUR CODE HERE
    pass
    con
    return (con,)


@app.cell
def _(con, g, mo):
    mo.stop(
        con is ...,
        mo.md("**Complete the code cell above first.** This check runs once `con` is defined."),
    )

    _shape = getattr(con, "shape", None)
    g.check(
        "ga-q01: A subject-level contrast",
        [
            (
                _shape is not None and _shape[0] == 20,
                "con should have one row per participant (20)",
                2,
            ),
            (
                _shape is not None and _shape[1] == 238955,
                "each row should be a whole-brain image of 238955 voxels",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("ga-q01")
    return


@app.cell
def _(g):
    g.feedback("ga-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. Take it to the group

        A contrast per participant is not yet a result. The group question is whether that
        difference is reliably non-zero *across* participants, which is a one-sample t-test at
        every voxel against a population mean of zero.

        Run it on `con` and put the returned dictionary in `stats`.

        *`BrainData.ttest()` does the whole thing in one call and returns a dict with keys
        `mean`, `t`, `z` and `p`, each a `BrainData`. Note that unlike older versions of
        nltools it takes no `threshold_dict` argument — thresholding is Q3's job.*
        """
    )
    return


@app.cell
def _(con):
    stats = ...
    # YOUR CODE HERE
    pass
    stats
    return (stats,)


@app.cell
def _(g, mo, stats):
    mo.stop(
        stats is ...,
        mo.md("**Complete the code cell above first.** This check runs once `stats` is defined."),
    )

    _t = stats["t"].data if isinstance(stats, dict) and "t" in stats else None
    g.check(
        "ga-q02: Take it to the group",
        [
            (
                isinstance(stats, dict) and {"mean", "t", "p"} <= set(stats),
                "stats should be the dict returned by ttest(), with mean, t and p",
                1,
            ),
            (
                _t is not None and abs(float(_t.max()) - 8.073) < 0.05,
                "the largest t statistic should be about 8.07",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("ga-q02")
    return


@app.cell
def _(g):
    g.feedback("ga-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. What correction costs you

        There are 238,955 voxels in this map, so an uncorrected threshold is not a threshold
        at all — at p < 0.05 you would expect about 12,000 false positives before any real
        signal. Measure what correction actually does to the map.

        Build a dictionary called `survivors` counting how many voxels pass each of four
        thresholds:

        ```python
        survivors = {"p05": ..., "p005": ..., "p001": ..., "fdr": ...}
        ```

        The first three are uncorrected p-value thresholds (0.05, 0.005, 0.001). The fourth is
        the FDR-corrected count at q < 0.05.

        *For the uncorrected counts you can work straight off `stats["p"].data`. For FDR, pass
        that same array to `fdr(p, q=0.05)`, which returns the p-value cutoff that controls the
        false discovery rate; count the voxels at or below it. Watch for the edge case in the
        next question.*
        """
    )
    return


@app.cell
def _(fdr, stats):
    survivors = ...
    # YOUR CODE HERE
    pass
    survivors
    return (survivors,)


@app.cell
def _(g, mo, survivors):
    mo.stop(
        survivors is ...,
        mo.md("**Complete the code cell above first.** This check runs once `survivors` is defined."),
    )

    _keys = {"p05", "p005", "p001", "fdr"}
    g.check(
        "ga-q03: What correction costs you",
        [
            (
                isinstance(survivors, dict) and set(survivors) == _keys,
                f"survivors should be a dict with exactly the keys {sorted(_keys)}",
                1,
            ),
            (
                abs(int(survivors.get("p001", 0)) - 3786) <= 5,
                "about 3786 voxels pass the uncorrected p < 0.001 threshold",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("ga-q03")
    return


@app.cell
def _(g):
    g.feedback("ga-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. Numbers against words

        The chapter's second exercise: which regions are more involved in processing **numbers**
        than **words**? This time modality is the nuisance rather than the effect, so collapse
        across it — compare the two computation conditions against the two sentence conditions.

        Build the contrast, run the group t-test, and put the FDR-corrected voxel count in
        `math_result`:

        ```python
        math_result = {"max_t": ..., "min_t": ..., "fdr_voxels": ...}
        ```

        *Adding `video_computation` to `audio_computation` and subtracting both sentence
        conditions gives each side equal weight, so there is no need to divide by two — the
        scaling cancels in the contrast. This downloads two more conditions, so expect a wait
        the first time.*
        """
    )
    return


@app.cell
def _(fdr, load_condition):
    math_result = ...
    # YOUR CODE HERE
    pass
    math_result
    return (math_result,)


@app.cell
def _(g, math_result, mo):
    mo.stop(
        math_result is ...,
        mo.md("**Complete the code cell above first.** This check runs once `math_result` is defined."),
    )

    g.check(
        "ga-q04: Numbers against words",
        [
            (
                isinstance(math_result, dict)
                and {"max_t", "min_t", "fdr_voxels"} <= set(math_result),
                "math_result needs the keys max_t, min_t and fdr_voxels",
                1,
            ),
            (
                abs(float(math_result.get("max_t", 0)) - 5.311) < 0.05,
                "the largest t should be about 5.31",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("ga-q04")
    return


@app.cell
def _(g):
    g.feedback("ga-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. Is the effect different for men and women?

        The chapter's third exercise asks for a two-sample comparison, which needs the
        participants' sex from `participants.tsv`. `SEX` below is already aligned to the order
        of `SUBJECTS` — 12 F and 8 M.

        Take the Q1 contrast `con`, split its rows by sex, and run an independent-samples
        t-test at every voxel. Report:

        ```python
        sex_result = {"n_female": ..., "n_male": ..., "max_t": ..., "fdr_voxels": ...}
        ```

        *`con.data` is an ordinary `(20, 238955)` numpy array, so a boolean mask splits it.
        `scipy.stats.ttest_ind(female, male, axis=0)` returns t and p per voxel.*

        **Read the `fdr` docstring before you write the last value.** It returns a p-value
        cutoff, except when *no* p-value survives correction — then it returns `-1` as a
        sentinel rather than a cutoff. Handle that case explicitly instead of letting the
        comparison decide for you: `p < -1` happens to count zero voxels here, but it is
        accidentally right rather than right, and the same code with the comparison the other
        way round would report all 238,955 voxels as significant.

        Compare what you get to Q3. That contrast is the same one, over the same people.
        """
    )
    return


@app.cell
def _(SUBJECTS, localizer, pd):
    _meta = pd.read_csv(localizer.download("dartbrains/localizer", "participants.tsv"), sep="\t")
    SEX = _meta[_meta.participant_id.isin(SUBJECTS)].set_index("participant_id").loc[SUBJECTS]["sex"].values
    print(f"sex, aligned to SUBJECTS: {list(SEX)}")
    return (SEX,)


@app.cell
def _(SEX, con, fdr, ttest_ind):
    sex_result = ...
    # YOUR CODE HERE
    pass
    sex_result
    return (sex_result,)


@app.cell
def _(g, mo, sex_result):
    mo.stop(
        sex_result is ...,
        mo.md("**Complete the code cell above first.** This check runs once `sex_result` is defined."),
    )

    g.check(
        "ga-q05: Is the effect different for men and women?",
        [
            (
                int(sex_result.get("n_female", 0)) == 12 and int(sex_result.get("n_male", 0)) == 8,
                "there are 12 female and 8 male participants",
                1,
            ),
            (
                abs(float(sex_result.get("max_t", 0)) - 4.784) < 0.05,
                "the largest two-sample t should be about 4.78",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("ga-q05")
    return


@app.cell
def _(g):
    g.feedback("ga-q05")
    return


if __name__ == "__main__":
    app.run()
