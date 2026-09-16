# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "matplotlib", "nltools @ git+https://github.com/cosanlab/nltools@master", "dartbrains-tools>=0.1.8", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "f2ff19d0,d6419fc8,8af6cf8a,f2624eac,19459ab0,b8115490,50e67a05,eba14cec,ab1b0909,219b64d3,80677d49,e5c5195f,565634bf,f540846e,b41818b6,e2b184bb,d38f7e15,4572a0b3,ec8f4378,9ed3156e,82c10082,ac2d2907,0bdb5fa5,d5a7434e,dcfb61f1,dc20953d"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "glm-single-subject"
# grader-assignment-id = "070a64d2-6421-4689-a359-947cc0f05373"
# grader-assignment-version = "913a6eda-2958-4cbc-8114-36efc742c513"
# grader-version = "1"
# ///
"""DartBrains assignment: GLM Single Subject Model."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from nltools.data import BrainData

    from dartbrains_tools.data import localizer

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return BrainData, g, localizer, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: GLM Single Subject Model

        These are the **Exercises** at the end of the
        [GLM Single Subject Model chapter](https://dartbrains.org/content/GLM_Single_Subject_Model.html).

        The chapter fit a first-level GLM for one participant and pulled out a beta image per
        condition. Those betas are published, so this assignment starts where the exercises
        start: **asking questions of them by building contrasts**.

        A contrast is just a weighted sum of condition images. The whole skill is choosing the
        weights so that the number you get answers the question you asked — and nothing else.

        **This downloads about 19 MB** (ten conditions for one participant), so run it in
        [molab](https://molab.marimo.io) rather than in the browser. It is quick.

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
    _marks = {"gss-q01": 5, "gss-q02": 5, "gss-q03": 5, "gss-q04": 5, "gss-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup

        Run this cell as it is. `CONDITIONS` is the ten conditions in the order this assignment
        uses throughout — every contrast below is a vector of ten weights lined up with it.

        The design crosses **modality** (`audio_` / `video_`) with **task** (`computation`,
        `sentence`, `left_hand`, `right_hand`), plus two checkerboards that are visual only.
        That crossing is what makes the exercises answerable: you can hold one factor fixed and
        vary the other.
        """
    )
    return


@app.cell
def _(localizer):
    SUBJECT = "S01"
    CONDITIONS = localizer.CONDITIONS
    for _i, _c in enumerate(CONDITIONS):
        print(f"  {_i}  {_c}")
    return CONDITIONS, SUBJECT


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. Load the betas

        Load `SUBJECT`'s beta image for each of the ten conditions into a single `BrainData`
        called `betas`, **in `CONDITIONS` order**. Row *i* must be `CONDITIONS[i]`, because
        every contrast from here on is a vector of weights indexed the same way.

        *`localizer.get_file(subject, "betas", condition)` returns a path, and `BrainData`
        takes a list of paths and stacks them.*
        """
    )
    return


@app.cell
def _(BrainData, CONDITIONS, SUBJECT, localizer):
    betas = ...
    # YOUR CODE HERE
    pass
    betas
    return (betas,)


@app.cell
def _(betas, g, mo):
    mo.stop(
        betas is ...,
        mo.md("**Complete the code cell above first.** This check runs once `betas` is defined."),
    )

    _shape = getattr(betas, "shape", None)
    g.check(
        "gss-q01: Load the betas",
        [
            (
                _shape is not None and _shape[0] == 10,
                "betas should have one row per condition (10)",
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
    g.submit_button("gss-q01")
    return


@app.cell
def _(g):
    g.feedback("gss-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. Weights that mean something

        Every exercise below is "these conditions versus those conditions", so write the weights
        once.

        `contrast_weights(positive, negative)` takes two lists of condition names and returns a
        length-10 numpy array: each condition in `positive` gets `1 / len(positive)`, each in
        `negative` gets `-1 / len(negative)`, everything else 0.

        Dividing by the group size is the part that matters. Weighting six visual conditions
        `+1` and four auditory `-1` does not compare visual against auditory — it compares *six
        things* against *four things*, and the difference in how many you added dominates the
        result. Averaging each side first makes the contrast a difference of means, and makes
        the weights sum to zero, so a voxel that responds equally to everything scores 0.

        ```python
        contrast_weights(["video_sentence"], ["audio_sentence"])
        # -> array([0., 0., 0., -1., 0., 0., 0., 0., 0., 1.])
        ```

        *`CONDITIONS.index(name)` gives you the position for a name.*
        """
    )
    return


@app.cell
def _(CONDITIONS, np):
    def contrast_weights(positive, negative):
        # YOUR CODE HERE
        pass

    return (contrast_weights,)


@app.cell
def _(contrast_weights, g, mo, np):
    _probe = contrast_weights(["video_sentence"], ["audio_sentence"])
    mo.stop(
        _probe is None,
        mo.md("**Complete the function above first.** It should `return` a length-10 array."),
    )

    _w = contrast_weights(
        ["horizontal_checkerboard", "vertical_checkerboard", "video_computation",
         "video_left_hand", "video_right_hand", "video_sentence"],
        ["audio_computation", "audio_left_hand", "audio_right_hand", "audio_sentence"],
    )
    g.check(
        "gss-q02: Weights that mean something",
        [
            (
                len(np.asarray(_probe)) == 10,
                "the weights should be one number per condition (10)",
                1,
            ),
            (
                abs(float(np.asarray(_w).sum())) < 1e-9,
                "the weights of a contrast should sum to zero",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("gss-q02")
    return


@app.cell
def _(g):
    g.feedback("gss-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. Visual against auditory

        The chapter's first exercise: which regions are more involved in **visual** than
        **auditory** sensory processing?

        Six conditions arrive through the eyes (both checkerboards and all four `video_`
        conditions) and four through the ears (the `audio_` conditions). Build
        `visual_vs_audio` as the contrast image — multiply `betas` by the weights.

        `BrainData * weights` already returns the weighted **sum** across rows, so the result is
        one image rather than ten. There is no `.sum()` to call afterwards; doing so collapses
        the map to a single number.
        """
    )
    return


@app.cell
def _(betas, contrast_weights):
    visual_vs_audio = ...
    # YOUR CODE HERE
    pass
    visual_vs_audio
    return (visual_vs_audio,)


@app.cell
def _(g, mo, np, visual_vs_audio):
    mo.stop(
        visual_vs_audio is ...,
        mo.md("**Complete the code cell above first.** This check runs once `visual_vs_audio` is defined."),
    )

    _d = np.asarray(visual_vs_audio.data)
    g.check(
        "gss-q03: Visual against auditory",
        [
            (
                _d.shape == (238955,),
                "the contrast should be one whole-brain image, not ten -- do not call .sum()",
                2,
            ),
            (
                abs(float(_d.max()) - 85.677) < 0.5,
                "the strongest visual-over-auditory voxel should be about 85.7",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("gss-q03")
    return


@app.cell
def _(g):
    g.feedback("gss-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. Two more questions of the same data

        The chapter's second and third exercises, both of which lean on the crossed design.

        - `numbers_vs_words` — numbers against words. `computation` versus `sentence`, collapsed
          across modality so the answer is about *content* rather than whether it was heard or
          read.
        - `motor_vs_cognitive` — moving against thinking. The two `*_hand` conditions against
          the `computation` and `sentence` ones, again across both modalities.

        Collapsing across modality is the point: include both the `audio_` and `video_` version
        of each task on each side, and whatever is modality-specific cancels.
        """
    )
    return


@app.cell
def _(betas, contrast_weights):
    numbers_vs_words = ...
    motor_vs_cognitive = ...
    # YOUR CODE HERE
    pass
    numbers_vs_words
    return motor_vs_cognitive, numbers_vs_words


@app.cell
def _(g, mo, motor_vs_cognitive, np, numbers_vs_words):
    mo.stop(
        numbers_vs_words is ... or motor_vs_cognitive is ...,
        mo.md("**Complete the code cell above first.** This check runs once both contrasts are defined."),
    )

    _n = np.asarray(numbers_vs_words.data)
    _m = np.asarray(motor_vs_cognitive.data)
    g.check(
        "gss-q04: Two more questions of the same data",
        [
            (
                _n.shape == (238955,) and _m.shape == (238955,),
                "both contrasts should be single whole-brain images",
                1,
            ),
            (
                abs(float(_n.max()) - 261.636) < 1.0,
                "numbers over words should peak at about 261.6",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("gss-q04")
    return


@app.cell
def _(g):
    g.feedback("gss-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. What smoothing does to your answer

        The chapter's fourth exercise: how are the results affected by different smoothing
        kernels? Smoothing is usually applied before modelling and then not thought about again,
        so it is worth seeing what it costs.

        Smooth `visual_vs_audio` at three kernel widths and measure two things each time: how
        far the peak drops, and how much the map still resembles the unsmoothed one.

        Build a dict called `smoothing` keyed by FWHM, each value a `(peak, correlation)` tuple:

        ```python
        smoothing = {4: (peak, corr), 8: (...), 12: (...)}
        ```

        where `peak` is the maximum of the smoothed map and `corr` is the Pearson correlation
        between the smoothed map's voxels and the unsmoothed `visual_vs_audio` voxels.

        *`BrainData.smooth(fwhm=...)` returns a new image. `np.corrcoef(a, b)[0, 1]` takes two
        1-D arrays — `np.asarray(img.data)` gets you one.*
        """
    )
    return


@app.cell
def _(np, visual_vs_audio):
    smoothing = ...
    # YOUR CODE HERE
    pass
    smoothing
    return (smoothing,)


@app.cell
def _(g, mo, smoothing):
    mo.stop(
        smoothing is ...,
        mo.md("**Complete the code cell above first.** This check runs once `smoothing` is defined."),
    )

    _peaks = [smoothing[k][0] for k in (4, 8, 12)] if isinstance(smoothing, dict) else []
    _corrs = [smoothing[k][1] for k in (4, 8, 12)] if isinstance(smoothing, dict) else []
    g.check(
        "gss-q05: What smoothing does to your answer",
        [
            (
                isinstance(smoothing, dict) and set(smoothing) == {4, 8, 12},
                "smoothing should be keyed by the three FWHM values",
                1,
            ),
            (
                len(_peaks) == 3 and _peaks[0] > _peaks[1] > _peaks[2],
                "a wider kernel should always lower the peak",
                1,
            ),
            (
                len(_corrs) == 3 and _corrs[0] > _corrs[1] > _corrs[2],
                "a wider kernel should leave the map less like the unsmoothed one",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("gss-q05")
    return


@app.cell
def _(g):
    g.feedback("gss-q05")
    return


if __name__ == "__main__":
    app.run()
