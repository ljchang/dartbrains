# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "scikit-learn", "matplotlib", "nltools @ git+https://github.com/cosanlab/nltools@master", "dartbrains-tools>=0.1.8", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "fbd7eb39,dcf6c18c,8af6cf8a,e9a977e4,4c938411,006ab303,f75f3ff2,e228f382,fc85058e,9f8d422a,3788f229,e82865e6,468db12f,26f06c08,cb558ea3,d22b42f4,9f7493a6,55b66a31,acf2d314,2307046f,3ffb4672,38db303d,a5ce2e7c,2eb22f45,d83030d5,2abf4963"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "multivariate-prediction"
# grader-assignment-id = "2aa51f07-3248-4dca-abbd-5fcad417b25e"
# grader-assignment-version = "6427c4f6-7cdd-4fcf-b4cf-5f7429b6aa24"
# grader-version = "1"
# ///
"""DartBrains assignment: Multivariate Prediction."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.model_selection import LeaveOneGroupOut

    from nltools.data import BrainData

    from dartbrains_tools.data import localizer

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return (
        BrainData,
        LeaveOneGroupOut,
        LogisticRegression,
        RidgeClassifier,
        SVC,
        g,
        localizer,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: Multivariate Prediction

        These are the **Exercises** at the end of the
        [Multivariate Prediction chapter](https://dartbrains.org/content/Multivariate_Prediction.html).

        The chapter trained a classifier to tell left-hand from right-hand responses. Here you
        do the same for **horizontal versus vertical checkerboards** — two stimuli that differ
        only in orientation, so any pattern that separates them has to be picking up something
        genuinely visual.

        Most of the assignment is about not fooling yourself. A number like "75% accurate" means
        nothing on its own: you need to know what chance looks like for *this* design, whether
        the pattern is specific to what you trained it on, and how much of the result would
        survive a different modelling choice.

        **This downloads about 380 MB in total** (ten conditions across 20 participants — the
        last question needs all of them) and takes a minute or so to run. Use
        [molab](https://molab.marimo.io).

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
    _marks = {"mp-q01": 5, "mp-q02": 5, "mp-q03": 5, "mp-q04": 5, "mp-q05": 5}
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
def _(BrainData, localizer):
    SUBJECTS = localizer.get_subjects()

    def load_condition(condition):
        """One condition's beta image for all 20 participants, stacked."""
        return BrainData([localizer.get_file(s, "betas", condition) for s in SUBJECTS])

    print(f"{len(SUBJECTS)} participants")
    return SUBJECTS, load_condition


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. Build the training set

        Stack the two conditions into one `BrainData` and label them.

        - `data` — the 20 `horizontal_checkerboard` images followed by the 20
          `vertical_checkerboard` ones. `BrainData.append` joins two stacks.
        - `Y` — 1 for every horizontal image, 0 for every vertical one.
        - `groups` — which participant each row came from.

        `groups` is the one that is easy to skip and expensive to get wrong. Each participant
        contributes **two** rows, one per condition. If cross-validation splits those two rows
        across train and test, the classifier gets to see that participant's brain before being
        asked about it, and the accuracy you report is partly a measure of how well it memorised
        them. The group labels are what stops that.

        *The order of the three must agree: row *i* of `data` is `Y[i]` from `groups[i]`.*
        """
    )
    return


@app.cell
def _(SUBJECTS, load_condition, np):
    data = ...
    Y = ...
    groups = ...
    # YOUR CODE HERE
    pass
    return Y, data, groups


@app.cell
def _(Y, data, g, groups, mo, np):
    mo.stop(
        data is ... or Y is ... or groups is ...,
        mo.md("**Complete the code cell above first.** This check runs once all three are defined."),
    )

    g.check(
        "mp-q01: Build the training set",
        [
            (
                getattr(data, "shape", None) == (40, 238955),
                "data should be 40 images (20 participants x 2 conditions)",
                2,
            ),
            (
                len(set(groups)) == 20 and len(groups) == 40,
                "groups should name 20 participants across 40 rows",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("mp-q01")
    return


@app.cell
def _(g):
    g.feedback("mp-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. Classify, holding out whole participants

        The chapter's first exercise. Train a linear SVM and cross-validate **leave-one-subject-out**:
        twenty folds, each holding out both images from one participant and testing on them.

        Put the result object in `svm_result`.

        ```python
        data.predict(y=Y, estimator=SVC(kernel="linear"),
                     cv=LeaveOneGroupOut(), groups=groups)
        ```

        Leave-one-subject-out is the strict version of what the chapter did with
        `GroupKFold(n_splits=5)`. Each fold trains on 19 participants and is asked about someone
        it has never seen, which is the question you actually care about: does this pattern
        generalise to a *new person*?

        *`.mean_score` and `.std_score` hold the accuracy; `.weight_map` is the trained pattern,
        which Q4 needs.*
        """
    )
    return


@app.cell
def _(LeaveOneGroupOut, SVC, Y, data, groups):
    svm_result = ...
    # YOUR CODE HERE
    pass
    return (svm_result,)


@app.cell
def _(g, mo, svm_result):
    mo.stop(
        svm_result is ...,
        mo.md("**Complete the code cell above first.** This check runs once `svm_result` is defined."),
    )

    g.check(
        "mp-q02: Classify, holding out whole participants",
        [
            (
                hasattr(svm_result, "mean_score") and hasattr(svm_result, "weight_map"),
                "svm_result should be what .predict() returned, not just the score",
                1,
            ),
            (
                abs(float(svm_result.mean_score) - 0.75) < 0.02,
                "leave-one-subject-out accuracy should be 0.75",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("mp-q02")
    return


@app.cell
def _(g):
    g.feedback("mp-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. Is 0.75 actually above chance?

        Chance is 50% *in principle*, but with 20 folds of two images each, "what this pipeline
        scores when there is nothing to find" is an empirical question. Answer it by breaking the
        only thing that carries the signal — the correspondence between images and labels — and
        re-running everything else unchanged.

        Permute `Y` **50 times** with `np.random.default_rng(0)`, re-run the identical
        leave-one-subject-out classification each time, and collect:

        ```python
        permutation = {"observed": ..., "null_mean": ..., "null_max": ..., "p": ...}
        ```

        with `p = (number of null scores >= observed + 1) / (number of permutations + 1)`.

        *Use `rng.permutation(Y)` inside the loop so all 50 draws come from the one seeded
        generator, in that order. This takes about half a minute.*
        """
    )
    return


@app.cell
def _(LeaveOneGroupOut, SVC, Y, data, groups, np, svm_result):
    permutation = ...
    # YOUR CODE HERE
    pass
    permutation
    return (permutation,)


@app.cell
def _(g, mo, permutation):
    mo.stop(
        permutation is ...,
        mo.md("**Complete the code cell above first.** This check runs once `permutation` is defined."),
    )

    g.check(
        "mp-q03: Is 0.75 actually above chance?",
        [
            (
                isinstance(permutation, dict)
                and {"observed", "null_mean", "null_max", "p"} <= set(permutation),
                "permutation needs the keys observed, null_mean, null_max and p",
                1,
            ),
            (
                abs(float(permutation.get("null_mean", 0)) - 0.4825) < 0.03,
                "the null should sit at chance -- about 0.48",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("mp-q03")
    return


@app.cell
def _(g):
    g.feedback("mp-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. What else does the pattern respond to?

        The chapter's second exercise. You have a pattern that separates horizontal from
        vertical checkerboards — but a pattern is just a weighted map, and you can apply it to
        any image. Does it pick out anything else?

        Take `svm_result.weight_map` and, for each of the ten conditions, compute the mean
        spatial correlation between that pattern and the condition's images across all
        participants. Put them in a dict called `generalization`, keyed by condition name.

        ```python
        load_condition(c).similarity(weight_map, metric="correlation")   # one value per subject
        ```

        Think about what you expect before you look. The chapter asks whether the pattern gets
        confused by other *visual* conditions — there are four `video_` conditions and they all
        arrive through the eyes.

        *`localizer.CONDITIONS` is the list of ten. This loads every condition, so it is the
        slow cell on a cold cache.*
        """
    )
    return


@app.cell
def _(load_condition, localizer, np, svm_result):
    generalization = ...
    # YOUR CODE HERE
    pass
    generalization
    return (generalization,)


@app.cell
def _(g, generalization, mo):
    mo.stop(
        generalization is ...,
        mo.md("**Complete the code cell above first.** This check runs once `generalization` is defined."),
    )

    _ranked = sorted(generalization.items(), key=lambda kv: -kv[1]) if isinstance(generalization, dict) else []
    g.check(
        "mp-q04: What else does the pattern respond to?",
        [
            (
                isinstance(generalization, dict) and len(generalization) == 10,
                "generalization should have one entry per condition (10)",
                1,
            ),
            (
                len(_ranked) == 10 and _ranked[0][0] == "horizontal_checkerboard",
                "the pattern should resemble horizontal checkerboards most -- it was trained to",
                1,
            ),
            (
                len(_ranked) == 10 and _ranked[-1][0] == "vertical_checkerboard",
                "and vertical checkerboards least, for the same reason",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("mp-q04")
    return


@app.cell
def _(g):
    g.feedback("mp-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. Would a different classifier have changed the story?

        The chapter covers regularization, and it is tempting to treat the choice of estimator as
        something to tune until the number improves. Measure how much it is actually worth here.

        Re-run the same leave-one-subject-out classification with three linear estimators and
        collect their accuracies in `estimator_scores`, keyed exactly as below:

        ```python
        estimator_scores = {"svc": ..., "ridge": ..., "logistic": ...}
        # SVC(kernel="linear"), RidgeClassifier(), LogisticRegression(max_iter=2000)
        ```

        Then compare the spread you get to the standard deviation of the null distribution from
        Q3. That comparison is the question: if swapping estimators moves the score by less than
        the noise floor, then picking the best of the three is picking noise — and reporting it
        as the result is how a pipeline quietly overfits to its own test set.
        """
    )
    return


@app.cell
def _(
    LeaveOneGroupOut,
    LogisticRegression,
    RidgeClassifier,
    SVC,
    Y,
    data,
    groups,
):
    estimator_scores = ...
    # YOUR CODE HERE
    pass
    estimator_scores
    return (estimator_scores,)


@app.cell
def _(estimator_scores, g, mo):
    mo.stop(
        estimator_scores is ...,
        mo.md("**Complete the code cell above first.** This check runs once `estimator_scores` is defined."),
    )

    _v = list(estimator_scores.values()) if isinstance(estimator_scores, dict) else []
    _spread = (max(_v) - min(_v)) if _v else None
    g.check(
        "mp-q05: Would a different classifier have changed the story?",
        [
            (
                isinstance(estimator_scores, dict)
                and set(estimator_scores) == {"svc", "ridge", "logistic"},
                "estimator_scores should be keyed svc, ridge and logistic",
                1,
            ),
            (
                abs(float(estimator_scores.get("svc", 0)) - 0.75) < 0.02,
                "the SVM should reproduce the 0.75 from Q2",
                1,
            ),
            (
                _spread is not None and _spread <= 0.06,
                "all three should land within about 0.05 of each other",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("mp-q05")
    return


@app.cell
def _(g):
    g.feedback("mp-q05")
    return


if __name__ == "__main__":
    app.run()
