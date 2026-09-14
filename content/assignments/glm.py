# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "matplotlib", "nilearn", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "d9bbf386,0def6f6b,8af6cf8a,02394504,2b636745,9dfa4cd8,65ae7bd0,d1247f35,704f1508,de24ba33,15f6ee82,a00823c9,64765855,e3adbd64,b98c399a,5edeb41c,ee748a1e,7215d5d9,8e2b8b0f,d031d589,60edda69,d6900c33,c5a3908d,543f6f7e,bd3d1a89,a0226b3b,cea8aa5d"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "glm"
# grader-assignment-id = "f9acfc9e-3068-4237-90cc-8427d166abab"
# grader-assignment-version = "11f738f7-22cf-406f-9673-41ea28b877e9"
# grader-version = "5"
# ///
"""DartBrains assignment: the General Linear Model (instructor notebook).

Adapted from the Exercises at the end of the GLM chapter:
https://dartbrains.org/content/GLM.html

Publish with (from dartbrains-grader/backend):
    uv run grader publish ../../dartbrains-assignments/assignments/glm.py \
        --server <server> --offering neuroimaging/2026-fall --slug glm --title "GLM"
"""

import marimo

__generated_with = "0.24.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from nilearn.glm.first_level import glover_hrf

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return g, glover_hrf, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: the General Linear Model

        This assignment works through the **Exercises** at the end of the
        [GLM chapter](https://dartbrains.org/content/GLM.html) on dartbrains.org.
        Re-read the chapter's simulation section first: everything here uses the same
        approach (simulate a voxel, build a design matrix, fit it with ordinary least squares)
        to answer questions about experimental design.

        Each question has a cell for your code, a **Check** cell that runs locally and gives
        hints, and a **Submit** button that records an attempt on the server. Sign in once at
        the top. Checks award partial credit; the final interpretation question is graded by hand.
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
    _marks = {"glm-q01": 5, "glm-q02": 5, "glm-q03": 5, "glm-q04": 5, "glm-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup: the simulation from the chapter

        The chapter simulated a single voxel that responds to *face* and *object* trials, convolved
        each condition with a double-gamma HRF, added Gaussian noise, and then recovered the
        amplitudes with a GLM. The helper below packages that recipe so you can vary one
        parameter at a time. It returns the design matrix `X` (intercept, faces, objects) and the
        simulated time series `Y`. Trial onsets are jittered (as in the chapter's last section),
        never overlap, and are drawn from a fixed random seed, so the same arguments always give
        the same data.

        The estimator functions (`ols_estimator`, `r_square`, `contrast_efficiency`) are copied
        from the chapter.
        """
    )
    return


@app.cell
def _(glover_hrf, np):
    tr = 2
    hrf = glover_hrf(tr, oversampling=1)  # double-gamma HRF sampled once per TR (16 samples)

    def simulate_voxel(
        face_intensity=2.0,
        object_intensity=1.0,
        sigma=0.15,
        n_trial=5,
        face_duration=1,
        object_duration=1,
        n_tr=200,
        seed=0,
    ):
        """Simulate one voxel's time series and the design matrix used to model it.

        Face and object trials are placed at random (jittered) onsets that never overlap,
        scaled by their intensities, held for ``*_duration`` TRs, convolved with the HRF,
        summed, and corrupted with N(0, sigma^2) noise. The design matrix ``X`` (intercept, faces,
        objects) always models each trial as a single-TR event.
        """
        rng = np.random.default_rng(seed)
        # Jittered design: draw all 2 * n_trial onsets from a grid of candidate TRs (so no two
        # trials of either condition sit closer than ``gap`` TRs), then assign them at random
        # to the two conditions. The variable inter-trial interval decorrelates the regressors.
        gap = min(8, (n_tr - 25) // (2 * n_trial))
        onsets = rng.choice(np.arange(5, n_tr - 20, gap), size=2 * n_trial, replace=False)
        rng.shuffle(onsets)
        face_onsets = np.sort(onsets[:n_trial])
        object_onsets = np.sort(onsets[n_trial:])

        face_signal, object_signal = np.zeros(n_tr), np.zeros(n_tr)
        face_events, object_events = np.zeros(n_tr), np.zeros(n_tr)
        for onset in face_onsets:
            face_signal[onset : onset + face_duration] = face_intensity
            face_events[onset] = 1
        for onset in object_onsets:
            object_signal[onset : onset + object_duration] = object_intensity
            object_events[onset] = 1

        signal = np.convolve(face_signal, hrf, mode="same") + np.convolve(object_signal, hrf, mode="same")
        Y = signal + sigma * rng.standard_normal(n_tr)
        X = np.vstack(
            [
                np.ones(n_tr),
                np.convolve(face_events, hrf, mode="same"),
                np.convolve(object_events, hrf, mode="same"),
            ]
        ).T
        return X, Y

    def ols_estimator(X, Y):
        return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)

    def r_square(Y, predicted_y):
        SS_total = np.sum((Y - np.mean(Y)) ** 2)
        SS_residual = np.sum((Y - predicted_y) ** 2)
        return 1 - (SS_residual / SS_total)

    def contrast_efficiency(X, contrast):
        c = np.array(contrast)
        return 1 / np.dot(np.dot(c, np.linalg.pinv(np.dot(X.T, X))), c.T)

    return contrast_efficiency, hrf, ols_estimator, r_square, simulate_voxel, tr


@app.cell
def _(np, ols_estimator, plt, simulate_voxel):
    # A quick look at the default simulation and its GLM fit.
    _X, _Y = simulate_voxel()
    _beta = ols_estimator(_X, _Y)
    _fig, _ax = plt.subplots(figsize=(12, 3))
    _ax.plot(_Y, linewidth=2, label="Simulated voxel")
    _ax.plot(np.dot(_X, _beta), linewidth=2, label="Predicted voxel")
    _ax.set_xlabel("TR")
    _ax.set_ylabel("Intensity")
    _ax.set_title(f"Default simulation: beta = {np.round(_beta, 2)}")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. What happens when we vary the signal amplitude?

        Some signals will be very strong and others weaker. How does the model fit change when
        the signal amplitudes are stronger and weaker?

        Simulate the voxel at three levels of signal amplitude by scaling *both* intensities:
        use `signal_amplitudes = [0.5, 1.0, 2.0]`, with face intensity `2 * amplitude` and object
        intensity `1 * amplitude` (leave `sigma` and `seed` at their defaults). For each level, fit
        the GLM with `ols_estimator`, compute $r^2$ with `r_square`, and store the three values in
        a numpy array `r2_by_amplitude`. Then make a plot of $r^2$ against signal amplitude.
        """
    )
    return


@app.cell
def _(np, ols_estimator, plt, r_square, simulate_voxel):
    r2_by_amplitude = ...
    signal_amplitudes = ...
    # YOUR CODE HERE
    pass
    return r2_by_amplitude, signal_amplitudes


@app.cell
def _(g, mo, np, r2_by_amplitude, signal_amplitudes):
    mo.stop(
        any(v is ... for v in (r2_by_amplitude, signal_amplitudes)),
        mo.md("**Complete the code cell above first.** This check runs once `r2_by_amplitude` and `signal_amplitudes` are defined."),
    )

    g.check(
        "glm-q01: r-squared vs signal amplitude",
        [
            (
                isinstance(r2_by_amplitude, np.ndarray) and r2_by_amplitude.shape == (3,),
                "r2_by_amplitude should be a numpy array with one r-squared per amplitude (shape (3,))",
                1,
            ),
            (
                bool(np.all((r2_by_amplitude >= 0) & (r2_by_amplitude <= 1))),
                "r-squared values must lie between 0 and 1",
                1,
            ),
            (
                bool(np.all(np.diff(r2_by_amplitude) > 0)),
                "r-squared should increase as the signal amplitude increases",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("glm-q01")
    return


@app.cell
def _(g):
    g.feedback("glm-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. What happens when we vary the noise?

        How does the amount of noise in the data impact our model fits?

        Repeat the simulation with the default intensities but three levels of noise:
        `noise_levels = [0.1, 0.5, 1.0]` (passed to `simulate_voxel` as `sigma`). Store the three
        $r^2$ values in a numpy array `r2_by_noise` and plot $r^2$ against the noise level.
        """
    )
    return


@app.cell
def _(np, ols_estimator, plt, r_square, simulate_voxel):
    noise_levels = ...
    r2_by_noise = ...
    # YOUR CODE HERE
    pass
    return noise_levels, r2_by_noise


@app.cell
def _(g, mo, noise_levels, np, r2_by_noise):
    mo.stop(
        any(v is ... for v in (noise_levels, r2_by_noise)),
        mo.md("**Complete the code cell above first.** This check runs once `noise_levels` and `r2_by_noise` are defined."),
    )

    g.check(
        "glm-q02: r-squared vs noise",
        [
            (
                isinstance(r2_by_noise, np.ndarray) and r2_by_noise.shape == (3,),
                "r2_by_noise should be a numpy array with one r-squared per noise level (shape (3,))",
                1,
            ),
            (
                bool(np.all((r2_by_noise >= 0) & (r2_by_noise <= 1))),
                "r-squared values must lie between 0 and 1",
                1,
            ),
            (
                bool(np.all(np.diff(r2_by_noise) < 0)),
                "r-squared should decrease as the noise level increases",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("glm-q02")
    return


@app.cell
def _(g):
    g.feedback("glm-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. How many trials do we need?

        A common question in experimental design is determining the optimal number of trials.

        Use `simulate_voxel` to build design matrices with `trial_counts = [5, 10, 20]` trials per
        condition (only `X` is needed here). For each design, compute the efficiency of the
        face-minus-object contrast `[0, 1, -1]` with `contrast_efficiency`, store the three values
        in a numpy array `efficiency_by_trials`, and plot efficiency against the number of trials.
        """
    )
    return


@app.cell
def _(contrast_efficiency, np, plt, simulate_voxel):
    efficiency_by_trials = ...
    trial_counts = ...
    # YOUR CODE HERE
    pass
    return efficiency_by_trials, trial_counts


@app.cell
def _(efficiency_by_trials, g, mo, np, trial_counts):
    mo.stop(
        any(v is ... for v in (efficiency_by_trials, trial_counts)),
        mo.md("**Complete the code cell above first.** This check runs once `efficiency_by_trials` and `trial_counts` are defined."),
    )

    g.check(
        "glm-q03: contrast efficiency vs number of trials",
        [
            (
                isinstance(efficiency_by_trials, np.ndarray) and efficiency_by_trials.shape == (3,),
                "efficiency_by_trials should be a numpy array with one value per trial count (shape (3,))",
                1,
            ),
            (
                bool(np.all(efficiency_by_trials > 0)),
                "efficiency is 1 / variance of the contrast, so it must be positive",
                1,
            ),
            (
                bool(np.all(np.diff(efficiency_by_trials) > 0)),
                "efficiency should increase with the number of trials",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("glm-q03")
    return


@app.cell
def _(g):
    g.feedback("glm-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. What is the impact of the stimulus duration?

        What if one condition simply results in processes that systematically take longer than
        the other condition?

        Create a simulation in which the signal intensity of the two conditions is identical
        (`face_intensity=1`, `object_intensity=1`) but the object condition lasts longer:
        `object_duration=4` TRs versus `face_duration=1`. Remember that `simulate_voxel` always
        *models* each trial as a single-TR event, so the design matrix does not know about the
        longer duration. Fit the GLM to this simulation and store the estimates in `betas_longer`.
        As a comparison, also fit a simulation where both durations are 1 TR (same intensities)
        and store those estimates in `betas_equal`. Make a bar plot comparing the two sets of
        $\beta$ estimates (intercept, faces, objects).
        """
    )
    return


@app.cell
def _(np, ols_estimator, plt, simulate_voxel):
    betas_equal = ...
    betas_longer = ...
    # YOUR CODE HERE
    pass
    return betas_equal, betas_longer


@app.cell
def _(betas_equal, betas_longer, g, mo, np):
    mo.stop(
        any(v is ... for v in (betas_equal, betas_longer)),
        mo.md("**Complete the code cell above first.** This check runs once `betas_equal` and `betas_longer` are defined."),
    )

    g.check(
        "glm-q04: beta estimates vs stimulus duration",
        [
            (
                np.shape(betas_equal) == (3,) and np.shape(betas_longer) == (3,),
                "betas_equal and betas_longer should each hold three estimates (intercept, faces, objects)",
                1,
            ),
            (
                abs(betas_equal[1] - betas_equal[2]) < 0.25,
                "with equal intensities and durations, the face and object betas should be roughly equal",
                1,
            ),
            (
                betas_longer[2] > 1.2 * betas_longer[1],
                "the longer-duration condition should produce a clearly larger beta even though its intensity is the same",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("glm-q04")
    return


@app.cell
def _(g):
    g.feedback("glm-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. Interpretation

        In Q4 the two conditions had *identical* signal intensity, yet the $\beta$ estimates
        differed. In two or three sentences, explain why this happens, and what it implies for
        interpreting a difference between two conditions' $\beta$ estimates (or a contrast between
        them) in a real experiment.
        """
    )
    return


@app.cell
def _(mo):
    answer = mo.ui.text_area(placeholder="Your answer...", full_width=True)
    answer
    return (answer,)


@app.cell
def _(answer, g):
    # UI element values are not part of the notebook file, so pass them as outputs;
    # they are stored with the attempt and shown to the grader next to the notebook.
    g.submit_button("glm-q05", outputs={"answer": answer.value})
    return


@app.cell
def _(g):
    g.feedback("glm-q05")
    return


if __name__ == "__main__":
    app.run()
