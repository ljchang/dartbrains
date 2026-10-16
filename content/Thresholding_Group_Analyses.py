# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dartbrains-tools>=0.3.3",
#     "matplotlib",
#     "nilearn",
#     "nltools==0.6.0.dev2",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
#     "seaborn",
# ]
# 
# [tool.grader]
# server = "https://grader.dartbrains.org"
# course = "neuroimaging"
# term = "2026-fall"
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from dartbrains_tools.notebook_utils import image
    from dartbrains_tools.notebook_utils import youtube

    return image, mo, youtube


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Thresholding Group Analyses

    *Written by Luke Chang*

    Now that we have learned how to estimate a single-subject model, create contrasts, and run a group-level analysis, the next important topic to cover is how we can threshold these group maps. This is not as straightforward as it might seem as we need to be able to correct for multiple comparisons.

    In this tutorial, we will cover how we go from modeling brain responses in each voxel for a single participant to making inferences about the group. We will cover the following topics:

    - Issues with correcting for multiple comparisons
    - Family Wise Error Rate
    - Bonferroni Correction
    - False Discovery Rate
    - Permutation tests and the resolution of a p-value
    - Max-statistic correction
    - Cluster inference, spatial dependence, and localization
    - Running corrections on real data, and reporting what survives

    Let's get started by watching an overview of multiple comparisons by Martin Lindquist.
    """)
    return


@app.cell
def _(youtube):
    youtube('AalIM9-5-Pk')
    return


@app.cell(hide_code=True)
def _(image, mo):
    mo.vstack([
        mo.md(r"""
    The primary goal in fMRI data analysis is to make inferences about how the brain processes information. These inferences can be in the form of predictions, but most often we are testing hypotheses about whether a particular region of the brain is involved in a specific type of process. This requires rejecting a $H_0$ hypothesis (i.e., that there is no effect). Null hypothesis testing is traditionally performed by specifying contrasts between different conditions of an experimental design and assessing if these differences between conditions are reliably present across many participants. There are two main types of errors in null-hypothesis testing.

    *Type I error*
    - $H_0$ is true, but we mistakenly reject it (i.e., False Positive)
    - This is controlled by significance level $\alpha$.

    *Type II error*
    - $H_0$ is false, but we fail to reject it (False Negative)

    The probability that a hypothesis test will correctly reject a false null hypothesis is described as the *power* of the test.

    Hypothesis testing in fMRI is complicated by the fact that we are running many tests across each voxel in the brain (hundreds of thousands of tests). Selecting an appropriate threshold requires finding a balance between sensitivity (i.e., true positive rate) and specificity (i.e., true negative rate). There are two main approaches to correcting for multiple tests in fMRI data analysis.

    **Familywise Error Rate** (FWER) attempts to control the probability of finding *any* false positives. Mathematically, FWER can be defined as the probability $P$ of observing any false positive ${FWER} = P({False Positives}\geq 1)$.

    While, **False Discovery Rate** (FDR) attempts to control the proportion of false positives among rejected tests. Formally, this is the expected proportion of false positive to the observed number of significant tests ${FDR} = E(\frac{False Positives}{\max(Significant Tests, 1)})$; the fraction is defined as zero when there are no discoveries.

    This should probably be no surprise to anyone, but fMRI studies can be expensive, and power depends on sample size, effect size, noise, and the correction procedure. Here is a simulation by Jeannette Mumford to show approximately how many participants you would need to achieve 80% power assuming a specific effect size in your contrast.
    """),
        image("thresholding/fmri_power.png"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulations

    Let's explore the concept of false positives to get an intuition about what the overall goals and issues are in controlling for multiple tests.

    Let's load the modules we need for this tutorial. `nltools.SimulateGrid` generates and plots the example grids. Reusable maximum-statistic and cluster helpers live in `dartbrains_tools.simulations`.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

    import os
    import glob
    import contextlib
    import io
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from scipy.ndimage import gaussian_filter
    from scipy.stats import ttest_1samp, t as student_t
    from nilearn.glm.second_level import non_parametric_inference
    from nilearn.plotting import plot_stat_map
    from nltools.data import BrainData
    from nltools import SimulateGrid
    from nltools.algorithms import fdr, threshold
    from nltools.templates import fetch_resource
    from nltools.algorithms.inference import one_sample_permutation_test
    from dartbrains_tools.data import localizer
    from dartbrains_tools.simulations import simulate_max_t, cluster_fwe_p

    return (
        BrainData,
        SimulateGrid,
        cluster_fwe_p,
        contextlib,
        fdr,
        fetch_resource,
        gaussian_filter,
        go,
        io,
        localizer,
        non_parametric_inference,
        np,
        one_sample_permutation_test,
        pd,
        plot_stat_map,
        plt,
        sns,
        simulate_max_t,
        student_t,
        threshold,
        ttest_1samp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Okay, let's get started and generate 100 x 100 voxels from $\mathcal{N}(0,1)$ distribution for 20 independent participants.
    """)
    return


@app.cell
def _(SimulateGrid, plt, sns):
    simulation = SimulateGrid(grid_width=100, n_subjects=20, random_state=0)
    _f, _a = plt.subplots(nrows=5, ncols=4, figsize=(15, 15), sharex=True, sharey=True)
    counter = 0
    for col in range(4):
        for row in range(5):
            sns.heatmap(simulation.data[:, :, counter], ax=_a[row, col], cmap='RdBu_r', vmin=-4, vmax=4)
            _a[row, col].set_title(f'Subject {counter + 1}', fontsize=16)
            counter = counter + 1
    plt.tight_layout()
    plt.gcf()
    return (simulation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each subject's simulated data is on a 100 x 100 grid. Think of this as a slice from their brain, where each pixel corresponds to the same spatial location across all participants. We have generated random noise separately for each subject. We have not added any true signal in this simulation yet.

    This figure is simply to highlight that we are working with 20 independent subjects. In the rest of the plots, we will be working with a single grid that aggregates the results across participants.

    Now we are going to start running some simulations to get a sense of the number of false positives we might expect to observe with this data. We will now run an independent one-sample t-test on every pixel in the grid across all 20 participants.
    """)
    return


@app.cell
def _(plt, simulation, sns):
    simulation.fit()

    sns.heatmap(simulation.t_values, square=True, cmap='RdBu_r', vmin=-4, vmax=4)
    plt.title("T Values", fontsize=18)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even though there was no signal in this simulation, you can see that there are a number of pixels in the grid that exceed a t-value above 2 and below -2, which is the approximate cutoff for p < 0.05. These are all false positives.

    Now let's apply a threshold. We can specify thresholds at a specific t-value using the `threshold_type='t'`. Alternatively, we can specify a specific p-value using the `threshold_type='p'`. To calculate the number of false positives, we can simply count the number of tests that exceed this threshold.

    If we run this simulation again 100 times, we can estimate the false positive rate, which is the average number of false positives over all 100 simulations.

    Let's see what this looks like for a threshold of p < 0.05.
    """)
    return


@app.cell
def _(SimulateGrid, plt):
    _threshold = 0.05
    simulation_1 = SimulateGrid(grid_width=100, n_subjects=20, random_state=0)
    simulation_1.plot_grid_simulation(threshold=_threshold, threshold_type='p', n_simulations=100)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The left panel is the average over all of the participants. The middle panel show voxels that exceed the statistical threshold. The right panel is the overall false-positive rate across the 100 simulations.

    In this simulation, a threshold of p < 0.05 results in observing at least one voxels that is a false positive across every one of our 100 simulations.

    What if we looked at a fewer number of voxels? How would this change our false positive rate?
    """)
    return


@app.cell
def _(SimulateGrid, plt):
    _threshold = 0.05
    simulation_2 = SimulateGrid(grid_width=5, n_subjects=20, random_state=0)
    simulation_2.plot_grid_simulation(threshold=_threshold, threshold_type='p', n_simulations=100)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This simulation shows that examining fewer numbers of voxels will yield considerably less false positives. One common approach to controlling for multiple tests involves only looking for voxels within a specific region of interest (e.g., small volume correction), or looking at average activation within a larger region (e.g., ROI based analyses).

    What about if we increase the threshold on our original 100 x 100 grid?
    """)
    return


@app.cell
def _(SimulateGrid, plt):
    _threshold = 0.0001
    simulation_3 = SimulateGrid(grid_width=100, n_subjects=20, random_state=0)
    simulation_3.plot_grid_simulation(threshold=_threshold, threshold_type='p', n_simulations=100)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can see that this dramatically decreases the number of false positives to the point that some of the simulations no longer contain any false positives.

    ### How bad is *uncorrected* really?

    Before we start correcting, it is worth pinning down the size of the problem with a number rather than an impression. Suppose we test every voxel at p < 0.001 — a threshold that sounds strict, and which you will see reported in plenty of papers as though it were a correction.

    If the voxels were independent, the probability that a map made of pure noise contains *at least one* false positive is

    $$P(\text{any false positive}) = 1 - (1 - \alpha)^{M}$$

    for $M$ voxels. Let's check that against a simulation on a smaller 30 x 30 grid, where $M = 900$.
    """)
    return


@app.cell
def _(np, ttest_1samp):
    # 100 simulated experiments containing nothing but noise. Every voxel that
    # survives a threshold here is a false positive by construction.
    _rng = np.random.default_rng(0)
    _width, _n_subjects, _n_sims = 30, 20, 100
    _n_voxels = _width * _width

    _any_false_positive = 0
    for _i in range(_n_sims):
        _noise = _rng.standard_normal((_n_subjects, _n_voxels))
        _tvals, _pvals = ttest_1samp(_noise, 0, axis=0)
        _any_false_positive += (_pvals < 0.001).any()

    empirical_any_fp = _any_false_positive / _n_sims
    theoretical_any_fp = 1 - (1 - 0.001) ** _n_voxels

    print(f"Maps with >= 1 false positive at p < .001: {empirical_any_fp:.0%}")
    print(f"Theory, 1 - (1 - .001)^{_n_voxels}:            {theoretical_any_fp:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Roughly **half of all completely null experiments** would report at least one "significant" voxel at p < 0.001, and the simulation lands close to the analytic prediction. And this is on a 900-voxel grid — a real brain has a couple of hundred thousand voxels, where the probability is essentially 1.

    This is the entire motivation for everything that follows. An uncorrected threshold does not control anything at the level of the map; it only controls the error rate of a single voxel considered in isolation, which is not the question anyone is actually asking.

    What is the optimal threshold that will give us an $\alpha=0.05$?

    To calculate this, we will run 100 simulations at different threshold levels to find the threshold that leads to a false positive rate that is lower than our alpha value.

    We could search over t-values, or p-values. Let's explore t-values first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The sweep below reuses simulated maps across cutoffs and runs quickly on a CPU. Spatial permutation tests later in the chapter, especially TFCE, can take much longer. If you are taking the course, sign in with your Dartmouth account before running them: the results are then kept in your course storage, and the course keeps a copy your instructor computed ahead of time, so they load in seconds instead. Signing in later works too: the slow cells rerun and pick up the stored copy. If you are not at Dartmouth, skip the button; everything works the same way.
    """)
    return


@app.cell
def _():
    from dartbrains_tools import storage

    signin = storage.signin_button()
    signin
    return signin, storage


@app.cell
def _(signin, storage):
    # The slow cells below depend on this one, so signing in reruns them against
    # course storage. Their cache keys are the same for every signed-in reader:
    # the button's value carries no token (marimo-grader-client >= 0.2.2).
    storage_ready = storage.connect(signin)
    return (storage_ready,)


@app.cell
def _(go, np, simulate_max_t, student_t):
    _alpha, _n_sims, _n_voxels, _df = 0.05, 100, 10_000, 19
    _cutoffs = np.arange(3, 7.01, 0.1)
    # One set of 100 maps, not a new set for each cutoff. Only the maxima
    # are retained, so memory does not grow with the number of cutoffs.
    _maxima = simulate_max_t(n_simulations=_n_sims)
    _rates = (_maxima[:, None] >= _cutoffs).mean(axis=0)
    _per_voxel_p = 2 * student_t.sf(_cutoffs, _df)
    _theory = -np.expm1(_n_voxels * np.log1p(-_per_voxel_p))
    # Pointwise Wilson intervals remain informative even for 0/100 hits.
    _z = 1.96
    _denom = 1 + _z**2 / _n_sims
    _centre = (_rates + _z**2 / (2 * _n_sims)) / _denom
    _half = _z * np.sqrt(_rates * (1 - _rates) / _n_sims + _z**2 / (4 * _n_sims**2)) / _denom
    bonferroni_t = float(student_t.isf(_alpha / (2 * _n_voxels), _df))
    _fig = go.Figure([
        go.Scatter(x=_cutoffs, y=_centre + _half, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'),
        go.Scatter(x=_cutoffs, y=_centre - _half, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31,119,180,0.18)', name='95% pointwise Wilson interval', hoverinfo='skip'),
        go.Scatter(x=_cutoffs, y=_rates, mode='lines+markers', name='100 simulated maps', line=dict(color='#1f77b4'), hovertemplate='|t| cutoff=%{x:.2f}<br>FWER=%{y:.1%}<extra></extra>'),
        go.Scatter(x=_cutoffs, y=_theory, mode='lines', name='Independent-voxel theory', line=dict(color='#d95f02', dash='dash')),
    ])
    _fig.add_hline(y=_alpha, line_dash='dot', annotation_text='Target FWER = 5%')
    _fig.add_vline(x=bonferroni_t, line_dash='dash', annotation_text=f'Bonferroni |t| = {bonferroni_t:.2f}')
    _fig.update_layout(xaxis_title='Two-sided threshold |t| (19 degrees of freedom)', yaxis_title='Probability of any false positive (FWER)', yaxis=dict(range=[0, 1], tickformat='.0%'), title='10,000 null voxels: simulation and theory', height=450, legend=dict(orientation='h', y=-.3), margin=dict(l=60, r=20, t=50, b=110))
    _fig
    return (bonferroni_t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simulated curve falls toward 5% near the Bonferroni cutoff. Its shaded interval shows Monte Carlo uncertainty: 100 experiments cannot pinpoint a 5% error rate precisely. Reusing the same maps makes the empirical curve monotone and avoids repeating the expensive t-tests. The dashed curve gives the exact FWER for independent voxels; Bonferroni provides a bound that also remains valid for dependent tests.

    Let's use that threshold for our simulation again.
    """)
    return


@app.cell
def _(SimulateGrid, bonferroni_t, plt):
    simulation_4 = SimulateGrid(grid_width=100, n_subjects=20, random_state=0)
    simulation_4.plot_grid_simulation(threshold=bonferroni_t, threshold_type='t', n_simulations=100)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice, that we are now observing a false positive rate of approximately .05, though this estimate varies across random seeds. We fix the seed so the lesson is reproducible.

    Another way to find the threshold that controls FWER is to divide alpha by the total number of tests across voxels. This is the **Bonferroni correction**. It controls FWER under arbitrary dependence, provided each null p-value is valid; independence is not required.

    ${bonferroni} = \frac{\alpha}{M}$, where $M$ is the number of voxels.
    """)
    return


@app.cell
def _(SimulateGrid, plt):
    _grid_width = 100
    _threshold = 0.05 / _grid_width ** 2
    simulation_5 = SimulateGrid(grid_width=_grid_width, n_subjects=20, random_state=0)
    simulation_5.plot_grid_simulation(threshold=_threshold, threshold_type='p', n_simulations=100)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This seems like a great way to ensure that we minimize our false positives.

    Now what happens when start adding signal to our simulation?

    We will represent signal in a smaller square in the middle of the simulation. The width of the square can be changed using the `signal_width` parameter. The amplitude of this signal is controlled by the `signal_amplitude` parameter.

    Let's see how well the bonferroni threshold performs when we add 100 voxels of signal.
    """)
    return


@app.cell
def _(SimulateGrid, plt):
    _grid_width = 100
    _threshold = 0.05 / _grid_width ** 2
    signal_width = 10
    _signal_amplitude = 1
    simulation_6 = SimulateGrid(signal_amplitude=_signal_amplitude, signal_width=10, grid_width=_grid_width, n_subjects=20, random_state=0)
    simulation_6.plot_grid_simulation(threshold=_threshold, threshold_type='p', n_simulations=100)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we show how many voxels were identified using the bonferroni correction.

    In the left panel is the average data across all 20 participants. The second panel, shows the voxels that exceed the statistical threshold. The third panel shows the false positive rate, and the 4th panel furthest on the right shows the average signal recovery (how many voxels survived within the true signal square across all 100 simulations.

    We can see that we have an effective false positive rate approximately equal to our alpha threshold. However, our threshold is so high, that we can barely detect any true signal with this amplitude. In fact, we are only recovering about 12% of the voxels that should have signal.

    This simulation highlights the main issue with using bonferroni correction in practice. The threshold is so conservative that the magnitude of an effect needs to be unreasonably large to survive correction over hundreds of thousands of voxels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## False Discovery Rate (FDR)
    You may be wondering why we need to control for *any* false positive when testing across hundreds of thousands of voxels. Surely a few are okay as long as they don't overwhelm the true signal.

    Let's learn about the False Discovery Rate (FDR) from another video by Martin Lindquist.
    """)
    return


@app.cell
def _(youtube):
    youtube('W9ogBO4GEzA')
    return


@app.cell(hide_code=True)
def _(image, mo):
    mo.vstack([
        mo.md(r"""
    The *false discovery rate* (FDR) is a more recent development in multiple testing correction originally described by [Benjamini & Hochberg, 1995](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1995.tb02031.x). While FWER is the probability of any false positives occurring in a family of tests, the FDR is the expected proportion of false positives among significant tests.

    The FDR is fairly straightforward to calculate.
     1. Choose a target expected false discovery proportion $q$, such as 0.05.
     2. Sort the $M$ p-values: $p_{(1)} \leq \cdots \leq p_{(M)}$.
     3. Find the **largest** rank $k$ satisfying $p_{(k)} \leq (k/M)q$.
     4. Reject every test with $p \leq p_{(k)}$, including ties. If no rank qualifies, reject none.
    """),
        image("thresholding/fdr_calc.png"),
        mo.md(r"""
    FDR control is an **expectation across repeated experiments**. It does not guarantee that 95% of the discoveries in this particular brain map are true, nor assign a 95% probability to an individual voxel. Benjamini–Hochberg controls FDR under independence or appropriate positive dependence; arbitrary dependence requires additional care, for example the more conservative Benjamini–Yekutieli procedure ([SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html)).

    With more strong signals, the procedure can admit a larger p-value cutoff and gain sensitivity. Under the complete null, every discovery is false, so the false discovery proportion is 1 whenever there is any discovery and 0 otherwise: FDR then equals FWER. Every procedure controlling FWER also controls FDR, but that does not impose a universal ordering on every pair of correction procedures.

    This figure is taken from Poldrack, Mumford, & Nichols (2011) and compares different procedures to control for multiple tests.
    """),
        image("thresholding/fdr.png"),
        mo.md(r"""
    For a more indepth overview of FDR, see this [tutorial](https://matthew-brett.github.io/teaching/fdr.html) by Matthew Brett.

    Let's now try to apply FDR to our own simulations. All we need to do is add a `correction='fdr'` flag to our simulation plot. We need to make sure that the `threshold=0.05` to use the correct $q$.
    """),
    ])
    return


@app.cell
def _(SimulateGrid, plt):
    _grid_width = 100
    _threshold = 0.05
    _signal_amplitude = 1
    simulation_8 = SimulateGrid(signal_amplitude=_signal_amplitude, signal_width=10, grid_width=_grid_width, n_subjects=20, random_state=0)
    simulation_8.plot_grid_simulation(threshold=_threshold, threshold_type='q', n_simulations=100, correction='fdr')
    print(f'FDR q < 0.05 corresponds to p-value of {simulation_8.corrected_threshold}')
    plt.gcf()
    return (simulation_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The printed FDR cutoff is more liberal than the Bonferroni threshold of p < 0.000005 and allows us to recover much more signal as a consequence. You can see that at this threshold there are more false positives, which leads to a much higher overall false positive rate. Remember, this metric is only used for calculating the family wise error rate and indicates the presence of *any* false positive across each of our 100 simulations.

    To calculate the empirical false discovery rate, we need to calculate the percent of any activated voxels that were false positives.
    """)
    return


@app.cell
def _(go, simulation_8):
    _fig = go.Figure()
    _fig.add_trace(go.Histogram(
        x=simulation_8.multiple_fdr, name='False discovery proportion',
        hovertemplate='FDR=%{x:.3f}<br>count=%{y}<extra></extra>',
    ))
    _fig.update_layout(
        xaxis_title='False discovery proportion in one simulated experiment',
        yaxis_title='Frequency',
        title='Variation across experiments; FDR is the mean of this distribution',
        height=400,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each histogram entry is a false discovery **proportion** from one experiment. Its average estimates FDR; individual experiments can exceed 5% even when the procedure controls the expectation.

    Both Bonferroni and FDR share an assumption that we have quietly been making all along: that the p-value attached to each voxel is correct. Both procedures take a set of p-values as input and decide which ones to keep — neither of them checks where those p-values came from. They came from the $t$ distribution, which is only the right reference distribution if the data are normally distributed and independent across subjects.

    Next we build a null distribution from the data. This replaces the normality assumption with assumptions about valid resampling; it does not remove assumptions altogether.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Family Wise Error Rate

    At this point you may be wondering if it even makes sense to assume that each test is independent. It seems reasonable to expect some degree of spatial correlation in our data. Our simulation is a good example of this as we have a square that contains signal across contiguous voxels. In practice, most of our functional neuroanatomy that we are investigating is larger than a single voxel and our spatial smoothing preprocessing step increase the spatial correlation.

    It can be shown that the Bonferroni correction is overally conservative in the presence of spatial dependence and results in a decreased power to detect voxels that are truly active.

    Let's watch a video by Martin Lindquist to learn more about different ways to control for the Family Wise Error Rate.
    """)
    return


@app.cell
def _(youtube):
    youtube('MxQeEdVNihg')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Permutation tests

    As an alternative to assuming a parametric null distribution, nonparametric methods use the data themselves to find the appropriate distribution. These methods can provide substantial improvements in power and validity, particularly with small sample sizes, so we recommend these in general. They can also be used to verify the validity of the less computationally expensive parametric approaches.

    For a one-sample test, the relevant procedure is a **sign-flipping** test. The logic is this: if a condition truly has no effect, then whether a given participant's value came out positive or negative is arbitrary. So we can generate a dataset that the null hypothesis could plausibly have produced by randomly flipping the sign of each participant's map, and recomputing the group statistic. Repeat this many times and you have an empirical null distribution — one that does not require normality. It does require independent subjects and errors symmetric about zero under the null. Flip the **whole participant map** with one sign to preserve its spatial dependence ([FSL design guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/glm.html)).

    `nltools` implements this in `one_sample_permutation_test`. Passing `return_null=True` gives us the full null distribution, with shape `(n_permutations, n_voxels)`, rather than just the p-values. That null is the raw material for everything in the rest of this tutorial, so it is worth looking at directly.

    We will drop down to a 30 x 30 grid with a small 6 x 6 square of true signal in the middle, which keeps the null distribution small enough to inspect.
    """)
    return


@app.cell(hide_code=True)
def _(contextlib, io):
    def quiet():
        """Suppress the per-call tqdm progress bar from the permutation test.

        Retained for compatibility with older nltools releases; the pinned
        release defaults to progress_bar=False.
        """
        return contextlib.redirect_stderr(io.StringIO())

    return (quiet,)


@app.cell
def _(SimulateGrid, np, one_sample_permutation_test, quiet):
    N_PERMUTE = 1000
    GRID_WIDTH = 30

    perm_sim = SimulateGrid(
        grid_width=GRID_WIDTH, n_subjects=20,
        signal_width=6, signal_amplitude=1, random_state=0,
    )

    # SimulateGrid stores data as (width, width, n_subjects); the permutation
    # test wants observations in rows, so (n_subjects, n_voxels).
    perm_data = perm_sim.data.reshape(-1, perm_sim.n_subjects).T

    # n_jobs=1 is deliberate: at this problem size the joblib pool costs more
    # to spin up than the permutations themselves take to run.
    with quiet():
        perm_result = one_sample_permutation_test(
            perm_data, n_permute=N_PERMUTE, return_null=True,
            random_state=0, n_jobs=1,
        )

    perm_null = np.asarray(perm_result['null_dist'])
    n_voxels = perm_null.shape[1]

    print(f"null distribution shape: {perm_null.shape}  (n_permutations x n_voxels)")
    print(f"smallest attainable p    = 1 / (1 + {N_PERMUTE}) = {1 / (N_PERMUTE + 1):.4f}")
    print(f"smallest observed p      = {np.min(perm_result['p']):.4f}")
    print(f"Bonferroni target for {n_voxels} voxels = {0.05 / n_voxels:.2e}")
    return GRID_WIDTH, perm_null, perm_result, perm_sim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A permutation p-value has a resolution limit

    Look carefully at those last three numbers, because they contain a trap that is easy to walk into.

    A permutation p-value is a proportion: *out of all the permutations I ran, how many produced a statistic at least as extreme as the one I observed?* If none of them did, the p-value is not zero — it is $1/(n_{permute}+1)$, because the observed data is itself one of the possible arrangements. With 1,000 permutations, **the smallest p-value that can exist is 0.001**, and our most extreme voxel is sitting exactly on that floor.

    Now compare that to the Bonferroni target of $0.05 / 900 = 5.6 \times 10^{-5}$. It is more than an order of magnitude below the smallest number our permutation test is capable of producing. **You cannot Bonferroni-correct 900 voxels using 1,000 permutations.** No voxel can ever survive, no matter how strong the effect. To reach a whole-brain Bonferroni threshold this way you would need millions of permutations.

    So classical corrections and permutation tests do not simply mix. If you feed permutation p-values into Bonferroni or FDR you will silently get an answer that is bounded by your computational budget rather than by your data.

    The resolution to this is one of the more elegant ideas in the multiple comparisons literature, and it is the subject of the next section: instead of correcting per-voxel p-values, build a null distribution for the *entire map at once*.
    """)
    return


@app.cell
def _(go, np, perm_null, perm_result):
    # Show the null distribution for the single most extreme voxel.
    _voxel = int(np.argmax(np.abs(perm_result['mean'])))
    _observed = perm_result['mean'][_voxel]

    _fig = go.Figure()
    _fig.add_trace(go.Histogram(
        x=perm_null[:, _voxel], nbinsx=60, name='null (sign-flipped)',
        hovertemplate='statistic=%{x:.3f}<br>count=%{y}<extra></extra>',
    ))
    _fig.add_vline(
        x=_observed, line=dict(color='red', width=3),
        annotation_text='observed', annotation_position='top right',
    )
    _fig.update_layout(
        xaxis_title='Group mean under the null',
        yaxis_title='Frequency',
        title=f'Sign-flip null distribution for the most extreme voxel (voxel {_voxel})',
        height=400,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Max-statistic correction

    Here is the key move. We do not actually care about the null distribution of *a* voxel. We care about the null distribution of the **largest value anywhere in the map**, because a family-wise error is by definition the event that *something*, somewhere, crossed the line when it should not have.

    So for each permutation, instead of keeping all 900 values, we keep only the maximum absolute value across the whole grid. That gives us one number per permutation and hence a null distribution of the maximum. The population 95th percentile targets a 5% probability that any voxel exceeds it under the complete null. We estimate that percentile from a finite sample of permutations, so the cutoff has Monte Carlo uncertainty.

    This is a **family-wise error rate correction**, and it takes three lines. It avoids needing a per-voxel p-value as small as α/M: instead we compare the observed map with a null distribution of map maxima at α. Corrected permutation p-values still have resolution 1/(B+1). The percentile cutoff shown here is an approximation useful for teaching; 1,000 permutations can demonstrate the procedure. The estimated cutoff still has Monte Carlo uncertainty; check stability across seeds and use more permutations for final analyses.
    """)
    return


@app.cell
def _(np):
    def max_stat_threshold(null, alpha=0.05):
        """Critical value from the null distribution of the maximum statistic.

        `null` has shape (n_permutations, n_voxels). Taking the max across
        voxels collapses each permutation to a single number: the most extreme
        value anywhere in that permuted map.
        """
        return np.percentile(np.abs(null).max(axis=1), 100 * (1 - alpha))

    return (max_stat_threshold,)


@app.cell
def _(go, max_stat_threshold, np, perm_null):
    _single_voxel = np.abs(perm_null[:, 0])
    _map_max = np.abs(perm_null).max(axis=1)
    _critical = max_stat_threshold(perm_null)

    _fig = go.Figure()
    _fig.add_trace(go.Histogram(
        x=_single_voxel, nbinsx=60, opacity=0.65,
        name='null for one voxel',
    ))
    _fig.add_trace(go.Histogram(
        x=_map_max, nbinsx=60, opacity=0.65,
        name='null of the map maximum',
    ))
    _fig.add_vline(
        x=_critical, line=dict(color='red', dash='dash', width=3),
        annotation_text=f'95th pct = {_critical:.3f}', annotation_position='top right',
    )
    _fig.update_layout(
        barmode='overlay',
        xaxis_title='|group mean| under the null',
        yaxis_title='Frequency',
        title='One voxel vs. the maximum over 900 voxels',
        height=420,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two distributions are the whole story of multiple comparisons in one figure. The null for a single voxel is centered near zero. The null for the *maximum over 900 voxels* is shifted well to the right — because taking the largest of 900 draws from a null distribution reliably gives you a fairly large number. The gap between them is exactly the correction. Thresholding at the red line rather than at a single-voxel cutoff is what buys you FWER control.

    Let's apply it and see how much of the true signal square we recover.
    """)
    return


@app.cell
def _(
    GRID_WIDTH,
    max_stat_threshold,
    np,
    perm_null,
    perm_result,
    perm_sim,
    plt,
    sns,
):
    _critical = max_stat_threshold(perm_null)
    _observed = np.abs(perm_result['mean']).reshape(GRID_WIDTH, GRID_WIDTH)
    _survivors = (_observed > _critical).astype(float)

    _f, _a = plt.subplots(ncols=3, figsize=(16, 5))
    sns.heatmap(_observed, square=True, cmap='viridis', vmin=0, ax=_a[0])
    _a[0].set_title('Observed |group mean|', fontsize=15)
    sns.heatmap(perm_sim.signal_mask, square=True, cmap='gray_r', cbar=False, ax=_a[1])
    _a[1].set_title('True signal', fontsize=15)
    sns.heatmap(_survivors, square=True, cmap='gray_r', cbar=False, ax=_a[2])
    _a[2].set_title(f'Survives max-stat FWE (p < .05)\n{int(_survivors.sum())} voxels', fontsize=15)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A threshold is only worth trusting if it does what it claims. We asserted that this procedure controls the family-wise error rate at 5% — so let's verify it, rather than take it on faith.

    We generate 100 experiments containing **no signal whatsoever**, run the whole procedure on each, and count how often *any* voxel survives. If the correction is calibrated, that should happen about 5 times out of 100.
    """)
    return


@app.cell
def _(max_stat_threshold, np, one_sample_permutation_test, quiet):
    _rng = np.random.default_rng(1)
    _n_sims, _n_subjects, _n_voxels = 100, 20, 900

    _hits = 0
    with quiet():
        for _i in range(_n_sims):
            _noise = _rng.standard_normal((_n_subjects, _n_voxels))
            _res = one_sample_permutation_test(
                _noise, n_permute=500, return_null=True,
                random_state=_i, n_jobs=1,
            )
            _null = np.asarray(_res['null_dist'])
            _hits += (np.abs(_res['mean']) > max_stat_threshold(_null)).any()

    maxstat_fwe = _hits / _n_sims
    _se = (maxstat_fwe * (1 - maxstat_fwe) / _n_sims) ** 0.5
    print(f"Max-statistic FWE rate over {_n_sims} null experiments: "
          f"{maxstat_fwe:.1%} (+/- {1.96 * _se:.1%})")
    print("Nominal alpha: 5.0%")
    return


@app.cell(hide_code=True)
def _(image, mo):
    mo.vstack([
        mo.md(r"""
    Calibrated. The measured rate sits on top of the nominal 5%, within Monte Carlo error.

    ## Cluster inference

    Max-statistic correction controls errors at the level of individual voxels, which makes it strict. But we usually expect true activation to be spatially extended — a real effect should show up as a *blob*, not an isolated voxel. Cluster inference exploits this by asking a different question: not "is this voxel too high?" but "is this blob too big?"

    The classic implementation approximates the distribution of the maximum cluster size using Gaussian Random Field Theory (RFT), which attempts to account for the spatial dependence of the data.
    """),
        image("thresholding/fwer.png"),
        mo.md(r"""
    This requires specifying an initial threshold to determine the *Euler Characteristic* or the number of blobs minus the number of holes in the thresholded image. The number of voxels in the blob and the overall smoothness can be used to calculate something called *resels* or resolution elements and can be effectively thought of as the spatial units that need to be controlled for using FWER. We won't be going into too much detail with this approach as the mathematical details are somewhat complicated. Its validity depends on the random-field assumptions and cluster-forming threshold; neither smoothing nor having 20 participants is a general guarantee. Though we won't be spending time simulating this today, I encourage you to check out this Python [simulation](https://matthew-brett.github.io/teaching/random_fields.html) by Matthew Brett and this [chapter](https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch14.pdf) for an introduction to random field theory.
    """),
        image("thresholding/grf.png"),
        mo.md(r"""
    We can build the permutation version of this ourselves, and it is conceptually identical to the max-statistic procedure. Pick a **cluster-forming threshold**, apply it to each permuted map, find the largest surviving blob, and record its size. Do that for every permutation and you have a null distribution of the largest cluster size. Then ask where the observed cluster falls in that distribution.

    One detail matters: cluster sizes are integers and heavily tied, so we use a **permutation p-value**, $(\#\{null \geq observed\} + 1) / (n_{permute} + 1)$, rather than cutting at a percentile. A strict `>` against a percentile is needlessly conservative when many null clusters have exactly the same size.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `cluster_fwe_p` applies the same threshold to observed and permuted maps,
    labels four-connected neighbours, and compares their largest clusters.
    This teaching statistic uses absolute values, so adjacent positive and negative
    voxels can join; neuroimaging tools generally cluster the signs separately.

    The helper automatically uses Numba when available, with the same results as
    its SciPy fallback. Install `dartbrains-tools[simulation]` to enable it in a
    native Python environment. The first call compiles the kernel; later calls
    reuse it. Browser Python can use the SciPy fallback.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial smoothness changes the cluster-size distribution

    Before we run it, distinguish two ideas: parametric random-field methods assume a spatial model; permutation cluster tests preserve the spatial structure already present in the participant maps.

    `SimulateGrid` generates each voxel independently — its noise is a plain `randn(width, width, n_subjects)`, with no spatial structure at all. Real brain data is nothing like that: neighbouring voxels are correlated through the underlying physiology, through head motion, through interpolation during registration, and above all through the spatial smoothing we deliberately apply during preprocessing.

    A permutation cluster test can be valid on independent noise too. At a stringent forming threshold, its clusters are mostly tiny and the discrete null distribution may make the test conservative. Smoothing produces larger clusters and changes that null; it is not a prerequisite for the validity of a permutation test.

    Let's demonstrate this directly by running the identical procedure twice — once on independent noise, and once on the same noise after applying a Gaussian filter.
    """)
    return


@app.cell
def _(
    cluster_fwe_p,
    gaussian_filter,
    np,
    one_sample_permutation_test,
    pd,
    quiet,
):
    def run_cluster_calibration(smoothing_sigma, forming_percentile=99, n_sims=100, width=30, n_subjects=20,
                                n_permute=500, alpha=0.05):
        """Family-wise error rate of the cluster test under the complete null."""
        rng = np.random.default_rng(2)
        hits, observed_sizes = 0, []
        with quiet():
            for i in range(n_sims):
                data = rng.standard_normal((n_subjects, width, width))
                if smoothing_sigma:
                    data = np.stack([gaussian_filter(x, sigma=smoothing_sigma) for x in data])
                    data = data / data.std()          # renormalize after smoothing
                res = one_sample_permutation_test(
                    data.reshape(n_subjects, -1), n_permute=n_permute,
                    return_null=True, random_state=i, n_jobs=1,
                )
                null = np.asarray(res['null_dist'])
                forming = np.percentile(np.abs(null), forming_percentile)
                p_value, size, _ = cluster_fwe_p(res['mean'], null, width, forming)
                hits += p_value < alpha
                observed_sizes.append(size)
        return hits / n_sims, float(np.median(observed_sizes))


    cluster_calibration = pd.DataFrame(
        [
            {'smoothing sigma': _s,
             'cluster FWE rate': _rate,
             'median largest cluster (voxels)': _median}
            for _s in (0, 2.0)
            for _rate, _median in [run_cluster_calibration(_s)]
        ]
    )
    cluster_calibration
    return cluster_calibration, run_cluster_calibration


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the right-hand column first, because it is the stable result: on independent noise the largest blob you can expect to find is about **1 voxel**, while after smoothing it is around **5 voxels**. The unsmoothed null is highly discrete, so there may be no attainable cluster size with a rejection probability close to exactly 5%. This affects resolution and sensitivity, not the basic validity of the permutation procedure.

    The error rates in the middle column point the same direction, but treat them with more caution than the tidy story usually told about them: with only 100 simulations the Monte Carlo error on a rate near 5% is roughly ±4%, so the two numbers are not as far apart as they look. Do not over-read a single run — if you re-run this cell with a different seed the rates will move around noticeably, while the median cluster sizes will not.

    The general lesson is robust, and it is the mechanism behind [Eklund, Nichols & Knutsson (2016)](https://www.pnas.org/content/113/28/7900): **a misspecified parametric spatial model can invalidate cluster-level inference.** Permutation methods instead require a valid resampling scheme, including symmetry for this one-sample sign-flip test. Eklund et al. found that the parametric (RFT) implementations in the major software packages were mis-modelling that autocorrelation and, as a result, producing false-positive rates far above nominal — in some configurations up to 70% instead of 5%.

    This is why cluster extent thresholding has become controversial. A related paper by [Woo et al. 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4214144/) showed that liberal forming thresholds can harm spatial localization and can invalidate parametric cluster approximations. This does not imply that a correctly calibrated permutation cluster test must inflate FWER at every liberal threshold. There is no optimal way to select the initial threshold and often slight changes will give very different results. Furthermore, this approach does not appear to work equally well across all types of findings, and it seems potentially problematic to assume that spatial smoothness is constant over the brain. Finally, it is important to note that this approach only allows us to make inferences for the entire cluster. We can say that there is some voxel in the cluster that is significant, but we can't really pinpoint which voxels within the cluster may be driving the effect.

    Nonparametric cluster inference of the kind we just built is considerably more defensible than the parametric version, because it estimates the null distribution of cluster size from the data instead of assuming a functional form for the smoothness. But the interpretational limitation — inference is about the blob, not the voxel — remains.

    ### Threshold Free Cluster Enhancement
    One interesting solution to the issue of finding an initial threshold seems to be addressed by the threshold free cluster enhancement method presented in [Smith & Nichols, 2009](https://www.sciencedirect.com/science/article/pii/S1053811908002978?via%3Dihub). In this approach, the authors propose a way to combine cluster extent and voxel height into a single metric that does not require specifying a specific initial threshold. It essentially involves calculating the integral of the overall product of a signal intensity and spatial extent over multiple thresholds. It has been shown to perform particularly well when combined with non-parameteric resampling approaches such as [randomise](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise/UserGuide) in FSL. For more details about this approach check out this [blog post](http://markallenthornton.com/blog/matlab-tfce/) by Mark Thornton, this [video](https://mumfordbrainstats.tumblr.com/post/130127249926/paper-overview-threshold-free-cluster-enhancement) by Jeanette Mumford, and the original [technical report](https://www.fmrib.ox.ac.uk/datasets/techrep/tr08ss1/tr08ss1.pdf).

    ### Parametric simulations
    One approach to estimating the inherent smoothness in the data, or it's spatial autocorrelation, is using parametric simulations. This was the approach originally adopted in AFNI's AlphaSim/3DClustSim. After it was [demonstrated](https://www.pnas.org/content/113/28/7900) that real fMRI data was not adequately modeled by a standard Gaussian distribution, the AFNI group quickly updated their software and implemented a range of different algorithms in their [3DClustSim](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dClustSim.html) tool. See this [paper](https://www.biorxiv.org/content/10.1101/065862v1) for an overview of these changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Thresholding Brain Maps

    In the remainder of the tutorial, we will move from simulation to playing with real data.

    Let's watch another video by Tor Wager on how multiple comparison approaches are used in practice, highlighting some of the pitfalls with some of the different approaches.
    """)
    return


@app.cell
def _(youtube):
    youtube('N7Iittt8HrU')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will be exploring two simple and fast ways to threshold your group analyses.

    First, we will simply threshold based on selecting an arbitrary statistical threshold. The values are completely arbitrary, but it is common to start with something like p < .001. We call this *uncorrected* because this is simply the threshold for any voxel as we are not controlling for multiple tests.
    """)
    return


@app.cell
def _(BrainData, localizer, threshold):
    con1_name = 'horizontal_checkerboard'
    con1_file_list = [localizer.get_file(sub, 'betas', con1_name) for sub in localizer.get_subjects()]
    con1_dat = BrainData(con1_file_list)

    # Estimating and thresholding are now two separate steps: .ttest() returns
    # the unthresholded maps, and `threshold` masks the t-map by the p-map.
    con1_stats = con1_dat.ttest()
    threshold(con1_stats['t'], con1_stats['p'], thr=0.001).iplot()
    return con1_dat, con1_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see some significant activations in visual cortex, but we also see strong t-tests in the auditory cortex.

    Why do you think this is?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because thresholding is now its own step, swapping in FDR correction just means computing a different cutoff. `fdr()` returns the p-value threshold that controls the false discovery rate at q, and we apply that cutoff to the p-map to mask the t-map.

    One thing to watch: if nothing survives, `fdr()` returns `-1`. Branch explicitly on that sentinel and keep **p ≤ cutoff** for a valid cutoff, including zero. The generic `threshold()` helper uses a strict inequality, so applying the FDR mask directly preserves the boundary discovery and ties.
    """)
    return


@app.cell
def _(con1_stats, fdr, mo):
    _fdr_thr = fdr(con1_stats['p'].data, q=0.05)
    if _fdr_thr >= 0:
        print(f"FDR threshold: p <= {_fdr_thr:.5f}")
        _thresholded = con1_stats['t'].copy()
        _thresholded.data[con1_stats['p'].data > _fdr_thr] = 0
        _view = _thresholded.iplot()
    else:
        _view = mo.md('Nothing survives FDR correction for this contrast.')
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can see that at least for this particular contrast, the FDR threshold appears to be more liberal than p < 0.001 uncorrected.

    Let's look at another contrast between vertical and horizontal checkerboards.
    """)
    return


@app.cell
def _(BrainData, con1_dat, localizer, threshold):
    con2_name = 'vertical_checkerboard'
    con2_file_list = [localizer.get_file(sub, 'betas', con2_name) for sub in localizer.get_subjects()]
    con2_dat = BrainData(con2_file_list)
    con1_v_con2 = con1_dat - con2_dat

    con1_v_con2_stats = con1_v_con2.ttest()
    threshold(con1_v_con2_stats['t'], con1_v_con2_stats['p'], thr=0.001).iplot()
    return (con1_v_con2_stats,)


@app.cell
def _(con1_v_con2_stats, fdr, mo):
    _fdr_thr = fdr(con1_v_con2_stats['p'].data, q=0.05)
    if _fdr_thr >= 0:
        _thresholded = con1_v_con2_stats['t'].copy()
        _thresholded.data[con1_v_con2_stats['p'].data > _fdr_thr] = 0
        _view = _thresholded.iplot()
    else:
        _view = mo.md('Nothing survives FDR correction for this contrast.')
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Permutation-based correction on real data

    The production inference in this section comes from two libraries. In the course's `nltools` version (`0.6.0.dev2`), `threshold()` applies a p-value cutoff, `fdr()` supplies a Benjamini–Hochberg cutoff, and `holm_bonf()` supplies a Holm–Bonferroni cutoff. Plain Bonferroni uses `alpha / n_tests`. `BrainData.ttest(permutation=True)` supplies voxelwise permutation p-values, but does **not** automatically correct for multiple comparisons.

    Max-statistic FWE, cluster-extent FWE, cluster-mass FWE, and TFCE correction are supplied here by `nilearn`, rather than a built-in `nltools` correction API. `BrainData.cluster_report()` describes an already thresholded map; it does not calibrate cluster significance. The small grid helpers above are teaching examples, not a general neuroimaging correction API.

    Also keep statistic units consistent: the null returned by `nltools`' one-sample permutation test contains **means**, so our grid example compares it with the observed mean. Nilearn's max-t correction uses observed and permuted **t-statistics**.

    `nilearn.glm.second_level.non_parametric_inference` returns all of these from a single call. For a one-sample group test, the design matrix is just a column of ones — an intercept — which is the second-level equivalent of "test whether the group mean differs from zero".

    ```{warning}
    The `threshold` argument of `non_parametric_inference` is a **p-value**, not a t-value. nilearn converts it to a t-statistic internally using the model's degrees of freedom. Passing something like `threshold=2.5` thinking it is a cluster-forming t-threshold does not raise an error: internally it computes `t.isf(1.25, df)`, which is `NaN`, every `array > NaN` comparison is `False`, and you silently get back cluster maps that are **entirely zero** while the voxel-level `logp_max_t` output still looks perfectly reasonable. Always pass a p-value here.
    ```
    """)
    return


@app.cell
def _(con1_dat, mo, non_parametric_inference, np, pd, storage, storage_ready):
    group_design = pd.DataFrame({'intercept': np.ones(len(con1_dat))})
    con1_imgs = [con1_dat[_i].to_nifti() for _i in range(len(con1_dat))]

    with mo.persistent_cache("threshold_cluster", store=storage.cache_store() if storage_ready else None):
        npi_cluster = non_parametric_inference(
            con1_imgs,
            design_matrix=group_design,
            second_level_contrast='intercept',
            mask=con1_dat.mask,
            n_perm=500,
            threshold=0.001,        # p-scale cluster-forming threshold, NOT a t-value
            tfce=False,
            two_sided_test=True,
            n_jobs=-1,
            random_state=0,
        )
    print(list(npi_cluster.keys()))
    return con1_imgs, group_design, npi_cluster


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The outputs pair up: `t` is the observed statistic map, and each `logp_max_*` map holds $-\log_{10}(p)$ for a different FWE-corrected procedure.

    | output | correction |
    |---|---|
    | `logp_max_t` | voxel-level FWE (the max-statistic procedure we built above) |
    | `logp_max_size` | cluster-extent FWE |
    | `logp_max_mass` | cluster-mass FWE (extent weighted by height) |

    Because these are $-\log_{10}$ p-values, "survives at p < .05" means the value exceeds $-\log_{10}(0.05) \approx 1.30$. Let's count survivors under each.
    """)
    return


@app.cell
def _(np, npi_cluster, pd):
    _alpha_logp = -np.log10(0.05)
    correction_comparison = pd.DataFrame([
        {
            'correction': _label,
            'voxels surviving p < .05': int(np.sum(npi_cluster[_key].get_fdata() > _alpha_logp)),
        }
        for _key, _label in [
            ('logp_max_t', 'voxel-level FWE (max-t)'),
            ('logp_max_size', 'cluster-extent FWE'),
            ('logp_max_mass', 'cluster-mass FWE'),
        ]
    ])
    correction_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ordering is the point. Voxel-level FWE is the strictest — it only keeps voxels whose individual height is extreme enough to beat the map maximum. The cluster-based procedures keep far more, because they are willing to credit a voxel of moderate height that happens to sit inside a large blob.

    That extra sensitivity is not free. Remember what a cluster-level result licenses you to say: *this blob contains signal somewhere*. It does not license any claim about a particular voxel inside it, even though the picture strongly invites one.
    """)
    return


@app.cell
def _(np, npi_cluster, plot_stat_map, plt):
    _alpha_logp = -np.log10(0.05)
    _f, _a = plt.subplots(nrows=3, figsize=(14, 11))
    for _ax, (_key, _label) in zip(_a, [
        ('logp_max_t', 'voxel-level FWE (max-t)'),
        ('logp_max_size', 'cluster-extent FWE'),
        ('logp_max_mass', 'cluster-mass FWE'),
    ]):
        plot_stat_map(
            npi_cluster[_key], threshold=_alpha_logp, display_mode='z',
            cut_coords=[-24, -12, 0, 12, 24, 36], axes=_ax, title=f'{_label}  (-log10 p)',
            cmap='viridis', symmetric_cbar=False, vmax=-np.log10(1 / 501), colorbar=True,
        )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TFCE, and paying attention to what things cost

    TFCE is the fourth procedure, and it is requested with `tfce=True` instead of a `threshold`. It is also dramatically more expensive than the others: runtime depends strongly on mask size, software version, CPU and thread count. Earlier measurements on this dataset took seconds per permutation for a whole-brain run. The example keeps the full 500-permutation budget, uses parallel CPU workers, and caches only the numerical result; plotting stays outside the cache so figures render on cache hits. This is a teaching budget, not a convergence guarantee.

    That cost scales with the number of voxels, which gives us a practical strategy that also happens to be good statistical practice. If you have an a priori anatomical hypothesis, restrict the analysis to that volume. You test fewer voxels (so the correction is less punishing) *and* the computation becomes tractable. This is the same "small volume correction" idea from the very first simulation in this tutorial, where shrinking the grid from 100 x 100 to 5 x 5 collapsed the false positive rate.

    Our contrast is a visual checkerboard, so let's restrict to occipital cortex using the probabilistic Harvard-Oxford atlas. This cuts the search volume from ~239,000 voxels to ~32,000, and brings a 500-permutation TFCE run down to about a minute.
    """)
    return


@app.cell
def _(BrainData, con1_dat, fetch_resource, pd):
    ho_labels = pd.read_csv(fetch_resource('atlases/labels_harvard_oxford.csv'))

    # The Harvard-Oxford resource is a 4D *probabilistic* atlas: one volume per
    # region, holding the probability (0-100) that each voxel belongs to it.
    # Passing mask= resamples it into the same 2mm space as our data.
    ho_atlas = BrainData(fetch_resource('atlases/atlas_harvard_oxford.nii.gz'), mask=con1_dat.mask)

    occipital_rows = ho_labels.index[
        ho_labels.name.str.contains('Occipital|Lingual|Calcarine|Cuneal', case=False)
    ].tolist()

    occipital_mask = con1_dat[0].copy()
    occipital_mask.data = (ho_atlas[occipital_rows].data.max(axis=0) > 25).astype(float)

    print(f"{len(occipital_rows)} occipital regions -> "
          f"{int(occipital_mask.data.sum())} voxels of {con1_dat.shape[1]}")
    return (occipital_mask,)


@app.cell
def _(
    con1_imgs,
    group_design,
    mo,
    non_parametric_inference,
    occipital_mask,
    storage,
    storage_ready,
):
    with mo.persistent_cache("threshold_tfce", store=storage.cache_store() if storage_ready else None):
        npi_tfce = non_parametric_inference(
            con1_imgs,
            design_matrix=group_design,
            second_level_contrast='intercept',
            mask=occipital_mask.to_nifti(),
            n_perm=500,
            threshold=None,
            tfce=True,
            two_sided_test=True,
            n_jobs=-1,
            random_state=0,
        )
    print(list(npi_tfce.keys()))
    return (npi_tfce,)


@app.cell
def _(np, npi_tfce, plot_stat_map, plt):
    _alpha_logp = -np.log10(0.05)
    _n_survivors = int(np.sum(npi_tfce['logp_max_tfce'].get_fdata() > _alpha_logp))
    plot_stat_map(
        npi_tfce['logp_max_tfce'], threshold=_alpha_logp, display_mode='z', cut_coords=6,
        title=f'TFCE within occipital mask, {_n_survivors} voxels survive (-log10 p)',
        colorbar=True,
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reporting what survived

    Once you have a corrected map, the last step is describing it: where are the surviving clusters, how big are they, and what anatomy do they cover? `nltools` provides `cluster_report` for this, which will also label peaks against an atlas.

    ```{note}
    This step is **descriptive, not corrective**. `cluster_report(cluster_threshold=...)` drops small clusters from the *table*; it does not compute a null distribution and it does not control any error rate. The `cluster_threshold` argument here is a display convenience and is a completely different thing from the cluster-extent FWE procedure above, despite the similar name. Do the correction first, then report.
    ```
    """)
    return


@app.cell
def _(BrainData, con1_dat, np, npi_cluster):
    # Report the signed t map only where voxel-level max-t correction survives.
    _corrected = BrainData(npi_cluster['t'], mask=con1_dat.mask)
    _logp = BrainData(npi_cluster['logp_max_t'], mask=con1_dat.mask)
    _corrected.data[_logp.data <= -np.log10(0.05)] = 0
    cluster_rep = _corrected.cluster_report(
        stat_threshold=None,
        cluster_threshold=1,
        atlas='harvard_oxford',
    )
    cluster_rep.clusters
    return (cluster_rep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `.clusters` gives one row per cluster with its peak coordinate, size, and atlas composition; `.peaks` gives the individual local maxima within each cluster. Both are Polars dataframes, and `.to_csv()` will write them out for a paper's table.
    """)
    return


@app.cell
def _(cluster_rep):
    cluster_rep.peaks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This concludes our overview of thresholding and multiple comparisons correction in fMRI data analysis.

    The practical summary: uncorrected thresholds do not control anything at the level of the map. Bonferroni and FDR operate on valid p-values, whether parametric or nonparametric; permutation resolution must be fine enough for the requested cutoff. Bonferroni can be conservative when tests are strongly dependent. Permutation tests require a valid resampling scheme (independent subjects and symmetric null errors for sign-flipping) and give you the null distribution itself, from which max-statistic, cluster, and TFCE corrections all follow. And whichever you choose, be clear about what the result licenses you to claim — a cluster-level result is a statement about a blob, not about a voxel.

    We will continue to add more advanced tutorials to the dartbrains.org website. Stay tuned!
    """)
    return


@app.cell(hide_code=True)
def _():
    from dartbrains_tools.notebook_utils import assignment_card

    assignment_card("thresholding")
    return


if __name__ == "__main__":
    app.run()
