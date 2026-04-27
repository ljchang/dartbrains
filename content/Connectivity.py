import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from pathlib import Path
    _ROOT = next(p for p in (Path.cwd(), *Path.cwd().resolve().parents) if (p / "book.yml").exists())
    IMG_DIR = _ROOT / "images" / "connectivity"
    return IMG_DIR, mo


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
    # Connectivity
    *Written by Luke Chang*

    So far, we have primarily been focusing on analyses related to task evoked brain activity. However, an entirely different way to study the brain is to characterize how it is intrinsically connected. There are many different ways to study functional connectivity.

    The primary division is studying how brain regions are *structurally* connected. In animal studies this might involve directly tracing bundles of neurons that are connected to other neurons. Diffusion imaging is a common way in which we can map how bundles of white matter are connected to each region, based on the direction in which water diffuses along white matter tracks. There are many different techniques such as fractional ansiotropy and probablistic tractography. We will not be discussing structural connectivity in this course.

    An alternative approach to studying connectivity is to examine how brain regions covary with each other in time. This is referred to as *functional connectivity*, but it is better to think about it as temporal covariation between regions as this does not necessarily imply that two regions are directly communication with each other.
    """),
        mo.image(str(IMG_DIR / "mediation.png")),
        mo.md(r"""
    For example, regions can *directly* influence each other, or they can *indirectly* influence each other via a mediating region, or they can be affected similarly by a *shared influence*. These types of figures are often called *graphs*. These types of *graphical* models can be *directed* or *undirected*. Directed graphs imply a causal relationship, where one region A directly influence another region B. Directed graphs or *causal models* are typically described as *effective connectivity*, while undirected graphs in which the relationship is presumed to be bidirectional are what we typically describe as *functional connectivity*.

    In this tutorial, we will work through examples on:
     - Seed-based functional connectivity
     - Psychophysiological interactions
     - Principal Components Analysis
     - Graph Theory

    Let's start by watching a short overview of connectivity by Martin Lindquist.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(youtube):
    youtube('J0KX_rW0hmc')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's dive in a little bit deeper into the specific details of functional connectivity.
    """)
    return


@app.cell
def _(youtube):
    youtube('OVAQujut_1o')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functional Connectivity
    ### Seed Voxel Correlations

    One relatively simple way to calculate functional connectivity is to compute the temporal correlation between two regions of interest (ROIs). Typically, this is done by extracting the temporal response from a *seed voxel* or the average response within a *seed region*. Then this time course is regressed against all other voxels in the brain to produce a whole brain map of anywhere that shares a similar time course to the seed.

    Let's try it ourselves with an example subject from the Pinel Localizer dataset. First, let's import the modules we need for this tutorial and set our paths.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.fft import fft, fftfreq
    from nltools.data import Brain_Data, Design_Matrix, Adjacency
    from nltools.mask import expand_mask, roi_to_brain
    from nltools.stats import zscore, fdr, one_sample_permutation
    from nltools.file_reader import onsets_to_dm
    from scipy.stats import binom, ttest_1samp
    from sklearn.metrics import pairwise_distances
    from copy import deepcopy
    import networkx as nx
    from nilearn.plotting import plot_stat_map, view_img_on_surf
    import nibabel as nib
    from dartbrains_tools.data import get_file, get_tr, load_events, load_confounds, get_subjects
    from dartbrains_tools.notebook_utils import youtube


    def get_csf_mask_path(subject):
        """Per-subject CSF probability mask from fmriprep outputs."""
        from pathlib import Path as _Path
        bold_path = _Path(get_file(subject, 'derivatives', 'bold'))
        return str(bold_path.parent.parent / 'anat' /
                   f'sub-{subject}_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz')

    return (
        Adjacency,
        Brain_Data,
        Design_Matrix,
        expand_mask,
        fft,
        fftfreq,
        get_csf_mask_path,
        get_file,
        get_tr,
        go,
        load_confounds,
        load_events,
        make_subplots,
        nib,
        np,
        nx,
        pairwise_distances,
        pd,
        plt,
        roi_to_brain,
        view_img_on_surf,
        youtube,
        zscore,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's load an example participant's preprocessed functional data.
    """)
    return


@app.cell
def _(Brain_Data, get_file, mo):
    sub = 'S01'
    _fwhm = 6
    data = Brain_Data(get_file(sub, 'derivatives', 'bold'))

    with mo.persistent_cache(name="connectivity_smoothed"):
        smoothed = data.smooth(fwhm=_fwhm)
    return data, smoothed, sub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we need to pick an ROI. Pretty much any type of ROI will work.

    In this example, we will be using a whole brain parcellation based on similar patterns of coactivation across over 10,000 published studies available in neurosynth (see this paper for more [details](http://cosanlab.com/static/papers/delaVega_2016_JNeuro.pdf)). We will be using a parcellation of 50 different functionally similar ROIs.
    """)
    return


@app.cell
def _(Brain_Data, mo):
    with mo.persistent_cache(name="connectivity_k50_mask"):
        mask_1 = Brain_Data('https://neurovault.org/media/images/8423/k50_2mm.nii.gz')
    mask_1.iplot()
    return (mask_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each ROI in this parcellation has its own unique number. We can expand this so that each ROI becomes its own binary mask using `nltools.mask.expand_mask`.

    Let's plot the first 5 masks.
    """)
    return


@app.cell
def _(expand_mask, mask_1):
    mask_x = expand_mask(mask_1)
    mask_x[0:5].iplot()
    return (mask_x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To use any mask we just need to index it by the correct label.

    Let's start by using the vmPFC mask (ROI=32) to use as a seed in a functional connectivity analysis.
    """)
    return


@app.cell
def _(go, mask_x, smoothed):
    vmpfc = smoothed.extract_roi(mask=mask_x[32])

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        y=vmpfc, mode='lines', name='vmPFC',
        line=dict(width=2),
        hovertemplate='TR=%{x}<br>intensity=%{y:.2f}<extra></extra>',
    ))
    _fig.update_layout(
        xaxis_title='Time (TRs)', yaxis_title='Mean Intensity',
        height=350, hovermode='x unified',
        margin=dict(l=60, r=20, t=20, b=50),
    )
    _fig
    return (vmpfc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Okay, now let's build our regression design matrix to perform the whole-brain functional connectivity analysis.

    The goal is to find which regions in the brain have a similar time course to the vmPFC, controlling for all of our covariates (i.e., nuisance regressors).

    Functional connectivity analyses are particularly sensitive to artifacts that might induce a temporal relationship, particularly head motion (See this [article](https://www.sciencedirect.com/science/article/pii/S1053811911011815) by Jonathan Power for more details). This means that we will need to use slightly different steps to preprocess data for this type of analyis then a typical event related mass univariate analysis.

    We are going to remove the mean from our vmPFC signal. We are also going to include the average activity in CSF as an additional nuisance regressor to remove physiological artifacts. Finally, we will be including our 24 motion covariates as well as linear and quadratic trends. We need to be a little careful about filtering as the normal high pass filter for an event related design might be too short and will remove potential signals of interest.

    Resting state researchers also often remove the global signal, which can reduce physiological and motion related artifacts and also increase the likelihood of observing negative relationships with your seed regressor (i.e., anticorrelated). This procedure has remained quite controversial in practice (see [here](https://www.physiology.org/doi/full/10.1152/jn.90777.2008) [here](https://www.sciencedirect.com/science/article/pii/S1053811908010264), [here](https://www.pnas.org/content/107/22/10238.short), and [here](https://www.sciencedirect.com/science/article/pii/S1053811916306711) for a more in depth discussion). We think that in general including covariates like CSF should be sufficient. It is also common to additionally include covariates from white matter masks, and also multiple principal components of this signal rather than just the mean (see more details about [compcorr](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2214855/).

    Overall, this code should seem very familiar as it is pretty much the same procedure we used in the single subject GLM tutorial. However, instead of modeling the task design, we are interested in calculating the functional connectivity with the vmPFC.
    """)
    return


@app.cell
def _(
    Brain_Data,
    Design_Matrix,
    data,
    get_csf_mask_path,
    get_tr,
    load_confounds,
    pd,
    smoothed,
    sub,
    vmpfc,
    zscore,
):
    tr = get_tr()
    _fwhm = 6
    n_tr = len(data)


    def make_motion_covariates(mc, tr):
        z_mc = zscore(mc)
        parts = {
            '': z_mc,
            '_sq': z_mc ** 2,
            '_diff': z_mc.diff(),
            '_diff_sq': z_mc.diff() ** 2,
        }
        all_mc = pd.concat(
            [df.rename(columns=lambda c: f'{c}{suffix}') for suffix, df in parts.items()],
            axis=1,
        )
        all_mc.fillna(value=0, inplace=True)
        return Design_Matrix(all_mc, sampling_freq=1 / tr)


    vmpfc_1 = zscore(pd.DataFrame(vmpfc, columns=['vmpfc']))
    _csf_mask = Brain_Data(get_csf_mask_path(sub)).threshold(upper=0.7, binarize=True)
    csf = zscore(pd.DataFrame(smoothed.extract_roi(mask=_csf_mask).T, columns=['csf']))
    spikes = smoothed.find_spikes(global_spike_cutoff=3, diff_spike_cutoff=3)
    _covariates = load_confounds(sub)
    _mc = _covariates[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
    mc_cov = make_motion_covariates(_mc, tr)
    dm = Design_Matrix(pd.concat([vmpfc_1, csf, mc_cov, spikes.drop(labels='TR', axis=1)], axis=1), sampling_freq=1 / tr)
    dm = dm.add_poly(order=2, include_lower=True)
    dm.convolved = ['vmpfc']

    smoothed.X = dm
    _stats_vmpfc = smoothed.regress()
    vmpfc_conn = _stats_vmpfc['beta']
    return csf, make_motion_covariates, mc_cov, spikes, tr, vmpfc_1, vmpfc_conn


@app.cell
def _(vmpfc_conn):
    vmpfc_conn.threshold(upper=25, lower=-25).iplot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how this analysis identifies the default network? This analysis is very similar to the [original papers](https://www.pnas.org/content/102/27/9673/) that identified the default mode network using resting state data.

    For an actual analysis, we would need to repeat this procedure over all of the participants in our sample and then perform a second level group analysis to identify which voxels are consistently coactive with the vmPFC. We will explore group level analyses in the exercises.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Psychophysiological Interactions

    Suppose we were interested in seeing if the vmPFC was connected to other regions differently when performing a finger tapping task compared to all other conditions. To compute this analysis, we will need to create a new design matrix that combines the motor regressors and then calculates an interaction term between the seed region activity (e.g., vmpfc) and the condition of interest (e.g., motor).

    This type of analysis called, *psychophysiological* interactions was originally [proposed](https://www.fil.ion.ucl.ac.uk/spm/doc/papers/karl_ppi.pdf) by Friston et al., 1997. For a more hands on and practical discussion read this [paper](https://pdfs.semanticscholar.org/dd86/1acdb332ea7fa9de8fb677a4048651eaea02.pdf) and watch this [video](https://www.youtube.com/watch?v=L3iBhfEYEgE) by Jeanette Mumford and a follow up [video](https://www.youtube.com/watch?v=M8APlF6oBgA)) of a more generalized method.
    """)
    return


@app.cell
def _(get_file, nib):
    _img = nib.load(get_file('S01', 'derivatives', 'bold'))
    print(f'shape: {_img.shape}')
    print(f'voxel size: {_img.header.get_zooms()}')
    print(f'TR: {_img.header.get_zooms()[3]} s')
    _img
    return


@app.cell
def _(
    Design_Matrix,
    csf,
    get_file,
    get_tr,
    load_events,
    mc_cov,
    nib,
    np,
    pd,
    plt,
    spikes,
    tr,
    vmpfc_1,
):
    def load_bids_events(subject):
        """Create a Design_Matrix from BIDS event file.

        Bypasses nltools.onsets_to_dm because that wrapper has a column-name
        mangling bug when TR is supplied. Calls nilearn directly with
        hrf_model='glover', then resets the index so the result can be
        horizontally concatenated with other 0..n_tr-indexed DataFrames.
        """
        from nilearn.glm.first_level import make_first_level_design_matrix as _make_dm
        tr = get_tr()
        n_tr = nib.load(get_file(subject, 'derivatives', 'bold')).shape[-1]
        onsets = load_events(subject)
        frame_times = np.arange(n_tr) * tr
        dm_raw = _make_dm(frame_times, events=onsets, hrf_model='glover', drift_model=None)
        convolved = [c for c in dm_raw.columns if c != 'constant']
        return Design_Matrix(dm_raw, convolved=convolved, sampling_freq=1 / tr,
                             polys=['constant']).reset_index(drop=True)


    dm_1 = load_bids_events('S01')
    motor_variables = ['video_left_hand', 'audio_left_hand', 'video_right_hand', 'audio_right_hand']
    ppi_dm = dm_1.drop(motor_variables, axis=1)
    ppi_dm['motor'] = pd.Series(dm_1.loc[:, motor_variables].sum(axis=1))
    ppi_dm_conv = ppi_dm.convolve()
    ppi_dm_conv['vmpfc'] = vmpfc_1.values
    ppi_dm_conv['vmpfc_motor'] = ppi_dm_conv['vmpfc'] * ppi_dm_conv['motor_c0']
    dm_1 = Design_Matrix(pd.concat([ppi_dm_conv, csf, mc_cov, spikes.drop(labels='TR', axis=1)], axis=1), sampling_freq=1 / tr)
    dm_1 = dm_1.add_poly(order=2, include_lower=True)
    dm_1.convolved = list(ppi_dm_conv.columns)
    dm_1.heatmap()
    plt.clf()
    return (dm_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Okay, now we are ready to run the regression analysis and inspect the interaction term to find regions where the connectivity profile changes as a function of the motor task.

    We will run the regression and smooth all of the images, and then examine the beta image for the PPI interaction term.
    """)
    return


@app.cell
def _(dm_1, np, smoothed):
    smoothed.X = dm_1
    ppi_stats = smoothed.regress()
    vmpfc_motor_ppi = ppi_stats['beta'][int(np.where(smoothed.X.columns == 'vmpfc_motor')[0][0])]
    vmpfc_motor_ppi.iplot()
    return (vmpfc_motor_ppi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This analysis tells us which regions are more functionally connected with the vmPFC during the motor conditions relative to the rest of experiment.

    We can make a thresholded interactive plot to interrogate these results, but it looks like it identifies the ACC/pre-SMA.
    """)
    return


@app.cell
def _(vmpfc_motor_ppi):
    vmpfc_motor_ppi.iplot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dynamic Connectivity

    All of the methods we have discussed so far assume that the relationship between two regions is stationary - or remains constant over the entire dataset. However, it is possible that voxels are connected to other voxels at specific points in time, but then change how they are connected when they are computing a different function or in different psychological state.

    Time-varying connectivity is beyond the scope of the current tutorial, but we encourage you to watch this [video](https://www.youtube.com/watch?v=lV9thGD18JI&list=PLfXA4opIOVrEFBitfTGRPppQmgKp7OO-F&index=22&t=0s) from Principles of fMRI for more details
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effective Connectivity
    Effective connectivity refers to the degree that one brain region has a directed influence on another region. This approach requires making a number of assumptions about the data and requires testing how well a particular model describes the data. Typically, most researchers will create a model of a small number of nodes and compare different models to each other. This is because the overall model fit is typically in itself uninterpretable and because formulating large models can be quite difficult and computationally expensive. The number of connections can be calculated as:

    $connections = \frac{n(n-1)}{2}$, where $n$ is the total number of nodes.

    Let's watch a short video by Martin Lindquist that provides an overview to different approaches to effectivity connectivity.
    """)
    return


@app.cell
def _(youtube):
    youtube('gv5ENgW0bbs')
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
    ### Structural Equation Modeling

    Structural equation modeling (SEM) is one early technique that was used to model the causal relationship between multiple nodes. SEM requires specifying a causal relationship between nodes in terms of a set of linear equations. The parameters of this system of equations reflects the connectivity matrix. Users are expected to formulate their own hypothesized relationship between variables with a value of one when there is an expected relationship, and zero when there is no relationship. Then we estimate the parameters of the model and evaluate how well the model describes the observed data.
    """),
        mo.image(str(IMG_DIR / "sem.png")),
        mo.md(r"""
    We will not be discussing this method in much detail. In practice, this method is more routinely used to examine how brain activations mediate relationships between other regions, or between different psychological constructs (e.g., X -> Z -> Y).

    Here are a couple of videos specifically examining how to conduct mediation and moderation analyses from Principles of fMRI ([Mediation and Moderation Part I](https://www.youtube.com/watch?v=0YqWXIfpu20),
    [Mediation and Moderation Part II](https://www.youtube.com/watch?v=0YqWXIfpu20))
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Granger Causality
    Granger causality was originally developed in econometrics and is used to determine temporal causality. The idea is to quantify how past values of one brain region predict the current value of another brain region. This analysis can also be performed in the frequency domain using measures of coherence between two regions. In general, this technique is rarely used in fMRI data analysis as it requires making assumptions that all regions have the same hemodynamic response function (which does not seem to be true), and that the relationship is stationary, or not varying over time.

    Here is a [video](https://www.youtube.com/watch?v=yE9aBHQ7bnA) from Principles of fMRI explaining Granger Causality in more detail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dynamic Causal Modeling

    Dynamic Causal Modeling (DCM) is a method specifically developed for conducting causal analyses between regions of the brain for fMRI data. The key innovation is that the developers of this method have specified a generative model for how neuronal firing will be reflected in observed BOLD activity. This addresess one of the problems with SEM, which assumes that each ROI has the same hemodynamic response.

    In practice, DCM is computationally expensive to estimate and researchers typically specify a couple small models and perform a model comparison (e.g., bayesian model comparison) to determine, which model best explains the data from a set of proposed models.

    Here is a [video](https://www.youtube.com/watch?v=JoJKoq5gmH8) from Principles of fMRI explaining Dynamic Causal Modeling in more detail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (multivariate-decomposition)=
    ## Multivariate Decomposition

    So far we have discussed functional connectivity in terms of pairs of regions. However, voxels are most likely not independent from each other and we may want to figure out some latent spatial components that are all functionally connected with each (i.e., covary similarly in time).

    To do this type of analysis, we typically use what are called *multivariate decomposition* methods, which attempt to factorize a data set (i.e., time by voxels) into a lower dimensional set of components, where each has their own unique time course.

    The most common decomposition methods or Principal Components Analysis (PCA) and Independent Components Analysis (ICA). You may recall that we have already played with ICA in the beginning of the course in the [ICA](ICA.md) lab.

    Let's learn more about the details of decomposition by watching a short video by Martin Lindquist.
    """)
    return


@app.cell
def _(youtube):
    youtube('Klp-8t5GLEg')
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
    ### Principal Components Analysis
    Principal Components Analysis (PCA) is a multivariate procedure that attempts to explain the variance-covariance structure of a high dimensional random vector. In this procedure, a set of correlated variables are transformed int a set of uncorrelated variables, ordered by the amount of variance in the data that they explain.

    In fMRI, we use PCA to find spatial maps or *eigenimages* in the data. This is usually computed using Singular Value Decomposition (SVD). This operation is defined as:

    $X = USV^T$, where $V^T V = I$, $U^T U = I$, and $S$ is a diagonal matrix whose elements are called singular values.

    In practice, $V$ corresponds to the eigenimages or spatial components and $U$ corresponds to the transformation matrix to convert the eigenimages into a timecourse. $S$ reflects the amount of scaling for each component.
    """),
        mo.image(str(IMG_DIR / "svd.png")),
        mo.md(r"""
    SVD is conceptually very similar to regression. We are trying to explain a matrix $X$ as a linear combination of components. Each term in the equation reflects a unique (i.e., orthogonal) multivariate signal present in $X$. For example, the $nth$ signal in X can be described by the dot product of a time course $u_n$ and the spatial map $Vn^T$  scaled by $s_n$.

    $X = s_1 u_1 v_1^T + s_2 u_2 v_2^T + s_n u_n v_n^T$

    Let's try running a PCA on our single subject data.

    First, let's denoise our data using a GLM comprised only of nuisance regressors. We will then work with the *residual* of this model, or what remains of our data that was not explained by the denoising model. This is essentially identical to the vmPFC analysis, except that we will not be including any seed regressors. We will then be working with the residual of our regression, which is the remaining signal after removing any variance associated with our covariates.
    """),
    ])
    return


@app.cell
def _(
    Brain_Data,
    Design_Matrix,
    get_csf_mask_path,
    load_confounds,
    make_motion_covariates,
    pd,
    smoothed,
    sub,
    tr,
    vmpfc_1,
    zscore,
):
    _csf_mask = Brain_Data(get_csf_mask_path(sub)).threshold(upper=0.7, binarize=True)
    csf_1 = zscore(pd.DataFrame(smoothed.extract_roi(mask=_csf_mask).T, columns=['csf']))
    spikes_1 = smoothed.find_spikes(global_spike_cutoff=3, diff_spike_cutoff=3)
    _covariates = load_confounds(sub)
    _mc = _covariates[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
    mc_cov_1 = make_motion_covariates(_mc, tr)
    dm_2 = Design_Matrix(pd.concat([vmpfc_1, csf_1, mc_cov_1, spikes_1.drop(labels='TR', axis=1)], axis=1), sampling_freq=1 / tr)
    dm_2 = dm_2.add_poly(order=2, include_lower=True)
    dm_2.convolved = ['vmpfc']

    smoothed.X = dm_2
    _stats_denoise = smoothed.regress()
    smoothed_denoised = _stats_denoise['residual']
    return (smoothed_denoised,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's run a PCA on this participant's denoised data. To do this, we will use the `.decompose()` method from nltools. All we need to do is specify the algorithm we want to use, the dimension we want to reduce (i.e., time - 'images' or space 'voxels'), and the number of components to estimate. Usually, we will be looking at reducing space based on similarity in time, so we will set `axis='images'`.
    """)
    return


@app.cell
def _(smoothed_denoised):
    n_components = 10

    pca_stats_output = smoothed_denoised.decompose(algorithm='pca', axis='images', n_components=n_components)
    return (pca_stats_output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's inspect the components with our interactive component viewer. Remember the ICA tutorial? Hopefully, you are now better able to understand everything.
    """)
    return


@app.cell
def _(mo, pca_stats_output):
    component_slider = mo.ui.slider(
        start=0, stop=len(pca_stats_output['components']) - 1, value=0, step=1,
        label='Component', show_value=True, full_width=True,
    )
    threshold_slider = mo.ui.slider(
        start=0.0, stop=4.0, value=2.0, step=0.1,
        label='Threshold (z)', show_value=True, full_width=True,
    )
    mo.hstack([component_slider, threshold_slider], justify='start')
    return component_slider, threshold_slider


@app.cell
def _(
    component_slider,
    fft,
    fftfreq,
    get_tr,
    go,
    make_subplots,
    mo,
    np,
    pca_stats_output,
    threshold_slider,
):
    _comp = component_slider.value
    _thresh = threshold_slider.value

    # z-score and threshold the selected component map
    _component = pca_stats_output['components'][_comp]
    _zscored = (_component - _component.mean()) * (1 / _component.std())
    _zscored.data = np.where(np.abs(_zscored.data) <= _thresh, 0, _zscored.data)

    # Title with variance explained (if PCA)
    import sklearn.decomposition as _skd
    _decomp = pca_stats_output['decomposition_object']
    if isinstance(_decomp, _skd.PCA):
        _var = _decomp.explained_variance_ratio_[_comp]
        _title = f'Component {_comp}/{len(pca_stats_output["components"])} — Variance Explained: {_var:.2%}'
    else:
        _title = f'Component {_comp}/{len(pca_stats_output["components"])}'

    # Timecourse + power spectrum (plotly subplots)
    _weights = pca_stats_output['weights'][:, _comp]
    _y = fft(_weights)
    _freqs = fftfreq(len(_y), d=get_tr())
    _pos_mask = _freqs > 0

    _fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.18,
        subplot_titles=(f'Timecourse (TR={get_tr()}s)', 'Power Spectrum'),
    )
    _fig.add_trace(go.Scatter(
        y=_weights, mode='lines', line=dict(color='red', width=2),
        hovertemplate='TR=%{x}<br>intensity=%{y:.3f}<extra></extra>',
        name='timecourse',
    ), row=1, col=1)
    _fig.add_trace(go.Scatter(
        x=_freqs[_pos_mask], y=np.abs(_y)[_pos_mask] ** 2,
        mode='lines', line=dict(width=2),
        hovertemplate='f=%{x:.3f} Hz<br>power=%{y:.2f}<extra></extra>',
        name='power',
    ), row=2, col=1)
    _fig.update_xaxes(title_text='Time (TRs)', row=1, col=1)
    _fig.update_yaxes(title_text='Intensity (AU)', row=1, col=1)
    _fig.update_xaxes(title_text='Frequency (Hz)', row=2, col=1)
    _fig.update_yaxes(title_text='Power', row=2, col=1)
    _fig.update_layout(
        title=_title, height=600, showlegend=False,
        margin=dict(l=60, r=20, t=80, b=50),
    )

    mo.vstack([_zscored.iplot(), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also examine the eigenvalues/singular values or scaling factor of each, which are the diagonals of $S$.

    These values are stored in the `'decomposition_object'` of the stats_output and are in the variable called `.singular_values_`.
    """)
    return


@app.cell
def _(go, np, pca_stats_output):
    _sv = pca_stats_output['decomposition_object'].singular_values_
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=np.arange(len(_sv)), y=_sv, mode='lines+markers',
        line=dict(width=2),
        hovertemplate='component %{x}<br>singular value=%{y:.3f}<extra></extra>',
    ))
    _fig.update_layout(
        xaxis_title='Component', yaxis_title='Singular Value',
        height=350, margin=dict(l=60, r=20, t=20, b=50),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use these values to calculate the overall variance explained by each component. These values are stored in the `'decomposition_object'` of the stats_output and are in the variable called `.explained_variance_ratio_`.

    These values can be used to create what is called a *scree* plot to figure out the percent variance of $X$ explained by each component. Remember, in PCA, components are ordered by descending variance explained.
    """)
    return


@app.cell
def _(go, make_subplots, np, pca_stats_output):
    _var = pca_stats_output['decomposition_object'].explained_variance_ratio_
    _cum = np.cumsum(_var)
    _fig = make_subplots(rows=1, cols=2, subplot_titles=('Variance Explained', 'Cumulative Variance Explained'))
    _fig.add_trace(go.Scatter(x=np.arange(len(_var)), y=_var, mode='lines+markers',
                              name='per-component', line=dict(width=2),
                              hovertemplate='component %{x}<br>variance=%{y:.3%}<extra></extra>'),
                   row=1, col=1)
    _fig.add_trace(go.Scatter(x=np.arange(len(_cum)), y=_cum, mode='lines+markers',
                              name='cumulative', line=dict(width=2),
                              hovertemplate='component %{x}<br>cumulative=%{y:.3%}<extra></extra>'),
                   row=1, col=2)
    _fig.update_xaxes(title_text='Component', row=1, col=1)
    _fig.update_xaxes(title_text='Component', row=1, col=2)
    _fig.update_yaxes(title_text='Percent Variance Explained', row=1, col=1, tickformat='.0%')
    _fig.update_yaxes(title_text='Percent Variance Explained', row=1, col=2, tickformat='.0%')
    _fig.update_layout(height=400, showlegend=False, margin=dict(l=60, r=20, t=50, b=50))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Independent Components Analysis

    Independent Components Analysis (ICA) is a method to blindly separate a source signal into spatially independent components. This approach assumes that the data consts of $p$ spatially independent components, which are linearly mixed and spatially fixed. PCA assumes orthonormality constraint, while ICA only assumes independence.

    $X = AS$, where $A$ is the *mixing matrix* and $S$ is the *source matrix*

    In ICA we find an un-mixing matrix $W$, such that $Y = WX$ provides an approximation to $S$. To estimate the mixing matrix, ICA assumes that the sources are (1) linearly mixed, (2) the components are statistically independent, and (3) the components are non-Gaussian.

    It is trivial to run ICA on our data as it only requires switching `algorithm='pca'` to `algorithm='ica'` when using the `decompose()` method.

    We will experiment with this in our exercises.

    We also encourage interested readers to check out our more in depth [ICA tutorial](Introduction_to_ICA.md).
    """)
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
    ## Graph Theory

    Similar to describing the structure of social networks, graph theory has also been used to characterize regions of the brain based on how they connected to other regions. Nodes in the network typically describe specific brain regions and edges represent the strength of the association between each edge. That is, the network can be represented as a graph of pairwise relationships between each region of the brain.

    There are many different metrics of graphs that can be used to describe the overall efficiency of a network (e.g., small worldness), or how connected a region is to other regions (e.g., degree, centrality), or how long it would take to send information from one node to another node (e.g., path length, connectivity).
    """),
        mo.image(str(IMG_DIR / "graph.png")),
        mo.md(r"""
    Let's watch a short video by Martin Lindquist providing a more in depth introduction to graph theory.
    """),
    ])
    return


@app.cell
def _(youtube):
    youtube('v8ls5VED1ng')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Suppose, we were interested in identifying which regions of the brain had the highest degree of centrality based on functional connectivity. There are many different ways to do this, but they all involve specifying a set of nodes (i.e., ROIs) and calculating the edges between each node. Finally, we would need to pick a centrality metric and calculate the overall level of centrality for each region.

    Let's do this quickly building off of our seed-based functional connectivity analysis.

    Similar, to the PCA example, let's work with the denoised data. First, let's extract the average time course within each ROI from our 50 parcels and plot the results.
    """)
    return


@app.cell
def _(go, mask_1, smoothed_denoised):
    rois = smoothed_denoised.extract_roi(mask=mask_1)
    _fig = go.Figure()
    for i in range(rois.shape[0]):
        _fig.add_trace(go.Scatter(
            y=rois[i], mode='lines', name=f'ROI {i}', line=dict(width=1),
            hovertemplate=f'ROI {i}<br>TR=%{{x}}<br>intensity=%{{y:.2f}}<extra></extra>',
        ))
    _fig.update_layout(
        xaxis_title='Time (TRs)', yaxis_title='Mean Intensity',
        height=400, showlegend=False, hovermode='closest',
        margin=dict(l=60, r=20, t=20, b=50),
    )
    _fig
    return (rois,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have specified our 50 nodes, we need to calculate the edges of the graph. We will be using pearson correlations. We will be using the `pairwise_distances` function from scikit-learn as it is much faster than most other correlation measures. We will then convert the distance metric into similarities by subtracting all of the values from 1.

    Let's visualize the resulting correlation matrix as a heatmap using seaborn.
    """)
    return


@app.cell
def _(go, pairwise_distances, rois):
    roi_corr = 1 - pairwise_distances(rois, metric='correlation')

    _fig = go.Figure(go.Heatmap(
        z=roi_corr, zmin=-1, zmax=1, colorscale='RdBu_r', reversescale=True,
        hovertemplate='ROI %{x} → %{y}<br>r=%{z:.3f}<extra></extra>',
    ))
    _fig.update_layout(
        title='ROI-to-ROI correlation', height=600, width=600,
        yaxis=dict(autorange='reversed', scaleanchor='x'),
        margin=dict(l=60, r=20, t=50, b=50),
    )
    _fig
    return (roi_corr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we need to convert this correlation matrix into a graph and calculate a centrality measure. We will use the `Adjacency` class from nltools as it has many functions that are useful for working with this type of data, including casting these type of matrices into networkx graph objects.

    We will be using the [networkx](https://networkx.github.io/documentation/stable/) python toolbox to work with graphs and compute different metrics of the graph.

    Let's calculate degree centrality, which is the total number of nodes each node is connected with. Unfortunately, many graph theory metrics require working with adjacency matrices, which are binary matrices indicating the presence of an edge or not. To create this, we will simply apply an arbitrary threshold to our correlation matrix.
    """)
    return


@app.cell
def _(Adjacency, roi_corr):
    _a = Adjacency(roi_corr, matrix_type='similarity', labels=[x for x in range(50)])
    a_thresholded = _a.threshold(upper=0.6, binarize=True)
    a_thresholded.plot()
    return (a_thresholded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Okay, now that we have a thresholded binary matrix, let's cast our data into a networkx object and calculate the degree centrality of each ROI and make a quick plot of the graph.
    """)
    return


@app.cell
def _(a_thresholded, go, nx):
    G = a_thresholded.to_graph()
    _pos = nx.kamada_kawai_layout(G)
    _node_degree = dict(G.degree())

    # Edges: flatten to (x0, x1, None, x0, x1, None, ...) for a single Scatter trace
    _edge_x = []
    _edge_y = []
    for u, v in G.edges():
        x0, y0 = _pos[u]
        x1, y1 = _pos[v]
        _edge_x.extend([x0, x1, None])
        _edge_y.extend([y0, y1, None])

    # Nodes
    _node_x = [_pos[n][0] for n in G.nodes()]
    _node_y = [_pos[n][1] for n in G.nodes()]
    _node_text = [f'ROI {n}<br>degree={_node_degree[n]}' for n in G.nodes()]
    _node_labels = [str(n) for n in G.nodes()]
    _node_sizes = [max(_node_degree[n] * 4, 8) for n in G.nodes()]
    _node_colors = [_node_degree[n] for n in G.nodes()]

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_edge_x, y=_edge_y, mode='lines',
        line=dict(width=1.5, color='rgba(80,80,100,0.25)'),
        hoverinfo='skip', showlegend=False,
    ))
    _fig.add_trace(go.Scatter(
        x=_node_x, y=_node_y, mode='markers+text',
        marker=dict(
            size=_node_sizes, color=_node_colors, colorscale='Reds',
            reversescale=True, line=dict(color='darkslategray', width=2),
            colorbar=dict(title='Degree', thickness=15),
        ),
        text=_node_labels, textposition='middle center',
        textfont=dict(color='darkslategray', size=10),
        hovertext=_node_text, hoverinfo='text', showlegend=False,
    ))
    _fig.update_layout(
        height=700, width=900,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
    )
    _fig
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also plot the distribution of degree using this threshold.
    """)
    return


@app.cell
def _(G, go):
    _degrees = list(dict(G.degree).values())
    _fig = go.Figure(go.Histogram(
        x=_degrees, name='degree',
        hovertemplate='degree=%{x}<br>count=%{y}<extra></extra>',
    ))
    _fig.update_layout(
        xaxis_title='Degree', yaxis_title='Frequency',
        height=350, margin=dict(l=60, r=20, t=20, b=50),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What if we wanted to map the degree of each node back onto the brain?

    This would allow us to visualize which of the parcels had more direct pairwise connections.

    To do this, we will simply scale our expanded binary mask object by the node degree. We will then combine the masks by concatenating through recasting as a brain_data object and then summing across all ROIs.
    """)
    return


@app.cell
def _(G, mask_x, pd, roi_to_brain):
    degree = pd.Series(dict(G.degree()))
    brain_degree = roi_to_brain(degree, mask_x)
    brain_degree.iplot()
    return (brain_degree,)


@app.cell
def _(brain_degree, view_img_on_surf):
    view_img_on_surf(brain_degree.to_nifti())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This analysis shows that the insula is one of the regions that appears to have the highest degree in this analysis. This is a fairly classic [finding](https://link.springer.com/article/10.1007/s00429-010-0262-0) with the insula frequently found to be highly connected with other regions. Of course, we are only looking at one subject in a very short task (and selecting a completely arbitrary cutoff). We would need to show this survives correction after performing a group analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    Let's practice what we learned through a few different exercises.

    ### 1) Let's calculate seed-based functional connectivity using a different ROI - the right motor cortex

    - Calculate functional connectivity using roi=48 with the whole brain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2) Calculate a group level analysis for this connectivity analysis
    - this will require running this analysis over all subjects
    - then running a one sample t-test
    - then correcting for multiple tests with fdr.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3) Calculate an ICA
    - run an ICA analysis for subject01 with 5 components
    - plot each spatial component and its associated timecourse
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4) Calculate Eigenvector Centrality for each Region
    - figure out how to calculate eigenvector centrality and compute it for each region.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5) Calculate a group level analysis for this graph theoretic analysis
    - this will require running this analysis over all subjects
    - then running a one sample t-test
    - then correcting for multiple tests with fdr.
    """)
    return


if __name__ == "__main__":
    app.run()
