import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    IMG_DIR = _ROOT / "images" / "single_subject"
    return IMG_DIR, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modeling Single Subject Data
    *Written by Luke Chang*

    Now that we have learned the basics of the GLM using simulations, it's time to apply this to working with real data. The first step in fMRI data analysis is to build a model for each subject to predict the activation in a single voxel over the entire scanning session. To do this, we need to build a design matrix for our general linear model. We expect distinct brain regions to be involved in processing specific aspects of our task. This means that we will construct separate regressors that model different brain processes.

    In this tutorial, we will learn how to build and estimate a single subject first-level model and will cover the following topics:
    - Building a design matrix
    - Modeling noise in the GLM with nuisance variables
    - Estimating GLM
    - Performing basic contrasts

    ## Dataset
    We will continue to work with the Pinel Localizer dataset from our preprocessing examples.

    The Pinel Localizer task was designed to probe several different types of basic cognitive processes, such as visual perception, finger tapping, language, and math. Several of the tasks are cued by reading text on the screen (i.e., visual modality) and also by hearing auditory instructions (i.e., auditory modality). The trials are randomized across conditions and have been optimized to maximize efficiency for a rapid event related design. There are 100 trials in total over a 5-minute scanning session. Read the original [paper](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-8-91) for more specific details about the task and the [dataset paper](https://doi.org/10.1016/j.neuroimage.2015.09.052).

    This dataset is well suited for these tutorials as it is (a) publicly available to anyone in the world, (b) relatively small (only about 5min), and (c) provides many options to create different types of contrasts.

    There are a total of 94 subjects available, but we will primarily only be working with a smaller subset of 10-20 participants. See our tutorial on how to download the data if you are not taking the Psych60 version of the class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building a Design Matrix

    First, we will learn the basics of how to build a design matrix for our GLM.

    Let's load all of the python modules we will need to complete this tutorial.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo

    import os
    import glob
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import nibabel as nib
    from nltools.stats import zscore
    from nltools.data import BrainData, DesignMatrix
    from nilearn.plotting import view_img, glass_brain, plot_stat_map
    from dartbrains_tools.data import localizer
    from dartbrains_tools.notebook_utils import youtube


    return (
        BrainData,
        DesignMatrix,
        localizer,
        nib,
        np,
        pd,
        pl,
        plt,
        sns,
        youtube,
        zscore,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To build the design matrix, we will be using the `DesignMatrix` class from the `nltools` toolbox. It reads a BIDS events file (onsets, durations, and condition labels) directly and turns it into a design matrix with one row per TR and one column per condition.

    Two arguments do the work here. `run_length=n_tr` tells it how many TRs the scan has, and `TR=tr` tells it how long each one is — together those define the sampling grid that event times (in seconds) get mapped onto. We also pass `hrf_model=None`, because we want to look at the raw onset regressors first; we will convolve them with the HRF ourselves in a later step. (Leave `hrf_model` out and it convolves for you with a Glover HRF, which is what you would usually want in real analysis.)
    """)
    return


@app.cell
def _(DesignMatrix, localizer, nib):
    def load_bids_events(subject):
        '''Create a DesignMatrix instance from a BIDS events file'''

        tr = localizer.get_tr()
        n_tr = nib.load(localizer.get_file(subject, 'derivatives', 'bold')).shape[-1]
        events_file = localizer.get_file(subject, 'raw', 'events', '.tsv')

        return DesignMatrix(events_file, run_length=n_tr, TR=tr, hrf_model=None)

    dm = load_bids_events('S01')
    return (dm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `DesignMatrix` class is built on top of [Polars](Introduction_to_Polars.md) DataFrames — the same library we covered earlier — and adds methods specific to building design matrices. If you are used to pandas, the main differences are that there is no row index (so no `.loc` / `.iloc`) and that a column comes back as a Polars Series, so you use `.to_numpy()` rather than `.values`. Be sure to check out the [nltools documentation](https://nltools.org/) for more on this class.

    Printing the object shows its shape along with which columns are HRF-convolved regressors and which are nuisance confounds — metadata the class tracks for you as you build the model up.
    """)
    return


@app.cell
def _(dm):
    dm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The underlying Polars dataframe is always available as `.data` if you want to work with the numbers directly.
    """)
    return


@app.cell
def _(dm):
    dm.data.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot each regressor's time course with `.plot(method='timeseries')`.
    """)
    return


@app.cell
def _(dm, plt):
    _f, _a = plt.subplots(figsize=(20, 3))
    dm.plot(method='timeseries', ax=_a)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This plot can be useful sometimes, but here there are too many regressors, which makes it difficult to see what is going on.

    The default `.plot()` draws an SPM-style heatmap instead, which is usually a more useful view of the whole design — time runs down the rows, regressors across the columns.
    """)
    return


@app.cell
def _(dm, plt):
    dm.plot()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### HRF Convolution
    Recall what we learned about convolution in our signal processing tutorial. We can now convolve all of the onset regressors with an HRF function using the `.convolve()` method. By default it will convolve all regressors with the standard double gamma HRF function, though you can specify custom ones and also specific regressors to convolve. Check out the docstrings for more information by adding a `?` after the function name. If you are interested in learning more about different ways to model the HRF using temporal basis functions, watch this [video](https://www.youtube.com/watch?v=YfeMIcDWwko&t=9s).
    """)
    return


@app.cell
def _(dm, plt):
    dm_conv = dm.convolve()
    dm_conv.plot()
    plt.gcf()
    return (dm_conv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that convolving renamed every column: `horizontal_checkerboard` became `horizontal_checkerboard_c0`. The `_c0` suffix identifies which convolution kernel produced the column — useful when you convolve with a basis set of several kernels and get `_c0`, `_c1`, `_c2`. The design matrix keeps track of these names for you in `.convolved`, which matters later when we write contrasts by name.

    You can see that each of the regressors is now a bit blurrier and has the shape of an HRF. We can plot a single regressor to see this more clearly by passing `columns=`.
    """)
    return


@app.cell
def _(dm_conv, plt):
    _f, _a = plt.subplots(figsize=(15, 3))
    dm_conv.plot(method='timeseries', columns=['horizontal_checkerboard_c0'], ax=_a)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Maybe we want to plot both of the checkerboard regressors.
    """)
    return


@app.cell
def _(dm_conv, plt):
    _f, _a = plt.subplots(figsize=(15, 3))
    dm_conv.plot(
        method='timeseries',
        columns=['horizontal_checkerboard_c0', 'vertical_checkerboard_c0'],
        ax=_a,
    )
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
        ### Multicollinearity
        In statistics, collinearity or multicollinearity is when one regressor can be strongly linearly predicted from the others. While this does not actually impact the model's ability to predict data as a whole, it will impact our ability to accurately attribute variance to a single regressor. Recall that in multiple regression, we are estimating the independent variance from each regressor from `X` on `Y`. If there is substantial overlap between the regressors, then the estimator can not attribute the correct amount of variance each regressor accounts for `Y` and the coefficients can become unstable. A more intuitive depiction of this problem can be seen in the venn diagram. The dark orange area in the center at the confluence of all 3 circles reflects the shared variance between `X1` and `X2` on `Y`. If this area becomes bigger, the unique variances become smaller and individually reflect less of the total variance on `Y`.
        """),
        mo.image(str(IMG_DIR / "MultipleRegression.png")),
        mo.md(r"""
        One way to evaluate multicollinearity is to examine the pairwise correlations between each regressor. `.plot(method='corr')` draws that correlation matrix for us.
        """),
    ])
    return


@app.cell
def _(dm_conv, plt):
    dm_conv.plot(method='corr')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Variance Inflation Factor
    Pairwise correlations will let you know if any regressor is correlated with another regressor. However, we are even more concerned about being able to explain any regressor as a linear combination of the other regressors. For example, *can one regressor be explained by three or more of the remaining regressors?* The variance inflation factor (VIF) is a metric that can help us detect multicollinearity. Specifically, it is simply the ratio of variance in a model with multiple terms, divided by the variance of a model with only a single term. This ratio reduces to the following formula:

    $$VIF_j=\frac{1}{1-R_j^2}$$

    Where $R_j^2$ is the $R^2$ value obtained by regressing the $jth$ predictor on the remaining predictors. This means that each regressor $j$ will have its own variance inflation factor.

    How should we interpret the VIF values?

    A VIF of 1 indicates that there is no correlation among the $jth$ predictor and the remaining variables. Values greater than 4 should be investigated further, while VIFs exceeding 10 indicate significant multicollinearity and will likely require intervention.

    Here we will use the `.vif()` method to calculate the variance inflation factor for our design matrix.

    See this [overview](https://online.stat.psu.edu/stat501/lesson/12/12.4) for more details on VIFs.
    """)
    return


@app.cell
def _(dm_conv, plt):
    plt.plot(dm_conv.columns, dm_conv.vif(), linewidth=3)
    plt.xticks(rotation=90)
    plt.ylabel('Variance Inflation Factor')
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
        #### Orthogonalization
        There are many ways to deal with collinearity. In practice, don't worry about collinearity between your covariates. The more pernicious issues are collinearity in your experimental design.

        It is commonly thought that using a procedure called orthogonalization should be used to address issues of multicollinearity. In linear algebra, orthogonalization is the process of prioritizing shared variance between regressors to a single regressor. Recall that the standard GLM already accounts for shared variance by removing it from individual regressors. Orthogonalization allows a user to assign that variance to a specific regressor. However, the process of performing this procedure can introduce artifact into the model and often changes the interpretation of the beta weights in unanticipated ways.
        """),
        mo.image(str(IMG_DIR / "Orthogonalization.png")),
        mo.md(r"""
        In general, we do not recommend using orthogonalization in most use cases, with the exception of centering regressor variables. We encourage the interested reader to review this very useful [overview](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0126255) of collinearity and orthogonalization by Jeanette Mumford and colleagues.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nuisance Variables
    """)
    return


@app.cell
def _(youtube):
    youtube('DEtwsFdFwYc')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filtering
    Recall from our signal processing tutorial, that there are often other types of artifacts in our signal that might take the form of slow or fast oscillations. It is common to apply a high pass filter to the data to remove low frequency artifacts. Often this can also be addressed by simply using a few polynomials to model these types of trends. If we were to directly filter the brain data using something like a butterworth filter as we did in our signal processing tutorial, we would also need to apply it to our design matrix to make sure that we don't have any low frequency drift in experimental design. One easy way to simultaneously perform both of these procedures is to simply build a filter into the design matrix. We will be using a discrete cosine transform (DCT), which is a basis set of cosine regressors of varying frequencies up to a filter cutoff of a specified number of seconds. Many software use 100s or 128s as a default cutoff, but we encourage caution that the filter cutoff isn't too short for your specific experimental design. Longer trials will require longer filter cutoffs. See this [paper](https://www.sciencedirect.com/science/article/pii/S1053811900906098) for a more technical treatment of using the DCT as a high pass filter in fMRI data analysis. In addition, here is a more detailed discussion about [filtering](https://web.archive.org/web/20200224000649/http://mindhive.mit.edu/node/116).
    """)
    return


@app.cell
def _(dm_conv):
    # include_constant=False so that .add_poly() below stays the single source
    # of the intercept — otherwise the two would collide and produce a
    # rank-deficient design.
    dm_conv_filt = dm_conv.add_dct_basis(duration=128, include_constant=False)
    return (dm_conv_filt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cosine regressors that `.add_dct_basis()` just added are automatically registered as *confounds* — nuisance regressors we want to model but don't care about interpreting. We can pull that list off the design matrix with `.confounds` and plot just those columns.
    """)
    return


@app.cell
def _(dm_conv_filt, plt):
    _f, _a = plt.subplots(figsize=(20, 3))
    dm_conv_filt.plot(method='timeseries', columns=dm_conv_filt.confounds, ax=_a)
    return


@app.cell
def _(dm_conv, plt):
    dm_conv_filt_1 = dm_conv.add_dct_basis(duration=128, include_constant=False)
    dm_conv_filt_1.plot()
    plt.gcf()
    return (dm_conv_filt_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Intercepts
    We almost always want to include an intercept in our model. This will usually reflect the baseline, or the average voxel response during the times that are not being modeled as a regressor. It is important to note that you must have some sparsity to your model, meaning that you can't model every point in time, as this will make your model rank deficient and unestimable.

    If you are concatenating runs and modeling them all together, it is recommended to include a separate intercept for each run rather than a single intercept spanning the full concatenated model. This means that the average response within a voxel might differ across runs. You can add an intercept by simply creating a new column of ones (e.g., `dm['Intercept'] = 1`). Here we provide an example using the `.add_poly()` method, which adds an intercept by default.
    """)
    return


@app.cell
def _(dm_conv_filt_1, plt):
    dm_conv_filt_poly = dm_conv_filt_1.add_poly()
    dm_conv_filt_poly.plot()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Linear Trends
    We also often want to remove any slow drifts in our data.  This might include a linear trend and a quadratic trend. We can also do this with the `.add_poly()` method and adding all trends up to an order of 2 (e.g., quadratic). We typically use this approach rather than applying a high pass filter when working with naturalistic viewing data.

    Notice that these do not appear to be very different from the high pass filter basis set. It's actually okay if there is collinearity in our covariate regressors. Collinearity is only a problem when it correlates with the task regressors as it means that we will not be able to uniquely model the variance. The DCT can occasionally run into edge artifacts, which can be addressed by the linear trend.
    """)
    return


@app.cell
def _(dm_conv_filt_1, plt):
    dm_conv_filt_poly_1 = dm_conv_filt_1.add_poly(order=2, include_lower=True)
    dm_conv_filt_poly_1.plot()
    plt.gcf()
    return (dm_conv_filt_poly_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Noise Covariates
    Another important thing to consider is removing variance associated with head motion. Remember the preprocessed data has already realigned each TR in space, but head motion itself can nonlinearly distort the magnetic field. There are several common strategies for trying to remove artifacts associated with head motion. One is using a data driven denoising algorithm like ICA and combining it with a classifer such as FSL's [FIX](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIX) module. Another approach is to include the alignment correction applied to each TR as a covariate. For example, if someone moved a lot in a single TR, there will be a strong change in their realignment parameters. It is common to include the 6 parameters as covariates in your regression model. However, as we already noted, often motion can have a nonlinear relationship with signal intensity, so it is often good to include other transformations of these signals to capture nonlinear signal changes resulting from head motion. We typically center the six realignment parameters (or zscore) and then additionally add a quadratic version, a derivative, and the square of the derivatives, which becomes 24 additional regressors.

    In addition, it is common to model out big changes using a regressor with a single value indicating the timepoint of the movement. This will be zeros along time, with a single value of one at the time point of interest. This effectively removes any variance associated with this single time point. It is important to model each "spike" as a separate regressor as there might be distinct spatial patterns associated with different types of head motions. We strongly recommend against using a single continuous frame displacement metric as is often recommended by the fMRIprep team. This assumes (1) that there is a *linear* relationship between displacement and voxel activity, and (2) that there is a *single* spatial generator or pattern associated with frame displacement. As we saw in the ICA noise lab, there might be many different types of head motion artifacts. This procedure of including spikes as nuisance regressors is mathematically equivalent to censoring your data and removing the bad TRs. We think it is important to do this in the context of the GLM as it will also reduce the impact if it happens to covary with your task.

    First, let's load preprocessed data from one participant.
    """)
    return


@app.cell
def _(BrainData, localizer):
    sub = 'S01'
    data = BrainData(localizer.get_file(sub, 'derivatives', 'bold'))
    return data, sub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's inspect the realignment parameters for this participant. These pertain to how much each volume had to be moved in the (X,Y,Z) planes and rotations around each axis. We are standardizing the data so that rotations and translations are on the same scale.
    """)
    return


@app.cell
def _(localizer, pd, plt, zscore):
    covariates = pd.read_csv(localizer.get_file('S01', 'derivatives', 'confounds'), sep='\t')

    mc = covariates[['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z']]

    plt.figure(figsize=(15,5))
    plt.plot(zscore(mc).to_numpy())
    return (mc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's build the 24 covariates related to head motion. We include the 6 realignment parameters that have been standardized. In addition, we add their quadratic, their derivative, and the square of their derivative.

    We can create a quick visualization to see what the overall pattern is across the different regressors.
    """)
    return


@app.cell
def _(DesignMatrix, localizer, mc, pl, plt, zscore):
    def make_motion_covariates(mc, tr):
        # nltools' zscore accepts pandas but always returns Polars, so we build
        # the expansion with Polars expressions. Each expression is evaluated
        # against the original z-scored columns, so `.diff()` below refers to
        # the realignment parameters and not to the squared versions.
        z = zscore(mc)
        cols = z.columns
        all_mc = z.with_columns(
            [pl.col(c).pow(2).alias(f"{c}_sq") for c in cols]
            + [pl.col(c).diff().alias(f"{c}_diff") for c in cols]
            + [pl.col(c).diff().pow(2).alias(f"{c}_diff_sq") for c in cols]
        ).fill_null(0)
        return DesignMatrix(all_mc, sampling_freq=1/tr)

    tr = localizer.get_tr()
    mc_cov = make_motion_covariates(mc, tr)

    mc_cov.plot()
    plt.gcf()
    return mc_cov, tr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's try to find some spikes in the data. This is performed by finding TRs that exceed a global mean threshold and also that exceed an overall average intensity change by a threshold.  We are using an arbitrary cutoff of 3 standard deviations as a threshold.

    First, let's plot the average signal intensity across all voxels over time.
    """)
    return


@app.cell
def _(data, np, plt):
    plt.figure(figsize=(15,3))
    plt.plot(np.mean(data.data, axis=1), linewidth=3)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Intensity', fontsize=18)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice there is a clear slow drift in the signal that we will need to remove with our high pass filter.

    Now, let's see if there are any spikes in the data that exceed our threshold. What happens if we use a different threshold?
    """)
    return


@app.cell
def _(data, plt, tr):
    # find_spikes returns a DesignMatrix with one indicator column per spike,
    # already marked as confounds. Passing TR sets its sampling frequency so it
    # can be appended to the main design matrix below.
    spikes = data.find_spikes(global_spike_cutoff=2, diff_spike_cutoff=2.5, TR=tr)
    _f, _a = plt.subplots(figsize=(15, 3))
    spikes.plot(method='timeseries', ax=_a)
    return (spikes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this subject, our spike identification procedure identified 3 spikes. Two of these spikes look like they are temporally contiguous. Let's add all of these covariates to our design matrix.

    In this example, we will append each of these additional matrices to our main design matrix.

    **Note**: `.append()` requires that all matrices are a design_matrix with the same sampling frequency.
    """)
    return


@app.cell
def _(dm_conv_filt_poly_1, mc_cov, plt, spikes):
    dm_conv_filt_poly_cov = dm_conv_filt_poly_1.append([mc_cov, spikes], axis=1)
    dm_conv_filt_poly_cov.plot()
    plt.gcf()
    return (dm_conv_filt_poly_cov,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note**: As discussed above, multicollinearity can lead to unstable estimates of the regression coefficients, which is particularly important to keep in mind for regressors of interest. Multicollinearity between covariates of no interest tends to be less of a problem because we generally are just interested in explaining noise in our model and are rarely interested in interpreting the individual covariate beta coefficients. However, in cases of extreme collinearity, the model may become rank deficient, which can lead to difficulty with even fitting the model.

    A simple fix to this problem is to use the `.clean()` method. This method will remove any columns that are perfectly collinear with other columns in the design matrix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Smoothing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To increase the signal to noise ratio and clean up the data, it is common to apply spatial smoothing to the image.

    Here we will convolve the image with a 3-D gaussian kernel, with a 6mm full width half maximum (FWHM) using the `.smooth()` method.
    """)
    return


@app.cell
def _(data):
    fwhm=6
    smoothed = data.smooth(fwhm=fwhm)
    return fwhm, smoothed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take a look and see how this changes the image.
    """)
    return


@app.cell
def _(data):
    data.mean().iplot()
    return


@app.cell
def _(smoothed):
    smoothed.mean().iplot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimate GLM for all voxels
    Now we are ready to estimate the regression model for all voxels.

    We pass our design matrix to the `.fit()` method with `model='glm'`. This runs the same regression separately on every voxel in the brain and stores the results back on the `BrainData` object as attributes, in the style of a scikit-learn estimator.
    """)
    return


@app.cell
def _(dm_conv_filt_poly_cov, smoothed):
    smoothed.fit(model='glm', X=dm_conv_filt_poly_cov)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, it's done! Let's take a look at the results.

    `.fit()` attached the main results to the object as `BrainData` instances, one image per quantity: `.glm_betas` (a beta image per regressor), plus `.glm_t`, `.glm_p`, `.glm_se`, `.glm_residual`, `.glm_r2`, and the fitted model itself in `.model_`.

    Remember we have run the same regression model separately on each voxel of the brain.

    Let's take a look at one of the regressors. Each row of `.glm_betas` corresponds to a column of the design matrix, so we can print the design matrix column names to see what we have. Let's plot the first one, which corresponds to `audio_computation_c0`, an arithmetic problem presented in the auditory domain.
    """)
    return


@app.cell
def _(dm_conv_filt_poly_cov):
    print(dm_conv_filt_poly_cov.columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BrainData instances have their own plotting methods. We will be using `.iplot()` here, which can allow us to interactively look at all of the values.

    If you would like to see the top values, we can quickly apply a threshold. Try using `95`% threshold, and be sure to click the `percentile_threshold` option.
    """)
    return


@app.cell
def _(smoothed):
    smoothed.glm_betas[0].iplot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save Image
    We will frequently want to save different brain images we are working with to a nifti file. This is useful for saving intermediate work, or sharing our results with others. This is easy with the `.write()` method. Be sure to specify a path and file name for the file.

    **Note**: You can only write to folders where you have permission. Try changing the path to your own directory.
    """)
    return


@app.cell
def _(fwhm, smoothed, sub):
    smoothed.write(f'{sub}_betas_denoised_smoothed{fwhm}_preprocessed_fMRI_bold.nii.gz')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contrasts

    Now that we have estimated our model, we will likely want to create contrasts to examine brain activation to different conditions.

    This procedure is identical to those introduced in our GLM tutorial.

    Let's watch another video by Tor Wager to better understand contrasts at the first-level model stage.
    """)
    return


@app.cell
def _(youtube):
    youtube('7MibM1ATai4')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's try making a simple contrast where we average only the regressors pertaining to motor. This is essentially summing all of the motor regressors and dividing by the number of regressors.

    A contrast is a set of weights over the regressors. We express it by *name* using `compute_contrasts`, which accepts a string of regressor names combined with `+`, `-`, and scalar multipliers. Naming the regressors is much safer than indexing them by position: the column order depends on how the events file was read, so positional indices are a classic source of silent errors.

    One thing to be aware of: by default `.fit()` runs `.clean()` on the design before estimating, dropping any column that correlates above `design_clean_thresh` (0.95) with an earlier one. Here that removes `poly_1` and `poly_2`, because our linear and quadratic drift terms are nearly identical to the first two cosine regressors — the DCT basis and the polynomial trends are modeling the same slow drift, so including both is redundant.

    Two caveats worth internalizing. First, this is a *correlation* heuristic, not a rank test: this particular design matrix is full rank (48 of 48 columns), so it was perfectly estimable as specified. Second, `clean()` keeps the first column of a correlated pair and drops the second, so **which** regressor survives depends on the order you built the design in — had we called `.add_poly()` before `.add_dct_basis()`, the cosines would have been dropped instead. If you want full control, pass `design_clean=False` and handle redundancy yourself.

    So the fitted model can have fewer regressors than the design matrix you handed it. You can see what was actually estimated in `smoothed.model_.design_matrices_[0].columns` — another reason to refer to regressors by name rather than by index.
    """)
    return


@app.cell
def _(smoothed):
    print(f"design matrix regressors: {len(smoothed.model_.design_matrices_[0].columns)}")

    motor = smoothed.compute_contrasts(
        '0.25*audio_left_hand_c0 + 0.25*audio_right_hand_c0 '
        '+ 0.25*video_left_hand_c0 + 0.25*video_right_hand_c0',
        statistic='beta',
    )

    motor.iplot()
    return (motor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, now we can clearly see regions specifically involved in motor processing.

    Now let's see which regions are more active when making motor movements with our right hand compared to our left hand.

    The contrast reads directly as the hypothesis: right-hand regressors minus left-hand regressors.
    """)
    return


@app.cell
def _(smoothed):
    motor_rvl = smoothed.compute_contrasts(
        'audio_right_hand_c0 + video_right_hand_c0 '
        '- audio_left_hand_c0 - video_left_hand_c0',
        statistic='beta',
    )

    motor_rvl.iplot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What do you see?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    For homework, let's get a better handle on how to play with our data and test different hypotheses.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Which regions are more involved with visual compared to auditory sensory processing?
     - Create a contrast to test this hypothesis
     - plot the results
     - write the file to your output folder.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Which regions are more involved in processing numbers compared to words?
     - Create a contrast to test this hypothesis
     - plot the results
     - write the file to your output folder.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Which regions are more involved with motor compared to cognitive processes (e.g., language and math)?
     - Create a contrast to test this hypothesis
     - plot the results
     - write the file to your output folder.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. How are your results impacted by different smoothing kernels?
     - Pick two different sized smoothing kernels and create two new brain images with each smoothing kernel
     - Pick any contrast of interest to you and evaluate the impact of smoothing on the contrast.
     - plot the results
     - write the file to your output folder.
    """)
    return


if __name__ == "__main__":
    app.run()
