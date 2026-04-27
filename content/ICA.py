import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    _ROOT = next(p for p in (Path.cwd(), *Path.cwd().resolve().parents) if (p / "book.yml").exists())
    IMG_DIR = _ROOT / "images" / "ica"
    return IMG_DIR, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Separating Signal From Noise With ICA
    *Written by Luke Chang*

    In this tutorial we will use ICA to explore which signals in our imaging data might be real signal or artifacts. We cover ICA in more depth in the [multivariate decomposition](Connectivity.md) subsection of the [connectivity](Connectivity.md) tutorial and also using simulations in the [ICA](Introduction_to_ICA.md) tutorial.

    To run this tutorial, we will be working with data that has already been preprocessed. *If you are in Psych60, this has already been done for you*. If you reading this online, then I recommend reading the [preprocessing tutorial](Preprocessing.md), or downloading the data using the [Download Data](Download_Data.md) tutorial.

    For a brief overview of types of artifacts that might be present in your data, I recommend watching this video by Tor Wager and Martin Lindquist.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <iframe width="560" height="315"
        src="https://www.youtube.com/embed/7Kk_RsGycHs"
        frameborder="0" allowfullscreen>
    </iframe>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading Data
    Ok, let's load a subject and run an ICA to explore signals that are present. Since we have completed preprocessing, our data should be realigned and also normalized to MNI stereotactic space. We will use the [nltools](https://nltools.org/) package to work with this data in python.
    """)
    return


@app.cell
def _():
    import numpy as np
    from numpy.fft import fft, fftfreq
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from nltools.data import Brain_Data
    from nilearn.plotting import view_img
    from dartbrains_tools.data import get_file, get_tr

    return (
        Brain_Data,
        fft,
        fftfreq,
        get_file,
        get_tr,
        go,
        make_subplots,
        np,
        view_img,
    )


@app.cell
def _(Brain_Data, get_file):
    sub = 'S01'
    data = Brain_Data(get_file(sub, 'derivatives', 'bold'))
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## More Preprocessing
    Even though, we have technically already run most of the preprocessing there are a couple of more steps that will help make the ICA cleaner.

    First, we will run a high pass filter to remove any low frequency scanner drift. We will pick a fairly arbitrary filter size of 0.0078hz (1/128s). We will also run spatial smoothing with a 6mm FWHM gaussian kernel to increase a signal to noise ratio at each voxel. These steps are very easy to run using nltools after the data has been loaded.
    """)
    return


@app.cell
def _(data, get_tr, mo):
    tr = get_tr()
    with mo.persistent_cache("ica_preprocess"):
        data_1 = data.filter(sampling_freq=1 / tr, high_pass=1 / 128)
        data_1 = data_1.smooth(6)
    return data_1, tr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independent Component Analysis (ICA)
    Ok, we are finally ready to run an ICA analysis on our data.

    ICA attempts to perform blind source separation by decomposing a multivariate signal into additive subcomponents that are maximally independent.

    We will be using the `decompose()` method on our `Brain_Data` instance. This runs the [FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html) algorithm implemented by scikit-learn. You can choose whether you want to run spatial ICA by setting `axis='voxels` or temporal ICA by setting `axis='images'`. We also recommend running the whitening flat `whiten=True`. By default `decompose` will estimate the maximum components that are possible given the data. We recommend using a completely arbitrary heuristic of 20-30 components.
    """)
    return


@app.cell
def _(data_1, mo):
    with mo.persistent_cache("ica_decompose"):
        # 10 components keeps the WASM-mode page snappy: the slider
        # range stays small and ICA convergence in Pyodide is faster.
        # Cranking back to 20-30 is one number-edit away if the reader
        # wants more granularity.
        output = data_1.decompose(algorithm='ica', n_components=10, axis='images', whiten='unit-variance')
    return (output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Viewing Components

    We will use an interactive component viewer to explore the results of the analysis. Use the **Component** slider to select which component to view; the plot updates automatically as you move it.

    Components have been standardized so we can threshold the brain in terms of standard deviations. We display voxels that load greater or less than 2 standard deviations on the component (a tighter or looser cutoff than 2.0 σ requires running this notebook in `marimo edit`).

    The second plot is the time course of the voxels that load on the component. The x-axis is in TRs, which for this dataset is 2.4 sec.

    The third plot is the powerspectrum of the timecourse. There is not a large range of possible values as we can only observe signals at the nyquist frequency, which is half of our sampling frequency of 1/2.4s (approximately 0.21hz) to a lower bound of 0.0078hz based on our high pass filter. There might be systematic oscillatory signals. Remember, that signals that oscillate a faster frequency than the nyquist frequency will be aliased. This includes physiological artifacts such as respiration and cardiac signals.

    It is important to note that ICA cannot resolve the sign of the component. So make sure you consider signals that are positive as well as negative.
    """)
    return


@app.cell(hide_code=True)
def _(mo, output):
    # Single slider over component index. `stop=9` is hardcoded (must
    # match n_components-1 in the decompose() call above) — marimo-book's
    # precompute scanner only detects literal slider bounds, so a dynamic
    # `stop=len(output['components'])-1` would silently render frozen.
    # Threshold is fixed at 2.0 σ to keep the lookup table small.
    component_slider = mo.ui.slider(
        start=0, stop=9, step=1,
        value=0, label="Component", show_value=True,
    )
    component_slider
    return (component_slider,)


@app.cell(hide_code=True)
def _(
    component_slider,
    fft,
    fftfreq,
    go,
    make_subplots,
    mo,
    np,
    output,
    tr,
    view_img,
):
    _component = component_slider.value
    _threshold = 2.0  # standard-deviation threshold for the brain plot

    # Brain viewer: z-score the component, let view_img handle threshold display
    _comp = output['components'][_component]
    _zscored = (_comp - _comp.mean()) * (1 / _comp.std())
    _brain_html = view_img(
        _zscored.to_nifti(),
        threshold=_threshold,
        black_bg=True,
        symmetric_cmap=True,
        title=f'Component {_component}/{len(output["components"])}',
    )

    # Plotly: timecourse + power spectrum
    _timecourse = output['weights'][:, _component]
    _y = fft(_timecourse)
    _f = fftfreq(len(_y), d=tr)
    _freq_mask = _f > 0

    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Timecourse (TR={tr}s)', 'Power Spectrum'),
        horizontal_spacing=0.18,
    )
    _fig.add_trace(
        go.Scatter(
            x=list(range(len(_timecourse))),
            y=_timecourse,
            mode='lines',
            line=dict(color='#EF553B', width=2),
            name='Timecourse',
            hovertemplate='TR: %{x}<br>Intensity: %{y:.4f}<extra></extra>',
        ),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_f[_freq_mask],
            y=np.abs(_y)[_freq_mask] ** 2,
            mode='lines',
            line=dict(color='#636EFA', width=2),
            name='Power',
            hovertemplate='Freq: %{x:.4f} Hz<br>Power: %{y:.4e}<extra></extra>',
        ),
        row=1, col=2,
    )
    _fig.update_xaxes(title_text='TR', row=1, col=1)
    _fig.update_yaxes(title_text='Intensity (AU)', row=1, col=1)
    _fig.update_xaxes(title_text='Frequency (Hz)', row=1, col=2)
    _fig.update_yaxes(title_text='Power', row=1, col=2)
    _fig.update_layout(
        height=280,
        showlegend=False,
        margin=dict(t=40, b=50, l=70, r=30),
    )

    mo.vstack([mo.Html(_brain_html._repr_html_()), _fig], gap=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    For this tutorial, try to guess which components are signal and which are noise. Also, be sure to label the type of noise you think you might be seeing (e.g., head motion, scanner spikes, cardiac, respiration, etc.) Do this for subjects `s01` and `s02`.

    What features do you think are important to consider when making this judgment?  Does the spatial map provide any useful information? What about the timecourse of the component? Does it map on to the plausible timecourse of the task.What about the power spectrum?
    """)
    return


if __name__ == "__main__":
    app.run()
