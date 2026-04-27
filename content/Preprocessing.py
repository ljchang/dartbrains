import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium", app_title="Preprocessing")


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
        # Preprocessing
        *Written by Luke Chang*

        Being able to study brain activity associated with cognitive processes
        in humans is an amazing achievement. However, there is an extraordinary
        amount of noise and very low levels of signal, which makes it difficult
        to make inferences about brain function using BOLD imaging. A critical
        step before any analysis is to remove as much noise as possible. The
        series of steps to remove noise comprise our *neuroimaging data
        **preprocessing** pipeline*. See slides on our preprocessing lecture [here](../images/lectures/Preprocessing.pdf).
        """),
        mo.image(str(IMG_DIR / "preprocessing.png")),
        mo.md(r"""
        In this lab, we will go over the basics of preprocessing fMRI data using the [fmriprep](https://fmriprep.org/) preprocessing pipeline. We will cover:

        - **Image transformations** (rigid body and affine)
        - **Cost functions** for image registration
        - **Head motion correction** (realignment)
        - **Spatial normalization**
        - **Spatial smoothing**
        - **fMRIPrep** automated preprocessing pipeline

        There are other preprocessing steps that are also common, but not necessarily performed by all labs such as slice timing and distortion correction. We will not be discussing these in depth outside of the videos.

        Let’s start with watching a short video by Martin Lindquist to get a general overview of the main steps of preprocessing and the basics of how to transform images and register them to other images.
        """),
    ])
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from pathlib import Path
    from dartbrains_tools.mr_widgets import TransformCubeWidget, CostFunctionWidget, SmoothingWidget

    # Locate the repo root so we can find the sibling images/ directory.
    # Under MarimoIslandGenerator (WASM build), both __file__ and
    # mo.notebook_dir() resolve to marimo-internal paths (.venv/bin/),
    # so we walk up from cwd looking for book.yml. Falls back to cwd
    # in marimo-edit (where cwd is already the project root).
    def _find_root() -> Path:
        for candidate in (Path.cwd(), *Path.cwd().resolve().parents):
            if (candidate / "book.yml").exists():
                return candidate
        return Path.cwd()

    _ROOT = _find_root()
    IMG_DIR = _ROOT / "images" / "preprocessing"
    return (
        CostFunctionWidget,
        IMG_DIR,
        SmoothingWidget,
        TransformCubeWidget,
        mo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
      <iframe
          width="560" height="315"
          src="https://www.youtube.com/embed/Qc3rRaJWOc4"
          frameborder="0" allowfullscreen>
      </iframe>
      """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Image Transformations

    Ok, now let’s dive deeper into how we can transform images into different spaces using linear transformations.

    Recall from our introduction to neuroimaging data lab, that neuroimaging data is typically stored in a nifti container, which contains a 3D or 4D matrix of the voxel intensities and also an **affine matrix** that maps voxel
    coordinates \((i, j, k)\) to world coordinates \((x, y, z)\).


    $$\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \mathbf{A} \begin{bmatrix} i \\ j \\ k \\ 1 \end{bmatrix}$$

    A **rigid body transformation** has 6 parameters:
    - 3 **translations** (shift in x, y, z)
    - 3 **rotations** (around x, y, z axes)

    A full **affine transformation** adds 6 more (12 total):
    - 3 **scale** factors
    - 3 **shear** parameters

    Let’s create an interactive plot to get an intuition for how these affine matrices can be used to transform a 3D image. The ghost (wireframe) shows the original position; the solid red cube shows the transformed
    position.

    We can move the sliders to play with applying rigid body transforms to a 3D cube. A rigid body transformation has 6 parameters: translation in x,y, & z, and rotation around each of these axes. The key thing to remember is that a rigid body transform doesn’t allow the image to be fundamentally changed. A full 12 parameter affine transformation adds an additional 3 parameters each for scaling and shearing, which can change the shape of the cube.

    Try moving some of the sliders around. Each time you move a slider it is applying an affine transformation to the matrix and re-plotting.

    Translation moves the cube in x, y, and z dimensions.

    We can also rotate the cube around the x, y, and z axes where the origin is the center point. Continuing to rotate around the point will definitely lead to the cube leaving the current field of view, but it will come back if you keep rotating it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    tx_slider = mo.ui.slider(start=-15, stop=15, step=0.5, value=0, label="Translate X")
    ty_slider = mo.ui.slider(start=-15, stop=15, step=0.5, value=0, label="Translate Y")
    tz_slider = mo.ui.slider(start=-15, stop=15, step=0.5, value=0, label="Translate Z")
    rx_slider = mo.ui.slider(start=-180, stop=180, step=5, value=0, label="Rotate X (\u00b0)")
    ry_slider = mo.ui.slider(start=-180, stop=180, step=5, value=0, label="Rotate Y (\u00b0)")
    rz_slider = mo.ui.slider(start=-180, stop=180, step=5, value=0, label="Rotate Z (\u00b0)")
    sx_slider = mo.ui.slider(start=0.5, stop=2.0, step=0.1, value=1.0, label="Scale X")
    sy_slider = mo.ui.slider(start=0.5, stop=2.0, step=0.1, value=1.0, label="Scale Y")
    sz_slider = mo.ui.slider(start=0.5, stop=2.0, step=0.1, value=1.0, label="Scale Z")
    return (
        rx_slider,
        ry_slider,
        rz_slider,
        sx_slider,
        sy_slider,
        sz_slider,
        tx_slider,
        ty_slider,
        tz_slider,
    )


@app.cell(hide_code=True)
def _(
    TransformCubeWidget,
    mo,
    rx_slider,
    ry_slider,
    rz_slider,
    sx_slider,
    sy_slider,
    sz_slider,
    tx_slider,
    ty_slider,
    tz_slider,
):
    _widget = TransformCubeWidget(
        trans_x=float(tx_slider.value), trans_y=float(ty_slider.value), trans_z=float(tz_slider.value),
        rot_x=float(rx_slider.value), rot_y=float(ry_slider.value), rot_z=float(rz_slider.value),
        scale_x=float(sx_slider.value), scale_y=float(sy_slider.value), scale_z=float(sz_slider.value),
    )
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        mo.md("**Translation:**"),
        mo.hstack([tx_slider, ty_slider, tz_slider], justify="start", gap=1),
        mo.md("**Rotation:**"),
        mo.hstack([rx_slider, ry_slider, rz_slider], justify="start", gap=1),
        mo.md("**Scale:**"),
        mo.hstack([sx_slider, sy_slider, sz_slider], justify="start", gap=1),
        _wrapped,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**Try this:** Move translation sliders to shift the cube. "
            "Rotate around each axis. Change scale to stretch/compress "
            "(goes beyond rigid body!). The affine matrix on the right "
            "updates live. **Drag to orbit** the 3D view."
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The affine matrix encodes all of these transformations compactly. The
    upper-left 3\(\times\)3 block contains rotation and scaling, while the
    rightmost column contains translation. The bottom row is always
    \([0, 0, 0, 1]\).

    **Rotation matrices** for each axis:

    $$R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}$$

    $$R_y(\phi) = \begin{bmatrix} \cos\phi & 0 & \sin\phi \\ 0 & 1 & 0 \\ -\sin\phi & 0 & \cos\phi \end{bmatrix}$$

    $$R_z(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

    The combined rotation is \(R = R_x \cdot R_y \cdot R_z\), and the full
    affine matrix is built by combining rotation, scaling, and translation.

    Every time we apply a transformation to our images, the result is not
    a perfect representation of the original data -- we need **interpolation**
    to fill gaps, which introduces small errors. This is why minimizing the
    number of resampling steps matters in preprocessing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Cost Functions

    Now that we understand affine transformations, how do we use them to
    **register** one brain image to another? We need a way to quantify
    how well two images are aligned.

    The key is to identify a way to quantify how aligned the two images are to each other. Our visual systems are very good at identifying when two images are aligned, however, we need to create a quantitative alignment measure. These measures are called
    **cost functions**.

    A common cost function is the **Sum of Squared Errors (SSE)**. You may remember that this same cost function is used by linear regression to find the best fitting line to data. This measure works best if the images are of the same type and have roughly equivalent signal intensities.

    $$SSE = \sum_{i} (I_{target}(i) - I_{reference}(i))^2$$

    The goal is to find the transformation parameters that **minimize**
    the cost function. This process is called **optimization**.

    Let’s create another interactive plot and find the optimal X & Y translation parameters that minimize the difference between a two-dimensional target image to a reference image. Try to align the target image (blue square) with the reference image
    by adjusting the translation sliders. Watch the SSE drop to zero when perfectly aligned!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cost_tx = mo.ui.slider(start=0, stop=20, step=1, value=0, label="Translate X")
    cost_ty = mo.ui.slider(start=0, stop=20, step=1, value=0, label="Translate Y")
    return cost_tx, cost_ty


@app.cell(hide_code=True)
def _(CostFunctionWidget, cost_tx, cost_ty, mo):
    _widget = CostFunctionWidget(
        trans_x=float(cost_tx.value),
        trans_y=float(cost_ty.value),
    )
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        mo.hstack([cost_tx, cost_ty], justify="start", gap=2),
        _wrapped,
        mo.callout(
            mo.md(
                "**Goal:** Find the translation that makes SSE = 0 (perfect overlap, shown in green). "
                "This is exactly what registration algorithms do automatically -- they search the "
                "parameter space to minimize the cost function."
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You probably had to move the sliders around back and forth until you were able to reduce the sum of squared error to zero. This cost function increases exponentially the further you are away from your target. The process of minimizing (or sometimes maximizing) cost functions to identify the best fitting parameters is called optimization and is a concept that is core to fitting models to data across many different disciplines.

    Different cost functions work best for different image types:

    | Cost Function | Use Case | Example |
    |:---:|:---:|:---:|
    | Sum of Squared Error | Same modality & scaling | Two T2* images |
    | Normalized correlation | Same modality | Two T1 images |
    | Correlation ratio | Any modality | T1 and FLAIR |
    | Mutual information | Any modality | T1 and CT |
    | Boundary Based Registration | Contrast across boundaries | EPI and T1 |
    """)
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
        ---
        ## Realignment

        Now let's put everything we learned together to understand how we can correct for head motion in functional images that occurred during a scanning session. It is extremely important to make sure that a specific voxel has the same 3D coordinate across all time points to be able to model neural processes. This of course is made difficult by the fact that participants move during a scanning session and also in between runs.

        Realignment is the preprocessing step in which a rigid body transformation is applied to each volume to align them to a common space. One typically needs to choose a reference volume, which might be the first, middle, or last volume, or the mean of all volumes.

        Let's look at an example of the translation and rotation parameters after running realignment on our first subject.
        """),
        mo.image(str(IMG_DIR / "realignment_parameters.png")),
        mo.md(r"""
        Don't forget that even though we can approximately put each volume into a similar position with realignment, head motion always distorts the magnetic field and can lead to nonlinear changes in signal intensity that will not be addressed by this procedure. In the resting-state literature, where many analyses are based on functional connectivity, head motion can lead to spurious correlations. Some researchers choose to exclude any subject that moved more than a certain amount. Others choose to remove the impact of these time points in their data through removing the volumes via *scrubbing* or modeling out the volume with a dummy code in the first level general linear models.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Spatial Normalization

    There are several other preprocessing steps that involve image registration. The main one is called *spatial normalization*, in which each subject's brain data is warped into a common stereotactic space. Talairach is an older space, that has been subsumed by various standards developed by the Montreal Neurological Institute.

    There are a variety of algorithms to warp subject data into stereotactic space. Linear 12 parameter affine transformations have increasingly been replaced by more complicated nonlinear normalizations that have hundreds to thousands of parameters.

    One nonlinear algorithm that has performed very well across comparison studies is *diffeomorphic registration*, which can also be inverted so that subject space can be transformed into stereotactic space and back to subject space. This is the core of the [ANTs](http://stnava.github.io/ANTs/) algorithm that is implemented in fmriprep.

    Let's watch another short video by Martin Lindquist and Tor Wager to learn more about the core preprocessing steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
      <iframe
          width="560" height="315"
          src="https://www.youtube.com/embed/qamRGWSC-6g"
          frameborder="0" allowfullscreen>
      </iframe>
      """)
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
        There are many different steps involved in the spatial normalization process and these details vary widely across various imaging software packages. We will briefly discuss some of the steps involved in the anatomical preprocessing pipeline implemented by fMRIprep and will be showing example figures from the output generated by the pipeline.

        First, brains are extracted from the skull and surrounding dura mater. You can check and see how well the algorithm performed by examining the red outline.
        """),
        mo.image(str(IMG_DIR / "T1_normalization.png")),
        mo.md(r"""
        Next, the anatomical images are segmented into different tissue types. These tissue maps are used for various types of analyses, including providing a grey matter mask to reduce the computational time in estimating statistics. In addition, they provide masks to aid in extracting average activity in CSF, or white matter, which might be used as covariates in the statistical analyses to account for physiological noise.
        """),
        mo.image(str(IMG_DIR / "T1_segmentation.png")),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial normalization of the anatomical T1w reference

    fmriprep uses [ANTs](http://stnava.github.io/ANTs/) to perform nonlinear spatial normalization. It is easy to check to see how well the algorithm performed by viewing the results of aligning the T1w reference to the stereotactic reference space. Hover on the panels with the mouse pointer to transition between both spaces. We are using the MNI152NLin2009cAsym template.
    """)
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.Html(f'<div style="max-width:100%">{(IMG_DIR / "sub-S01_space-MNI152NLin2009cAsym_T1w.svg").read_text()}</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Alignment of functional and anatomical MRI data

    Next, we can evaluate the quality of alignment of the functional data to the anatomical T1 image. FSL `flirt` was used to generate transformations from EPI-space to T1w-space — the white matter mask calculated with FSL `fast` (brain tissue segmentation) was used for BBR. Note that Nearest Neighbor interpolation is used in the reportlets in order to highlight potential spin-history and other artifacts, whereas final images are resampled using Lanczos interpolation. Notice these images are much blurrier and show some distortion compared to the T1s.
    """)
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.Html(f'<div style="max-width:100%">{(IMG_DIR / "sub-S01_task-localizer_desc-flirtbbr_bold.svg").read_text()}</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Spatial Smoothing

    Spatial smoothing applies a 3D Gaussian kernel to each volume,
    blurring the image to remove high-frequency spatial noise. The amount
    of smoothing is specified by the **Full Width at Half Maximum (FWHM)**
    of the Gaussian kernel.

    Why blur the image *after* trying so hard to increase resolution?
    - Increases **signal-to-noise ratio**
    - Reduces impact of **partial volume effects**
    - Compensates for **residual misalignment** after normalization
    - Satisfies assumptions of **Random Field Theory** for multiple
      comparisons correction

    A common FWHM is 6mm for standard fMRI analyses, though this varies
    by analysis type (e.g., MVPA typically uses little or no smoothing).

    Adjust the FWHM below to see how smoothing affects the image.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fwhm_slider = mo.ui.slider(start=0, stop=20, step=0.5, value=0, label="FWHM (mm)", full_width=True)
    return (fwhm_slider,)


@app.cell(hide_code=True)
def _(SmoothingWidget, fwhm_slider, mo):
    _widget = SmoothingWidget(fwhm=float(fwhm_slider.value))
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        fwhm_slider,
        _wrapped,
        mo.callout(
            mo.md(
                "**Try this:** Start at FWHM=0 (no smoothing) and slowly increase. "
                "Notice how fine details (ventricle edges, cortical folds) blur first, "
                "while overall brain shape is preserved. At FWHM > 10mm, tissue boundaries "
                "start to disappear. The kernel visualization in the center shows the "
                "Gaussian width scaling with FWHM."
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is what a 3D Gaussian kernel looks like.
    """)
    return


@app.cell(hide_code=True)
def _():
    import numpy as _np
    import matplotlib.pyplot as _plt

    def plot_gaussian(sigma=2, kind='surface', cmap='viridis', linewidth=1, **kwargs):
        '''Generates a 3D matplotlib plot of a Gaussian distribution'''
        mean = 0
        domain = 10
        x = _np.arange(-domain + mean, domain + mean, sigma / 10)
        y = _np.arange(-domain + mean, domain + mean, sigma / 10)
        x, y = _np.meshgrid(x, x)
        r = (x ** 2 + y ** 2) / (2 * sigma ** 2)
        z = 1 / (_np.pi * sigma ** 4) * (1 - r) * _np.exp(-r)

        fig = _plt.figure(figsize=(12, 6))
        ax = _plt.axes(projection='3d')
        if kind == 'wire':
            ax.plot_wireframe(x, y, z, cmap=cmap, linewidth=linewidth, **kwargs)
        elif kind == 'surface':
            ax.plot_surface(x, y, z, cmap=cmap, linewidth=linewidth, **kwargs)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_zlabel('z', fontsize=16)
        _plt.axis('off')
        return fig

    plot_gaussian(kind='surface', linewidth=1)
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
        ---
        ## fmriprep

        Throughout this lab and course, you have frequently heard about [fmriprep](https://fmriprep.readthedocs.io/en/stable/), which is a functional magnetic resonance imaging (fMRI) data preprocessing pipeline that was developed by a team at the [Center for Reproducible Research](http://reproducibility.stanford.edu/) led by Russ Poldrack and Chris Gorgolewski. Fmriprep was designed to provide an easily accessible, state-of-the-art interface that is robust to variations in scan acquisition protocols, requires minimal user input, and provides easily interpretable and comprehensive error and output reporting. Fmriprep performs basic processing steps (coregistration, normalization, unwarping, noise component extraction, segmentation, skullstripping etc.) providing outputs that are ready for data analysis.

        fmriprep was built on top of [nipype](https://nipype.readthedocs.io/en/latest/), which is a tool to build preprocessing pipelines in python using graphs. This provides a completely flexible way to create custom pipelines using any type of software while also facilitating easy parallelization of steps across the pipeline on high performance computing platforms. Nipype is completely flexible, but has a fairly steep learning curve and is best for researchers who have strong opinions about how they want to preprocess their data, or are working with nonstandard data that might require adjusting the preprocessing steps or parameters. In practice, most researchers typically use similar preprocessing steps and do not need to tweak the pipelines very often. In addition, many researchers do not fully understand how each preprocessing step will impact their results and would prefer if somebody else picked suitable defaults based on current best practices in the literature. The fmriprep pipeline uses a combination of tools from well-known software packages, including FSL, ANTs, FreeSurfer and AFNI. This pipeline was designed to provide the best software implementation for each stage of preprocessing, and is quickly being updated as methods evolve and bugs are discovered by a growing user base.

        This tool allows you to easily do the following:

        - Take fMRI data from raw to fully preprocessed form.
        - Implement tools from different software packages.
        - Achieve optimal data processing quality by using the best tools available.
        - Generate preprocessing quality reports, with which the user can easily identify outliers.
        - Receive verbose output concerning the stage of preprocessing for each subject, including meaningful errors.
        - Automate and parallelize processing steps, which provides a significant speed-up from typical linear, manual processing.
        - More information and documentation can be found at [https://fmriprep.readthedocs.io/](https://fmriprep.readthedocs.io/)
        """),
        mo.image(str(IMG_DIR / "fmriprep.png")),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Running fmriprep

    Running fmriprep is a (mostly) trivial process of running a single line in the command line specifying a few choices and locations for the output data. One of the annoying things about older neuroimaging software that was developed by academics is that the packages were developed using many different development environments and on different operating systems (e.g., unix, windows, mac). It can be a nightmare getting some of these packages to install on more modern computing systems. As fmriprep uses many different packages, they have made it much easier to circumvent the time-consuming process of installing many different packages by releasing a [docker container](https://fmriprep.readthedocs.io/en/stable/docker.html) that contains everything you need to run the pipeline.

    If you're interested in running this on your local computer, here is the code you could use to run it in a jupyter notebook, or even better in the command line on a high performance computing environment.

    ```python
    import os
    base_dir = '/Users/lukechang/Dropbox/Dartbrains/Data'
    data_path = os.path.join(base_dir, 'localizer')
    output_path = os.path.join(base_dir, 'preproc')
    work_path = os.path.join(base_dir, 'work')

    sub = 'S01'
    subs = [f'S{x:0>2d}' for x in range(10)]
    for sub in subs:
        !fmriprep-docker {data_path} {output_path} participant --participant_label sub-{sub} --write-graph --fs-no-reconall --notrack --fs-license-file ~/Dropbox/Dartbrains/License/license.txt --work-dir {work_path}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(IMG_DIR, mo):
    mo.vstack([
        mo.md(r"""
        ### Quick primer on High Performance Computing

        We could run fmriprep on our computer, but this could take a long time if we have a lot of participants. Because we have a limited amount of computational resources on our laptops (e.g., cpus, and memory), we would have to run each participant sequentially. For example, if we had 50 participants, it would take 50 times longer to run all participants than a single one.

        Imagine if you had 50 computers and ran each participant separate at the same time in parallel across all of the computers. This would allow us to run 50 participants in the same amount of time as a single participant. This is the basic idea behind high performance computing, which contains a cluster of many computers that have been installed in racks. Below is a picture of what Dartmouth's [Discovery cluster](https://rc.dartmouth.edu/index.php/discovery-overview/) looks like:
        """),
        mo.image(str(IMG_DIR / "hpc.png")),
        mo.md(r"""
        A cluster is simply a collection of nodes. A node can be thought of as an individual computer. Each node contains processors, which encompass multiple cores. Discovery contains 3000+ cores, which is certainly a lot more than your laptop!

        In order to submit a job, you can create a Portable Batch System (PBS) script that sets up the parameters (e.g., how much time you want your script to run, specifying directory to run, etc) and submits your job to a queue.

        **NOTE**: If you end up working in a lab in the future, you will likely need to request access to a system like *discovery* using [this type of link](https://rcweb.dartmouth.edu/accounts/).

        For a detailed walkthrough of running fmriprep on Dartmouth's Discovery cluster — SLURM scripts, data access on Rolando, and environment setup — see the companion tutorial: [Running fMRIPrep on HPC](./fmriprep_on_discovery.md).
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### fmriprep output

    You can see a summary of the operations fmriprep performed by examining the `.html` files in the `derivatives/fmriprep` folder within the `localizer` data directory.

    Spend some time looking at the outputs and feel free to examine multiple subjects. The HTML reports let you visually inspect each preprocessing step — brain extraction quality (red outline on T1), tissue segmentation maps, registration quality (hover to compare spaces), carpet plots showing signal quality over time, and motion traces with framewise displacement.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Limitations of fmriprep

    In general, we recommend using this pipeline if you want a sensible default. Considerable thought has gone into selecting reasonable default parameters and selecting preprocessing steps based on best practices in the field (as determined by the developers). This is not necessarily the case for any of the default settings in any of the more conventional software packages (e.g., spm, fsl, afni, etc).

    However, there is an important tradeoff in using this tool. On the one hand, it's nice in that it is incredibly straightforward to use (one line of code!), has excellent documentation, and is actively being developed to fix bugs and improve the overall functionality. There is also a growing user base to ask questions. [Neurostars](https://neurostars.org/) is an excellent forum to post questions and learn from others. On the other hand, fmriprep is unfortunately in its current state not easily customizable. If you disagree with the developers about the order or specific preprocessing steps, it is very difficult to modify. Future versions will hopefully be more modular and easier to make custom pipelines. If you need this type of customizability we strongly recommend using nipype over fmriprep.

    In practice, it's always a little bit finicky to get everything set up on a particular system. Sometimes you might run into issues with a specific missing file like the [freesurfer license](https://fmriprep.readthedocs.io/en/stable/usage.html#the-freesurfer-license) even if you're not using it. You might also run into issues with the format of the data that might have some conflicts with the [bids-validator](https://github.com/bids-standard/bids-validator). In our experience, there is always some frustration getting this to work, but it's very nice once it's done.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Exercises

    ### Exercise 1. Inspect HTML output of other participants.

    For this exercise, you will need to navigate to the derivatives folder containing the fmriprep preprocessed data `../data/localizer/derivatives/fmriprep` and inspect the HTML output of subjects other than `S01`. Did the preprocessing steps work? Are there any issues with the data that we should be concerned about?
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
