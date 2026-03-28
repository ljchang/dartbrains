import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="Preprocessing")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preprocessing
    *Written by Luke Chang*

    Being able to study brain activity associated with cognitive processes
    in humans is an amazing achievement. However, there is an extraordinary
    amount of noise and very low levels of signal, which makes it difficult
    to make inferences about brain function using BOLD imaging. A critical
    step before any analysis is to remove as much noise as possible. The
    series of steps to remove noise comprise our *neuroimaging data
    **preprocessing** pipeline*.

    In this lab, we will cover:
    - **Image transformations** (rigid body and affine)
    - **Cost functions** for image registration
    - **Head motion correction** (realignment)
    - **Spatial normalization**
    - **Spatial smoothing**
    - **fMRIPrep** automated preprocessing pipeline
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Code.mr_widgets import TransformCubeWidget, CostFunctionWidget, SmoothingWidget

    return CostFunctionWidget, SmoothingWidget, TransformCubeWidget, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Image Transformations

    Neuroimaging data is stored in a NIfTI container with a 3D/4D matrix
    of voxel intensities and an **affine matrix** that maps voxel
    coordinates \((i, j, k)\) to world coordinates \((x, y, z)\):

    $$\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \mathbf{A} \begin{bmatrix} i \\ j \\ k \\ 1 \end{bmatrix}$$

    A **rigid body transformation** has 6 parameters:
    - 3 **translations** (shift in x, y, z)
    - 3 **rotations** (around x, y, z axes)

    A full **affine transformation** adds 6 more (12 total):
    - 3 **scale** factors
    - 3 **shear** parameters

    Use the sliders below to manipulate a 3D cube. The ghost (wireframe)
    shows the original position; the solid red cube shows the transformed
    position. The affine matrix updates in real time.
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
    how well two images are aligned. These measures are called
    **cost functions**.

    A common cost function is the **Sum of Squared Errors (SSE)**:

    $$SSE = \sum_{i} (I_{target}(i) - I_{reference}(i))^2$$

    The goal is to find the transformation parameters that **minimize**
    the cost function. This process is called **optimization**.

    Try to align the target image (blue square) with the reference image
    by adjusting the translation sliders. Watch the SSE drop to zero
    when perfectly aligned!
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
def _(mo):
    mo.md(r"""
    ---
    ## Realignment (Motion Correction)

    It is critically important that each voxel has the same 3D coordinate
    across all time points. Participants move during scanning, which shifts
    the brain between volumes.

    **Realignment** applies a rigid body transformation to each volume to
    align them to a reference (typically the first or mean volume). The
    6 motion parameters (3 translation + 3 rotation) are saved and can be
    used later as confound regressors.

    Below is an example of motion parameters from a localizer scan. Notice
    how even small movements (< 1mm) are tracked and corrected.

    Even after realignment, head motion distorts the magnetic field and
    causes nonlinear signal changes that aren't fully corrected. In
    resting-state analyses, excessive motion can produce spurious
    correlations. Strategies include:
    - **Scrubbing**: removing high-motion volumes
    - **Confound regression**: including motion parameters in the GLM
    - **Framewise displacement** thresholds for subject exclusion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Spatial Normalization

    Each subject's brain is a different size and shape. To compare across
    subjects, we **warp** each brain into a common stereotactic space
    (typically MNI space).

    **Linear normalization** uses a 12-parameter affine transformation.
    **Nonlinear normalization** uses hundreds to thousands of parameters
    to deform the brain with much higher accuracy. The
    [ANTs](http://stnava.github.io/ANTs/) algorithm (diffeomorphic
    registration) is state-of-the-art and is used by fMRIPrep.

    Key steps in the anatomical preprocessing pipeline:
    1. **Brain extraction** (skull stripping)
    2. **Tissue segmentation** (gray matter, white matter, CSF)
    3. **Nonlinear registration** to MNI template
    4. **Functional-to-anatomical** coregistration (typically using
       Boundary Based Registration)
    """)
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
    ---
    ## fMRIPrep

    [fMRIPrep](https://fmriprep.readthedocs.io/en/stable/) is an automated
    fMRI preprocessing pipeline developed by the Center for Reproducible
    Research (Poldrack lab, Stanford). It provides:

    - **State-of-the-art** default parameters based on best practices
    - **Minimal user input** -- one command line call
    - **Comprehensive QC reports** (HTML output with visual checks)
    - **BIDS compatibility** -- reads standard BIDS-formatted data
    - Built on [Nipype](https://nipype.readthedocs.io/) for reproducible workflows

    ### Running fMRIPrep

    fMRIPrep is distributed as a Docker/Singularity container that bundles
    all dependencies (FSL, FreeSurfer, ANTs, etc.):

    ```bash
    fmriprep /path/to/bids_dir /path/to/output_dir participant \
        --participant-label sub-01 \
        --fs-license-file /path/to/license.txt \
        --output-spaces MNI152NLin2009cAsym
    ```

    ### Key Preprocessing Steps

    fMRIPrep performs (in roughly this order):
    1. **Anatomical**: Brain extraction, tissue segmentation, surface
       reconstruction (FreeSurfer), spatial normalization (ANTs)
    2. **Functional**: Reference volume estimation, head motion correction,
       slice-timing correction, susceptibility distortion correction,
       coregistration to anatomical, confound estimation (aCompCor,
       framewise displacement, DVARS)

    ### fMRIPrep Output

    The HTML reports let you visually inspect each preprocessing step:
    - Brain extraction quality (red outline on T1)
    - Tissue segmentation maps
    - Registration quality (hover to compare spaces)
    - Carpet plots showing signal quality over time
    - Motion traces and framewise displacement

    ### Limitations

    - Not easily customizable (opinionated defaults)
    - May not be optimal for special populations (infants, patients
      with large lesions)
    - Requires BIDS-formatted input
    - Computationally expensive (~6-12 hours per subject)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    | Step | What it does | Key parameters |
    |------|-------------|----------------|
    | **Realignment** | Corrects head motion between volumes | 6 DOF rigid body |
    | **Slice timing** | Corrects for sequential slice acquisition | Reference slice |
    | **Coregistration** | Aligns functional to anatomical | BBR cost function |
    | **Normalization** | Warps to standard space (MNI) | Nonlinear (ANTs) |
    | **Smoothing** | Blurs image to increase SNR | FWHM (typically 6mm) |

    **Key principles:**
    - Each resampling step introduces interpolation error -- minimize the
      number of separate resampling steps
    - fMRIPrep combines multiple transforms into a single resampling step
    - Always inspect QC reports before trusting preprocessed data
    - Motion is the #1 enemy of fMRI -- no amount of preprocessing fully
      removes its effects

    **Next:** The preprocessed data is now ready for statistical analysis
    using the General Linear Model (GLM).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
