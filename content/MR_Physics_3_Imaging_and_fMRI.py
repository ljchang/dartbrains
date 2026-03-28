import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="MR Physics 3: Imaging & fMRI")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MR Physics 3: Imaging & fMRI

    *Written by Luke Chang*

    We now know how to generate a signal (Notebook 1) and how different
    tissues produce different signals (Notebook 2). But there's a critical
    problem: **where did the signal come from?**

    If the entire volume inside the scanner experiences the same \(B_0\)
    field, all protons precess at the same frequency, and we can't tell
    whether the signal is coming from your frontal cortex or your cerebellum.
    We need **spatial encoding** -- a way to tag different locations with
    different frequencies or phases so we can reconstruct an image.

    This notebook covers:
    - How **gradients** create spatial encoding
    - How **k-space** and the Fourier transform produce images
    - How the **BOLD signal** lets us measure brain activity
    - How **fMRI pulse sequences** (EPI) work

    This is **Part 3** of a three-notebook series:
    1. Magnetism & Resonance
    2. Signal & Contrast
    3. **Imaging & fMRI** (this notebook)
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Code.mr_simulations import (
        GAMMA_H, TISSUE_PROPERTIES,
        spin_echo_signal, gradient_echo_signal,
        hrf, image_to_kspace, kspace_to_image, mask_kspace,
        plot_kspace_and_image, plot_pulse_sequence, plot_contrast_bars,
    )
    from Code.mr_widgets import EncodingWidget, KSpaceWidget, ConvolutionWidget

    return (
        ConvolutionWidget,
        EncodingWidget,
        GAMMA_H,
        KSpaceWidget,
        go,
        hrf,
        make_subplots,
        mo,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Interactive version:** [Open this notebook in molab](https://molab.marimo.io/github/ljchang/dartbrains/blob/v2-marimo-migration/content/MR_Physics_3_Imaging_and_fMRI.py) to run code, interact with widgets, and modify examples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. The Localization Problem: Why We Need Gradients

    In a perfectly uniform \(B_0\) field, every proton in the body
    precesses at the same Larmor frequency. Our receiver coil picks up
    the *sum* of all these signals, with no way to tell where each
    contribution came from.

    The solution: **gradient coils**. These are additional electromagnetic
    coils that make the magnetic field vary *linearly* across space. If
    we add a gradient along the x-axis:

    $$B(x) = B_0 + G_x \cdot x$$

    then the Larmor frequency also varies with position:

    $$f(x) = \gamma \cdot (B_0 + G_x \cdot x)$$

    Now protons at different x-positions precess at different frequencies
    -- and we can separate them using the Fourier transform!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    grad_strength_slider = mo.ui.slider(
        start=0.0, stop=40.0, step=1.0, value=0.0,
        label="Gradient strength Gx (mT/m)",
        full_width=True,
    )
    return (grad_strength_slider,)


@app.cell(hide_code=True)
def _(GAMMA_H, go, grad_strength_slider, make_subplots, mo, np):
    _gx = grad_strength_slider.value  # mT/m
    _b0 = 3.0  # Tesla
    _positions = np.linspace(-0.15, 0.15, 200)  # meters (30cm FOV)

    # B field at each position
    _b_field = _b0 + (_gx / 1000) * _positions  # Convert mT/m to T/m
    _freq = GAMMA_H * _b_field  # MHz

    _fig = make_subplots(rows=1, cols=2,
                          subplot_titles=["Magnetic Field B(x)", "Larmor Frequency f(x)"])

    # B field
    _fig.add_trace(go.Scatter(
        x=_positions * 100, y=_b_field,
        mode="lines", line=dict(color="#636EFA", width=3),
        name="B(x)",
    ), row=1, col=1)

    # Frequency
    _fig.add_trace(go.Scatter(
        x=_positions * 100, y=_freq,
        mode="lines", line=dict(color="#EF553B", width=3),
        name="f(x)",
    ), row=1, col=2)

    _fig.update_xaxes(title_text="Position (cm)", row=1, col=1)
    _fig.update_xaxes(title_text="Position (cm)", row=1, col=2)
    _fig.update_yaxes(title_text="B (Tesla)", row=1, col=1)
    _fig.update_yaxes(title_text="Frequency (MHz)", row=1, col=2)

    _freq_range = _freq[-1] - _freq[0]
    _fig.update_layout(
        width=800, height=350, margin=dict(l=60, r=20, t=40, b=40),
        yaxis=dict(range=[2.99, 3.01]),
        yaxis2=dict(range=[127.3, 128.2]),
    )

    mo.vstack([
        mo.hstack([grad_strength_slider], justify="start", gap=2),
        _fig,
        mo.md(
            f"**Gradient: {_gx:.0f} mT/m** → Frequency spread across 30cm FOV: "
            f"**{abs(_freq_range * 1000):.1f} kHz** "
            f"({'uniform field, no spatial encoding!' if _gx == 0 else 'spatial encoding active'})"
        ),
        mo.callout(
            mo.md(
                "With **no gradient** (Gx=0), the field and frequency are the same "
                "everywhere -- no spatial information. As you increase the gradient, "
                "frequency varies across space, and the Fourier transform of the signal "
                "directly maps to spatial position!"
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Slice Selection

    MRI typically images one **slice** at a time. To select a specific
    slice, we apply a gradient along the z-axis (head-to-foot) during
    the RF pulse. The RF pulse has a limited bandwidth -- it only excites
    protons within a narrow range of frequencies. Since the z-gradient
    makes frequency vary with position, only protons in one slice
    are on-resonance and get excited.

    **Slice thickness** is controlled by:
    $$\Delta z = \frac{\Delta f}{\gamma \cdot G_z}$$

    where \(\Delta f\) is the RF pulse bandwidth and \(G_z\) is the
    slice-select gradient strength. Stronger gradient = thinner slice.
    Wider RF bandwidth = thicker slice.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ss_grad_slider = mo.ui.slider(
        start=5, stop=40, step=1, value=20,
        label="Slice-select gradient Gz (mT/m)",
    )
    ss_bw_slider = mo.ui.slider(
        start=500, stop=5000, step=100, value=2000,
        label="RF bandwidth (Hz)",
    )
    return ss_bw_slider, ss_grad_slider


@app.cell(hide_code=True)
def _(GAMMA_H, go, mo, np, ss_bw_slider, ss_grad_slider):
    _gz = ss_grad_slider.value / 1000  # T/m
    _bw = ss_bw_slider.value  # Hz
    _b0 = 3.0

    # Slice thickness in mm
    _slice_thickness = (_bw / (GAMMA_H * 1e6 * _gz)) * 1000  # mm

    # Positions along z
    _z_pos = np.linspace(-100, 100, 500)  # mm
    _freq_hz = GAMMA_H * 1e6 * (_b0 + _gz * _z_pos / 1000) - GAMMA_H * 1e6 * _b0  # Hz offset from center

    # RF excitation profile (sinc-like bandwidth)
    _center_freq = 0  # Hz (excite at isocenter)
    _excited = np.abs(_freq_hz - _center_freq) < (_bw / 2)

    _fig = go.Figure()

    # Frequency vs position
    _fig.add_trace(go.Scatter(
        x=_z_pos, y=_freq_hz / 1000,
        mode="lines", line=dict(color="#636EFA", width=2),
        name="Frequency offset",
    ))

    # Excited region
    _fig.add_trace(go.Scatter(
        x=_z_pos[_excited], y=_freq_hz[_excited] / 1000,
        mode="lines", line=dict(color="red", width=4),
        name="Excited slice",
    ))

    # RF bandwidth band
    _fig.add_hrect(y0=-_bw / 2000, y1=_bw / 2000,
                    fillcolor="rgba(255, 0, 0, 0.1)", line_width=0,
                    annotation_text="RF bandwidth")

    _fig.update_layout(
        title=f"Slice Selection: thickness = {_slice_thickness:.1f} mm",
        xaxis_title="Position along z (mm)",
        yaxis_title="Frequency offset (kHz)",
        yaxis=dict(range=[-5, 5]),
        width=700, height=400,
        margin=dict(l=60, r=20, t=40, b=40),
    )

    mo.vstack([
        mo.hstack([ss_grad_slider, ss_bw_slider], justify="start", gap=2),
        _fig,
        mo.md(
            f"With Gz = **{_gz * 1000:.0f} mT/m** and RF bandwidth = **{_bw} Hz**, "
            f"the selected slice is **{_slice_thickness:.1f} mm** thick.\n\n"
            "The red segment shows the positions where protons are within the RF bandwidth "
            "and will be excited. All other protons are off-resonance and unaffected."
        ),
        mo.callout(
            mo.md(
                "**Try this:** Increase the gradient strength -- the slice gets thinner "
                "(better resolution but less signal). Increase the RF bandwidth -- the "
                "slice gets thicker (more signal but lower resolution). This is a "
                "fundamental tradeoff in MRI."
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Frequency & Phase Encoding

    After selecting a slice, we still need to encode position within
    that 2D slice. MRI uses two clever tricks:

    **Frequency encoding (readout direction, x):** During signal readout,
    a gradient along x makes protons at different x-positions precess at
    different frequencies. The Fourier transform of the readout signal
    directly gives us the spatial profile along x.

    **Phase encoding (y-direction):** Before readout, a brief gradient
    pulse along y gives protons at different y-positions different *phases*.
    This is repeated many times with different gradient strengths to fill
    out the y-dimension. Each repetition fills one line of k-space.

    The combination gives each voxel a unique (frequency, phase) signature
    that can be decoded by a 2D Fourier transform.
    """)
    return


@app.cell(hide_code=True)
def _(EncodingWidget, mo):
    _widget = EncodingWidget(speed=1.0)
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        _wrapped,
        mo.md(
            "Watch how the spin arrows in each voxel respond to gradients:\n"
            "- **No gradients**: all spins precess at the same rate\n"
            "- **Frequency encoding (Gx)**: columns spin at different rates\n"
            "- **Phase encoding (Gy)**: rows accumulate different phase offsets\n"
            "- **Both**: each voxel gets a unique (frequency, phase) signature\n\n"
            "The 2D Fourier transform decodes these signatures back into an image."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Deep Dive: The Fourier encoding equations": mo.md(
                r"""
                The MRI signal at time \(t\) during readout, for a given
                phase-encode step with gradient area \(A_y\), is:

                $$S(t) = \int\int \rho(x, y) \cdot e^{-i 2\pi (\gamma G_x x t + \gamma A_y y)} \, dx \, dy$$

                where \(\rho(x, y)\) is the spin density (our image).

                If we define spatial frequencies:
                - \(k_x = \gamma G_x t\) (varies during readout)
                - \(k_y = \gamma A_y\) (set by phase-encode gradient)

                then the signal equation becomes:

                $$S(k_x, k_y) = \int\int \rho(x, y) \cdot e^{-i 2\pi (k_x x + k_y y)} \, dx \, dy$$

                This is exactly the **2D Fourier transform** of the image!
                Therefore, the image is simply the inverse Fourier transform
                of the acquired data:

                $$\rho(x, y) = \mathcal{F}^{-1}\{S(k_x, k_y)\}$$

                The space of \((k_x, k_y)\) is called **k-space**.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. K-Space: Where MRI Data Lives

    The raw data from an MRI scan doesn't look like an image -- it lives
    in **k-space**, the spatial frequency domain. K-space and the image
    are related by the 2D Fourier transform:

    $$\text{Image} = \text{FFT}^{-1}(\text{K-space})$$

    Different regions of k-space encode different features:
    - **Center of k-space** → overall contrast, brightness, low-frequency
      shapes
    - **Edges of k-space** → fine details, sharp edges, high spatial
      frequency

    This means you don't need ALL of k-space to get a useful image.
    Keeping only the center gives you a blurry but recognizable image.
    Keeping only the edges gives you an edge map with no contrast.

    Try masking different regions below to build intuition!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    kspace_mask_type = mo.ui.dropdown(
        options={
            "Progressive fill (animated)": "progressive",
            "Center only (low frequencies)": "center",
            "Periphery only (high frequencies)": "periphery",
            "Every 4th line (undersampled)": "undersampled",
        },
        value="progressive",
        label="K-space mode",
    )
    kspace_radius = mo.ui.slider(
        start=0.05, stop=0.5, step=0.05, value=0.2,
        label="Mask radius",
    )
    kspace_speed = mo.ui.slider(
        start=0.5, stop=5.0, step=0.5, value=2.0,
        label="Fill speed",
    )
    return kspace_mask_type, kspace_radius, kspace_speed


@app.cell(hide_code=True)
def _(KSpaceWidget, kspace_mask_type, kspace_radius, kspace_speed, mo):
    _widget = KSpaceWidget(
        mask_type=kspace_mask_type.value,
        radius_fraction=float(kspace_radius.value),
        speed=float(kspace_speed.value),
    )
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        mo.hstack([kspace_mask_type, kspace_radius, kspace_speed], justify="start", gap=2),
        _wrapped,
        mo.md(
            "**Progressive fill** shows how an MRI scanner acquires k-space line by line, "
            "with the image emerging gradually. Try other modes:\n"
            "- **Center only**: Blurry but recognizable -- contrast lives in the center\n"
            "- **Periphery only**: Only edges visible -- detail lives at the edges\n"
            "- **Undersampled**: Aliasing artifacts from skipping lines"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. BOLD & the Hemodynamic Response

    Now we arrive at **functional MRI (fMRI)** -- using MRI to measure
    brain *activity*. But MRI doesn't measure neural activity directly.
    Instead, it measures a downstream consequence: changes in **blood
    oxygenation**.

    The key insight: **deoxyhemoglobin is paramagnetic** (slightly magnetic),
    while **oxyhemoglobin is diamagnetic** (not magnetic). When a brain
    region becomes active:

    1. Neurons fire and consume oxygen locally
    2. The vascular system responds by increasing blood flow to that region
    3. Blood flow *overshoots* metabolic demand -- more oxygen arrives
       than is consumed
    4. The result: less deoxyhemoglobin locally
    5. Less deoxyhemoglobin → less local field distortion → **increased
       T₂* signal**

    This is the **Blood Oxygen Level Dependent (BOLD)** signal. The
    temporal shape of the BOLD response to a brief neural event is called
    the **Hemodynamic Response Function (HRF)**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    hrf_peak_slider = mo.ui.slider(
        start=3.0, stop=9.0, step=0.5, value=6.0,
        label="Peak time (s)",
    )
    hrf_undershoot_slider = mo.ui.slider(
        start=10.0, stop=25.0, step=1.0, value=16.0,
        label="Undershoot time (s)",
    )
    hrf_ratio_slider = mo.ui.slider(
        start=0.0, stop=0.5, step=0.05, value=0.167,
        label="Undershoot ratio",
    )
    return hrf_peak_slider, hrf_ratio_slider, hrf_undershoot_slider


@app.cell(hide_code=True)
def _(
    go,
    hrf,
    hrf_peak_slider,
    hrf_ratio_slider,
    hrf_undershoot_slider,
    mo,
    np,
):
    _peak = hrf_peak_slider.value
    _undershoot = hrf_undershoot_slider.value
    _ratio = hrf_ratio_slider.value

    _t = np.linspace(0, 30, 300)
    _h = hrf(_t, peak_time=_peak, undershoot_time=_undershoot,
             undershoot_ratio=_ratio)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_t, y=_h, mode="lines",
        line=dict(color="#636EFA", width=3), name="HRF",
    ))
    _fig.add_hline(y=0, line_dash="dot", line_color="gray")

    # Annotate key features
    _peak_idx = np.argmax(_h)
    _fig.add_annotation(x=_t[_peak_idx], y=_h[_peak_idx],
                         text=f"Peak (~{_t[_peak_idx]:.1f}s)",
                         arrowhead=2, ay=-40)

    if _ratio > 0:
        _undershoot_idx = np.argmin(_h[_peak_idx:]) + _peak_idx
        if _h[_undershoot_idx] < 0:
            _fig.add_annotation(x=_t[_undershoot_idx], y=_h[_undershoot_idx],
                                 text=f"Undershoot (~{_t[_undershoot_idx]:.1f}s)",
                                 arrowhead=2, ay=40)

    _fig.update_layout(
        title="Hemodynamic Response Function (HRF)",
        xaxis_title="Time after stimulus (s)",
        yaxis_title="BOLD signal change (a.u.)",
        yaxis=dict(range=[-0.3, 1.1]),
        width=700, height=350,
        margin=dict(l=60, r=20, t=40, b=40),
    )

    mo.vstack([
        mo.hstack([hrf_peak_slider, hrf_undershoot_slider, hrf_ratio_slider], justify="start", gap=2),
        _fig,
        mo.md(
            "The canonical HRF has several notable features:\n"
            f"- **Initial dip** (small, often not detectable at typical fMRI resolution)\n"
            f"- **Peak** at ~{_peak:.0f}s after stimulus\n"
            f"- **Post-stimulus undershoot** at ~{_undershoot:.0f}s\n"
            "- Returns to baseline after ~25-30s\n\n"
            "This sluggish hemodynamic response means fMRI has poor **temporal resolution** "
            "(~seconds) compared to EEG (milliseconds), even though the neural events "
            "happen on the millisecond timescale."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convolution: From Events to Predicted BOLD Signal

    In an fMRI experiment, we present multiple stimuli over time. The
    predicted BOLD signal is the **convolution** of the stimulus timing
    with the HRF:

    $$\text{Predicted BOLD}(t) = \text{stimulus}(t) * \text{HRF}(t)$$

    This is the foundation of the **General Linear Model (GLM)** used
    in fMRI analysis (covered in detail in later Dartbrains chapters).

    Try adjusting the stimulus timing below to see how the predicted BOLD
    signal changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    stim_pattern = mo.ui.dropdown(
        options={
            "Single event": "single",
            "3 events (spaced)": "spaced",
            "Block design (10s on/off)": "block",
            "Rapid event-related": "rapid",
        },
        value="single",
        label="Stimulus pattern",
    )
    conv_speed = mo.ui.slider(
        start=0.5, stop=4.0, step=0.5, value=1.5,
        label="Animation speed",
    )
    return conv_speed, stim_pattern


@app.cell(hide_code=True)
def _(ConvolutionWidget, conv_speed, mo, stim_pattern):
    _widget = ConvolutionWidget(
        pattern=stim_pattern.value,
        speed=float(conv_speed.value),
    )
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        mo.hstack([stim_pattern, conv_speed], justify="start", gap=2),
        _wrapped,
        mo.callout(
            mo.md(
                "Watch how each stimulus event spawns its own HRF response (purple ghost curves), "
                "and the predicted BOLD signal (blue) is their sum. In block designs, the BOLD "
                "builds up during ON periods. In rapid event-related designs, individual HRFs "
                "overlap -- this is **linear superposition**."
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. fMRI Pulse Sequences: Echo Planar Imaging (EPI)

    Standard MRI acquires one line of k-space per TR -- far too slow for
    fMRI, which needs to image the whole brain every 1-2 seconds.

    The solution is **Echo Planar Imaging (EPI)**: after a single RF
    excitation, rapidly oscillating gradients traverse *all* of k-space
    in about 50-100 milliseconds. This is the workhorse sequence for fMRI.

    **Typical fMRI parameters:**

    | Parameter | Structural (T₁w MPRAGE) | Functional (GRE-EPI) |
    |-----------|------------------------|---------------------|
    | Weighting | T₁ | T₂* |
    | TR | ~2000 ms | 500-2000 ms |
    | TE | ~3 ms | ~30 ms |
    | Flip angle | ~9° | ~70-90° |
    | Resolution | ~1 mm³ | ~2-3 mm³ |
    | Volumes | 1 (whole brain) | 100s-1000s |
    | Duration | ~5 min | 5-60 min |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    epi_tr_slider = mo.ui.slider(
        start=500, stop=3000, step=100, value=1000,
        label="TR (ms)",
    )
    epi_te_slider = mo.ui.slider(
        start=15, stop=50, step=5, value=30,
        label="TE (ms)",
    )
    epi_matrix_slider = mo.ui.slider(
        start=32, stop=128, step=16, value=64,
        label="Matrix size",
    )
    return epi_matrix_slider, epi_te_slider, epi_tr_slider


@app.cell(hide_code=True)
def _(epi_matrix_slider, epi_te_slider, epi_tr_slider, go, mo):
    _tr = epi_tr_slider.value
    _te = epi_te_slider.value
    _matrix = epi_matrix_slider.value

    # Calculate derived parameters
    _fov = 220  # mm (typical brain FOV)
    _voxel_size = _fov / _matrix
    _readout_time = _matrix * 0.5  # ~0.5ms per readout line (simplified)
    _n_slices = int(_tr / (_readout_time + 5))  # rough estimate

    # Tradeoff visualization
    _metrics = {
        "Spatial Resolution": f"{_voxel_size:.1f} mm",
        "Temporal Resolution": f"{_tr/1000:.1f} s",
        "Slices per TR": f"~{_n_slices}",
        "BOLD sensitivity": f"{'Good' if 25 <= _te <= 35 else 'Suboptimal'} (TE={_te}ms)",
        "Readout duration": f"~{_readout_time:.0f} ms",
    }

    # K-space trajectory: line density reflects matrix size
    _fig = go.Figure()
    _n_lines = _matrix
    _kx_all = []
    _ky_all = []
    _colors = []
    for _line in range(_n_lines):
        _y = _line / _n_lines - 0.5
        if _line % 2 == 0:
            _kx_all.extend([-0.5, 0.5, None])
        else:
            _kx_all.extend([0.5, -0.5, None])
        _ky_all.extend([_y, _y, None])

    _fig.add_trace(go.Scatter(
        x=_kx_all, y=_ky_all, mode="lines",
        line=dict(color="#636EFA", width=max(0.5, 3 - _n_lines / 40)),
        name=f"{_matrix} PE lines",
    ))
    _fig.update_layout(
        title=f"EPI Trajectory ({_matrix} lines)",
        xaxis_title="kx", yaxis_title="ky",
        xaxis=dict(range=[-0.6, 0.6]),
        yaxis=dict(range=[-0.6, 0.6], scaleanchor="x"),
        width=350, height=350,
        margin=dict(l=50, r=10, t=35, b=40),
    )

    # Parameters table
    _table_md = "| Parameter | Value |\n|-----------|-------|\n"
    for _k, _v in _metrics.items():
        _table_md += f"| {_k} | {_v} |\n"

    mo.vstack([
        mo.hstack([epi_tr_slider, epi_te_slider, epi_matrix_slider], justify="start", gap=2),
        mo.hstack([_fig, mo.md(_table_md)], justify="center"),
        mo.callout(
            mo.md(
                "**Key tradeoffs in fMRI:**\n"
                "- **Larger matrix** → better spatial resolution but slower readout, "
                "more distortion\n"
                "- **Shorter TR** → better temporal resolution but fewer slices\n"
                "- **TE ~30ms at 3T** → optimal BOLD sensitivity (matches T₂* of "
                "gray matter ~40-50ms)\n"
                "- Modern **multiband/simultaneous multi-slice** techniques can "
                "excite multiple slices at once, allowing sub-second whole-brain "
                "TR with good resolution"
            ),
            kind="warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. Putting It All Together: From Proton to Activation Map

    Let's trace the complete chain from physics to functional brain imaging:

    ```
    Strong magnetic field (B₀)
        │
        ▼
    Protons align → net magnetization (M₀)
        │
        ▼
    RF pulse at Larmor frequency → tips M into transverse plane
        │
        ▼
    Precessing Mxy induces signal in coil (FID)
        │
        ▼
    Gradients encode spatial position
        │
        ▼
    Raw signal fills k-space
        │
        ▼
    2D Inverse FFT → image for each slice
        │
        ▼
    Repeat with T₂*-weighted GRE-EPI every ~1s
        │
        ▼
    BOLD signal changes reflect neural activity (via hemodynamics)
        │
        ▼
    Statistical analysis (GLM) → activation maps
    ```

    **What we gain:**
    - Non-invasive measurement of brain activity
    - Whole-brain coverage
    - Reasonable spatial resolution (~2-3mm)
    - No ionizing radiation (unlike PET/CT)

    **What we lose / must be aware of:**
    - Temporal resolution is limited (~seconds, not milliseconds)
    - BOLD is an *indirect* measure of neural activity
    - Signal can be affected by head motion, physiological noise,
      susceptibility artifacts (especially near sinuses and ear canals)
    - The hemodynamic response varies across brain regions and individuals

    These limitations -- and how to address them -- are covered in the
    subsequent Dartbrains chapters on preprocessing, the GLM, and
    statistical analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary of the Three-Notebook Series

    | Notebook | Key Concepts |
    |----------|-------------|
    | **1. Magnetism & Resonance** | B₀ alignment, precession, Larmor equation, RF excitation, FID |
    | **2. Signal & Contrast** | T₁/T₂ relaxation, Bloch equations, TE/TR contrast, spin echo vs gradient echo |
    | **3. Imaging & fMRI** | Gradients, spatial encoding, k-space, BOLD signal, HRF, EPI |

    These concepts form the foundation for everything else in neuroimaging.
    Understanding *why* the signal looks the way it does -- and *what can go
    wrong* -- is essential for designing good experiments, preprocessing data
    correctly, and interpreting results with appropriate caution.

    **Continue your learning:** The subsequent Dartbrains chapters cover
    signal processing, preprocessing with fMRIPrep, the General Linear Model,
    group analysis, and multivariate pattern analysis.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
