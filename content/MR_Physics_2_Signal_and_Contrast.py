import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="MR Physics 2: Signal & Contrast")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MR Physics 2: Signal & Contrast

    *Written by Luke Chang*

    In the previous notebook, we learned how an RF pulse tips the net
    magnetization into the transverse plane, producing a Free Induction
    Decay (FID) signal. But **why does the signal decay?** And how do
    MRI scanners produce images where gray matter, white matter, and CSF
    all look different?

    The answers lie in **relaxation** -- the processes by which the
    magnetization returns to equilibrium after excitation. Different
    tissues relax at different rates, and MRI pulse sequences exploit
    these differences to create **contrast**.

    This is **Part 2** of a three-notebook series:
    1. Magnetism & Resonance
    2. **Signal & Contrast** (this notebook)
    3. Imaging & fMRI
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
        TISSUE_PROPERTIES, GAMMA_H,
        t1_recovery, t2_decay,
        spin_echo_signal, gradient_echo_signal,
        apply_rf_pulse, apply_relaxation, simulate_bloch,
        plot_signal_timeline, plot_contrast_bars, plot_pulse_sequence,
    )
    from Code.mr_widgets import PrecessionWidget, SpinEnsembleWidget

    return (


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Interactive version:** [Open this notebook in molab](https://molab.marimo.io/github/ljchang/dartbrains/blob/v2-marimo-migration/content/MR_Physics_2_Signal_and_Contrast.py) to run code, interact with widgets, and modify examples.
    """)
    return
        PrecessionWidget,
        SpinEnsembleWidget,
        TISSUE_PROPERTIES,
        go,
        make_subplots,
        mo,
        np,
        plot_contrast_bars,
        spin_echo_signal,
        t1_recovery,
        t2_decay,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. T1 Relaxation (Spin-Lattice Recovery)

    After a 90° RF pulse tips all the magnetization into the xy-plane,
    the longitudinal component \(M_z\) is zero. Over time, \(M_z\)
    **recovers** back to its equilibrium value \(M_0\) through a process
    called **T1 relaxation** (also known as spin-lattice relaxation).

    This recovery follows an exponential curve:

    $$M_z(t) = M_0 \left(1 - e^{-t/T_1}\right)$$

    The **T1 time constant** is the time it takes for \(M_z\) to recover
    to about 63% of \(M_0\). Crucially, **different tissues have different
    T1 values**:
    - **Fat** has a short T1 (fast recovery) because fat molecules are
      large and tumble slowly, efficiently transferring energy to the lattice
    - **CSF** has a long T1 (slow recovery) because water molecules are
      small and tumble quickly, making energy transfer less efficient
    - **Gray and white matter** fall in between
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    t1_slider = mo.ui.slider(
        start=100, stop=5000, step=100, value=1300,
        label="T₁ (ms)",
        full_width=True,
    )
    t1_field_select = mo.ui.dropdown(
        options={"1.5T": "1.5T", "3T": "3T"},
        value="3T",
        label="Field strength",
    )
    t1_show_tissues = mo.ui.switch(label="Show tissue curves", value=True)
    return t1_field_select, t1_show_tissues, t1_slider


@app.cell(hide_code=True)
def _(
    TISSUE_PROPERTIES,
    go,
    mo,
    np,
    t1_field_select,
    t1_recovery,
    t1_show_tissues,
    t1_slider,
):
    _t1_custom = t1_slider.value
    _field = t1_field_select.value
    _show_tissues = t1_show_tissues.value
    _t = np.linspace(0, 8000, 500)

    _fig = go.Figure()

    # Custom T1 curve
    _recovery = t1_recovery(_t, _t1_custom)
    _fig.add_trace(go.Scatter(
        x=_t, y=_recovery, mode="lines",
        line=dict(color="#636EFA", width=3),
        name=f"Custom T₁ = {_t1_custom} ms",
    ))

    # 63% line
    _fig.add_hline(y=0.63, line_dash="dot", line_color="gray",
                   annotation_text="63% recovery (t = T₁)")

    if _show_tissues:
        _tissue_colors = {
            "Gray Matter": "#808080", "White Matter": "#C8A882",
            "CSF": "#4169E1", "Fat": "#FFD700", "Muscle": "#CD5C5C"
        }
        for _tissue, _props in TISSUE_PROPERTIES[_field].items():
            _r = t1_recovery(_t, _props["T1"])
            _fig.add_trace(go.Scatter(
                x=_t, y=_r, mode="lines",
                line=dict(color=_tissue_colors.get(_tissue, "gray"), width=2, dash="dash"),
                name=f"{_tissue} (T₁={_props['T1']} ms)",
            ))

    _fig.update_layout(
        title=f"T₁ Recovery at {_field}",
        xaxis_title="Time after excitation (ms)",
        yaxis_title="Mz / M₀",
        yaxis=dict(range=[0, 1.05]),
        width=750, height=400,
        margin=dict(l=60, r=20, t=40, b=40),
    )

    mo.vstack([
        mo.hstack([t1_slider, t1_field_select, t1_show_tissues], justify="start", gap=2),
        _fig,
        mo.callout(
            mo.md(
                "Notice how **fat** (short T₁) recovers quickly while **CSF** "
                "(long T₁) takes much longer. Gray and white matter are in between "
                "but have *different* T₁ values -- this difference is what creates "
                "**T₁-weighted contrast** in MRI images."
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Deep Dive: The physics of T₁ relaxation": mo.md(
                r"""
                T₁ relaxation describes the return of longitudinal magnetization
                to equilibrium. The recovery follows the Bloch equation for \(M_z\):

                $$\frac{dM_z}{dt} = \frac{M_0 - M_z}{T_1}$$

                with the solution (after a 90° excitation):

                $$M_z(t) = M_0 \left(1 - e^{-t/T_1}\right)$$

                The physical mechanism involves energy exchange between the
                excited spin system and the surrounding molecular lattice
                (hence "spin-lattice" relaxation). The efficiency depends on
                molecular tumbling rates:

                - **Optimal T₁ relaxation** occurs when the molecular tumbling
                  frequency matches the Larmor frequency
                - **Small, fast-tumbling molecules** (like free water) are
                  inefficient → long T₁
                - **Large, slow-tumbling molecules** (like fat) are more
                  efficient → short T₁
                - T₁ **increases with field strength** because the Larmor
                  frequency moves further from typical tumbling frequencies
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. T2 Relaxation (Spin-Spin Decay)

    While \(M_z\) is recovering along the z-axis, the transverse
    magnetization \(M_{xy}\) is **decaying** in the xy-plane. This happens
    because individual proton spins gradually lose phase coherence --
    they precess at slightly different frequencies due to interactions
    with neighboring spins. This is **T2 relaxation** (spin-spin relaxation):

    $$M_{xy}(t) = M_{xy}(0) \cdot e^{-t/T_2}$$

    In practice, additional dephasing from \(B_0\) field inhomogeneities
    makes the signal decay even faster, characterized by \(T_2^*\):

    $$\frac{1}{T_2^*} = \frac{1}{T_2} + \frac{1}{T_2'}$$

    where \(T_2'\) represents dephasing from field inhomogeneities.
    **T2 is always shorter than or equal to T1** -- the transverse signal
    always decays before the longitudinal signal fully recovers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    t2_slider = mo.ui.slider(
        start=10, stop=300, step=10, value=80,
        label="T₂ (ms)",
        full_width=True,
    )
    t2prime_slider = mo.ui.slider(
        start=10, stop=500, step=10, value=100,
        label="T₂' (ms) — field inhomogeneity",
        full_width=True,
    )
    t2_show_tissues = mo.ui.switch(label="Show tissue curves", value=True)
    t2_field_select = mo.ui.dropdown(
        options={"1.5T": "1.5T", "3T": "3T"}, value="3T", label="Field strength",
    )
    return t2_field_select, t2_show_tissues, t2_slider, t2prime_slider


@app.cell(hide_code=True)
def _(
    TISSUE_PROPERTIES,
    go,
    mo,
    np,
    t2_decay,
    t2_field_select,
    t2_show_tissues,
    t2_slider,
    t2prime_slider,
):
    _t2_custom = t2_slider.value
    _t2prime = t2prime_slider.value
    _field = t2_field_select.value
    _t2star = 1.0 / (1.0 / _t2_custom + 1.0 / _t2prime)
    _t = np.linspace(0, 500, 500)

    _fig = go.Figure()

    # T2 decay
    _fig.add_trace(go.Scatter(
        x=_t, y=t2_decay(_t, _t2_custom), mode="lines",
        line=dict(color="#636EFA", width=3),
        name=f"T₂ = {_t2_custom} ms",
    ))
    # T2* decay
    _fig.add_trace(go.Scatter(
        x=_t, y=t2_decay(_t, _t2star), mode="lines",
        line=dict(color="#EF553B", width=3, dash="dash"),
        name=f"T₂* = {_t2star:.0f} ms",
    ))

    # 37% line (1/e)
    _fig.add_hline(y=0.37, line_dash="dot", line_color="gray",
                   annotation_text="37% remaining (t = T₂)")

    if t2_show_tissues.value:
        _tissue_colors = {
            "Gray Matter": "#808080", "White Matter": "#C8A882",
            "CSF": "#4169E1", "Fat": "#FFD700", "Muscle": "#CD5C5C"
        }
        for _tissue, _props in TISSUE_PROPERTIES[_field].items():
            _d = t2_decay(_t, _props["T2"])
            _fig.add_trace(go.Scatter(
                x=_t, y=_d, mode="lines",
                line=dict(color=_tissue_colors.get(_tissue, "gray"), width=2, dash="dot"),
                name=f"{_tissue} (T₂={_props['T2']} ms)",
            ))

    _fig.update_layout(
        title=f"T₂ and T₂* Decay at {_field}",
        xaxis_title="Time after excitation (ms)",
        yaxis_title="Mxy / Mxy(0)",
        yaxis=dict(range=[0, 1.05]),
        width=750, height=400,
        margin=dict(l=60, r=20, t=40, b=40),
    )

    mo.vstack([
        mo.vstack([
            mo.hstack([t2_slider, t2prime_slider], gap=2),
            mo.hstack([t2_field_select, t2_show_tissues], justify="start", gap=2),
        ]),
        _fig,
        mo.md(
            f"T₂* = **{_t2star:.0f} ms** (always shorter than T₂ = {_t2_custom} ms due "
            f"to field inhomogeneities)"
        ),
        mo.callout(
            mo.md(
                "**T₂ vs T₂*:** T₂ is an intrinsic tissue property (spin-spin interactions). "
                "T₂* includes *additional* dephasing from magnetic field non-uniformities. "
                "Spin echo sequences can recover the T₂' component (getting back to 'true' T₂), "
                "but gradient echo sequences are sensitive to T₂*. This distinction will "
                "become important for fMRI in Notebook 3!"
            ),
            kind="warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. T1 and T2 Together: The Full Picture

    After an RF excitation pulse, **both relaxation processes happen
    simultaneously**: \(M_z\) recovers (T1) while \(M_{xy}\) decays (T2).
    The combined motion of the magnetization vector traces a **spiral**
    path as it precesses, dephases in the transverse plane, and recovers
    along the longitudinal axis.

    This is the complete Bloch equation picture. The visualization below
    shows the 3D trajectory of the magnetization vector after a 90° pulse.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bloch_t1_slider = mo.ui.slider(
        start=200, stop=4000, step=100, value=1300,
        label="T\u2081 (ms)",
    )
    bloch_t2_slider = mo.ui.slider(
        start=10, stop=300, step=10, value=80,
        label="T\u2082 (ms)",
    )
    bloch_b0_slider = mo.ui.slider(
        start=0.5, stop=7.0, step=0.5, value=3.0,
        label="B\u2080 (T)",
    )
    return bloch_b0_slider, bloch_t1_slider, bloch_t2_slider


@app.cell(hide_code=True)
def _(PrecessionWidget, bloch_b0_slider, bloch_t1_slider, bloch_t2_slider, mo):
    _t1 = bloch_t1_slider.value
    _t2 = bloch_t2_slider.value
    _b0 = bloch_b0_slider.value

    _widget = PrecessionWidget(
        b0=float(_b0), flip_angle=90.0,
        t1=float(_t1), t2=float(_t2),
        show_relaxation=True,
    )
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        mo.hstack([bloch_t1_slider, bloch_t2_slider, bloch_b0_slider], justify="start", gap=2),
        _wrapped,
        mo.callout(
            mo.md(
                "**The spiral tells the whole story:** The magnetization simultaneously "
                "precesses (circles in xy), dephases (|Mxy| decays via T\u2082), and "
                "recovers (Mz returns to equilibrium via T\u2081). Try making T\u2082 very short -- the "
                "signal collapses quickly. Make T\u2081 very short -- Mz snaps back fast.\n\n"
                "The signal traces on the right show |Mxy| (red) and Mz (teal) in real time. "
                "**Drag to rotate** the 3D view."
            ),
            kind="success",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Deep Dive: The complete Bloch equations": mo.md(
                r"""
                The Bloch equations describe the time evolution of the
                magnetization vector \(\vec{M} = (M_x, M_y, M_z)\) in a
                magnetic field:

                $$\frac{dM_x}{dt} = \gamma (\vec{M} \times \vec{B})_x - \frac{M_x}{T_2}$$

                $$\frac{dM_y}{dt} = \gamma (\vec{M} \times \vec{B})_y - \frac{M_y}{T_2}$$

                $$\frac{dM_z}{dt} = \gamma (\vec{M} \times \vec{B})_z - \frac{M_z - M_0}{T_1}$$

                The cross-product term describes precession, while the decay
                terms describe relaxation. In matrix form, for a single time
                step \(\Delta t\) with only \(B_0\) along z:

                1. **Precession**: Rotate \(M\) about z by angle \(\Delta\phi = \gamma B_0 \Delta t\)
                2. **T2 decay**: Multiply \(M_x\) and \(M_y\) by \(e^{-\Delta t / T_2}\)
                3. **T1 recovery**: Update \(M_z \rightarrow M_z \cdot e^{-\Delta t / T_1} + M_0 (1 - e^{-\Delta t / T_1})\)

                This is exactly how our simulation module implements the Bloch
                equations -- using rotation matrices for precession and
                exponential factors for relaxation.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Tissue Contrast: How TE and TR Create Different Images

    This is where MRI gets its remarkable soft-tissue contrast. By choosing
    when we **repeat** the excitation pulse (\(TR\) = Repetition Time) and
    when we **read** the signal (\(TE\) = Echo Time), we can make the image
    sensitive to different tissue properties:

    | Weighting | TR | TE | Sensitive to |
    |-----------|----|----|-------------|
    | **T1-weighted** | Short (~500ms) | Short (~10ms) | T1 differences (anatomy) |
    | **T2-weighted** | Long (~3000ms) | Long (~80ms) | T2 differences (pathology) |
    | **PD-weighted** | Long (~3000ms) | Short (~10ms) | Proton density |

    The spin echo signal equation combines both relaxation effects:

    $$S = PD \cdot (1 - e^{-TR/T_1}) \cdot e^{-TE/T_2}$$

    The first term (\(1 - e^{-TR/T_1}\)) is T1-weighting: short TR means
    tissues haven't fully recovered, so T1 differences matter. The second
    term (\(e^{-TE/T_2}\)) is T2-weighting: long TE means more T2 decay
    has occurred, so T2 differences matter.

    **This is the most important interactive in these notebooks.** Use the
    sliders below to explore how TE and TR change the contrast between tissues.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    te_slider = mo.ui.slider(
        start=5, stop=200, step=5, value=10,
        label="TE (ms)",
        full_width=True,
    )
    tr_slider = mo.ui.slider(
        start=100, stop=6000, step=100, value=500,
        label="TR (ms)",
        full_width=True,
    )
    contrast_field = mo.ui.dropdown(
        options={"1.5T": "1.5T", "3T": "3T"}, value="3T", label="Field strength",
    )
    return contrast_field, te_slider, tr_slider


@app.cell(hide_code=True)
def _(
    TISSUE_PROPERTIES,
    contrast_field,
    go,
    make_subplots,
    mo,
    np,
    plot_contrast_bars,
    spin_echo_signal,
    t1_recovery,
    t2_decay,
    te_slider,
    tr_slider,
):
    _te = te_slider.value
    _tr = tr_slider.value
    _field = contrast_field.value
    _tissues = TISSUE_PROPERTIES[_field]

    # Calculate signal for each tissue
    _tissue_names = list(_tissues.keys())
    _signals = []
    for _name in _tissue_names:
        _p = _tissues[_name]
        _s = spin_echo_signal(_te, _tr, _p["T1"], _p["T2"], _p["PD"])
        _signals.append(_s)

    # Determine weighting type
    if _tr < 800 and _te < 30:
        _weighting = "T₁-weighted"
    elif _tr > 2000 and _te > 60:
        _weighting = "T₂-weighted"
    elif _tr > 2000 and _te < 30:
        _weighting = "PD-weighted"
    else:
        _weighting = "Mixed weighting"

    # Bar chart (compact)
    _bar_fig = plot_contrast_bars(_tissue_names, _signals,
                                   title=f"{_weighting}")
    _bar_fig.update_layout(width=300, height=300, margin=dict(l=40, r=5, t=30, b=60),
                            xaxis_tickangle=-35, xaxis_tickfont=dict(size=9))

    # T1 recovery and T2 decay as separate compact plots
    _t_t1 = np.linspace(0, 6000, 500)
    _t_t2 = np.linspace(0, 300, 500)

    _tissue_colors = {
        "Gray Matter": "#808080", "White Matter": "#C8A882",
        "CSF": "#4169E1", "Fat": "#FFD700", "Muscle": "#CD5C5C"
    }

    # T1 recovery plot
    _t1_fig = go.Figure()
    for _name in _tissue_names:
        _p = _tissues[_name]
        _color = _tissue_colors.get(_name, "gray")
        _t1_fig.add_trace(go.Scatter(
            x=_t_t1, y=t1_recovery(_t_t1, _p["T1"], _p["PD"]),
            mode="lines", line=dict(color=_color, width=2), name=_name,
        ))
    _t1_fig.add_vline(x=_tr, line_dash="dash", line_color="red",
                       annotation_text=f"TR={_tr}")
    _t1_fig.update_layout(
        title="T\u2081 Recovery", width=350, height=300,
        margin=dict(l=40, r=5, t=30, b=40),
        xaxis_title="Time (ms)", yaxis_title="Signal",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(font=dict(size=8), x=0.55, y=0.35),
    )

    # T2 decay plot
    _t2_fig = go.Figure()
    for _name in _tissue_names:
        _p = _tissues[_name]
        _color = _tissue_colors.get(_name, "gray")
        _t2_fig.add_trace(go.Scatter(
            x=_t_t2, y=_p["PD"] * t2_decay(_t_t2, _p["T2"]),
            mode="lines", line=dict(color=_color, width=2),
            name=_name, showlegend=False,
        ))
    _t2_fig.add_vline(x=_te, line_dash="dash", line_color="red",
                       annotation_text=f"TE={_te}")
    _t2_fig.update_layout(
        title="T\u2082 Decay", width=350, height=300,
        margin=dict(l=40, r=5, t=30, b=40),
        xaxis_title="Time (ms)", yaxis_title="Signal",
        yaxis=dict(range=[0, 1.05]),
    )

    mo.vstack([
        mo.hstack([te_slider, tr_slider, contrast_field], justify="start", gap=2),
        mo.hstack([_bar_fig, _t1_fig, _t2_fig], justify="center"),
        mo.callout(
            mo.md(
                f"**Current weighting: {_weighting}** \u2014 "
                "Try: TR=500, TE=10 (T\u2081w) | TR=4000, TE=80 (T\u2082w) | TR=4000, TE=10 (PD)\n\n"
                "The red dashed lines show where TR and TE sample the recovery/decay curves. "
                "The bar chart shows the resulting signal for each tissue."
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Spin Echo & Gradient Echo

    We've been talking about the signal at time TE after excitation. But
    how do we actually *collect* the signal at a specific echo time? There
    are two fundamental approaches:

    ### Spin Echo (SE)
    A **180° refocusing pulse** is applied at time TE/2 after the initial
    90° excitation. This reverses the dephasing caused by static field
    inhomogeneities, causing the spins to **rephase** and form an echo
    at time TE. The spin echo signal depends on **T2** (not T2*), because
    the refocusing pulse undoes the T2' dephasing.

    ### Gradient Echo (GRE)
    Instead of a 180° pulse, a **gradient reversal** is used to form the
    echo. This is faster but does NOT refocus static field inhomogeneities,
    so the signal depends on **T2*** (not T2). Gradient echoes are the
    basis of most fMRI sequences (because fMRI needs T2* sensitivity!).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    echo_type = mo.ui.radio(
        options={"Spin Echo (90\u00b0-180\u00b0)": "spin_echo", "Gradient Echo": "gradient_echo"},
        value="Spin Echo (90\u00b0-180\u00b0)",
        label="Sequence type",
    )
    echo_speed = mo.ui.slider(
        start=0.3, stop=3.0, step=0.1, value=1.0,
        label="Animation speed",
    )
    return echo_speed, echo_type


@app.cell(hide_code=True)
def _(SpinEnsembleWidget, echo_speed, echo_type, mo):
    _seq_type = echo_type.value
    _speed = echo_speed.value

    _widget = SpinEnsembleWidget(sequence_type=_seq_type, speed=_speed)
    _wrapped = mo.ui.anywidget(_widget)

    if _seq_type == "spin_echo":
        _seq_desc = (
            "**Spin Echo**: 90\u00b0 \u2192 wait TE/2 \u2192 180\u00b0 (refocus) \u2192 wait TE/2 \u2192 Echo\n\n"
            "The 180\u00b0 pulse reverses all static dephasing. The echo signal "
            "reflects **true T\u2082** decay only. Used for anatomical imaging (T\u2081w, T\u2082w)."
        )
    else:
        _seq_desc = (
            "**Gradient Echo**: \u03b1\u00b0 \u2192 gradient dephase \u2192 gradient rephase \u2192 Echo\n\n"
            "No refocusing pulse, so static field inhomogeneities are NOT reversed. "
            "The echo signal reflects **T\u2082*** decay. Faster than spin echo. "
            "Used for **fMRI** (BOLD signal depends on T\u2082*!)."
        )

    mo.vstack([
        mo.hstack([echo_type, echo_speed], justify="start", gap=2),
        _wrapped,
        mo.md(_seq_desc),
        mo.callout(
            mo.md(
                "**Why does this matter for fMRI?** The BOLD signal (which we'll "
                "cover in Notebook 3) depends on local magnetic field changes caused "
                "by deoxyhemoglobin. These are T\u2082* effects -- they would be refocused "
                "away by a spin echo! That's why fMRI uses **gradient echo** sequences.\n\n"
                "Watch the signal trace on the right: in **spin echo**, the signal fully "
                "recovers at the echo. In **gradient echo**, residual dephasing remains."
            ),
            kind="warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    In this notebook, we've explored how MRI creates contrast between tissues:

    1. **T₁ relaxation** (spin-lattice): Longitudinal magnetization recovers
       exponentially. Different tissues have different T₁ values.
    2. **T₂ relaxation** (spin-spin): Transverse magnetization decays
       exponentially. T₂* includes additional dephasing from field inhomogeneities.
    3. **The Bloch equations** describe both processes simultaneously --
       the magnetization spirals back to equilibrium.
    4. **TE and TR** control which relaxation mechanism dominates the image
       contrast. Short TR → T₁-weighted. Long TE → T₂-weighted.
    5. **Spin echo** refocuses static dephasing (T₂ signal), while
       **gradient echo** preserves it (T₂* signal, used in fMRI).

    **Next up:** In *MR Physics 3: Imaging & fMRI*, we'll learn how
    gradients encode spatial information, how k-space works, and how the
    BOLD signal lets us measure brain activity.
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
