import marimo

__generated_with = "0.23.2"
app = marimo.App(
    width="medium",
    app_title="MR Physics: From Protons to Brain Images",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MR Physics: From Protons to Brain Images

    *Written by Luke Chang & Ben Graul*

    How do MRI scanners work? And what signal are we actually measuring?
    Understanding the physics underlying MRI is essential for interpreting
    neuroimaging results, yet many people who use MRI don't fully grasp
    how it works -- largely because MR physics can be unintuitive.

    This course primarily focuses on Blood Oxygenated Level Dependent (BOLD) fMRI signals. Gaining a deep understanding of the MR physics and physiological basis for the BOLD fMRI signal is beyond the scope of this course and we refer the interested reader to the excellent [Huettel, Song, & McCarthy (2004) Functional magnetic resonance imaging textbook](https://www.amazon.com/Functional-Magnetic-Resonance-Imaging-Huettel/dp/0878936270/ref=pd_sbs_14_1/144-9493364-1935804?_encoding=UTF8&pd_rd_i=0878936270&pd_rd_r=ac61b1df-17bf-47c5-8db5-25dfa36bcd16&pd_rd_w=J61zv&pd_rd_wg=d1O2i&pf_rd_p=703f3758-d945-4136-8df6-a43d19d750d1&pf_rd_r=PCEXDFT3TQQ4JW7FD8HF&psc=1&refRID=PCEXDFT3TQQ4JW7FD8HF) for a more in depth conceptual and quantitative overview.

    The goal of this tutorial is to build your intuition from the ground
    up, starting with things you already know (magnets and compasses) and
    working toward functional brain imaging. Along the way, you'll interact
    with simulations that let you **see** and **feel** how changing physical
    parameters affects the signal.

    > **Note on simplifications:** We'll use *classical* physics explanations
    > throughout, following the approach advocated by
    > [Lars G. Hanson](https://www.drcmr.dk/bloch). While not quantum-mechanically
    > complete, this classical picture provides strong intuitions that will
    > serve you well. Where the full story differs, we'll flag it in
    > optional "Deep Dive" sections.

    This notebook covers three major topics:
    1. **Magnetism & Resonance** — magnetic fields, proton alignment, precession, RF excitation, and the FID signal
    2. **Signal & Contrast** — T₁/T₂ relaxation, the Bloch equations, tissue contrast, and pulse sequences
    3. **Imaging & fMRI** — gradients, spatial encoding, k-space, the BOLD signal, and EPI
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dartbrains_tools.mr_simulations import (
        GAMMA_H, GAMMA, TISSUE_PROPERTIES,
        rotation_x, rotation_y, rotation_z,
        apply_rf_pulse, apply_relaxation, simulate_bloch,
        fid_signal, compute_spectrum,
        t1_recovery, t2_decay,
        spin_echo_signal, gradient_echo_signal,
        hrf, image_to_kspace, kspace_to_image, mask_kspace,
        plot_magnetization_3d, plot_signal_timeline,
        plot_contrast_bars, plot_pulse_sequence,
        plot_kspace_and_image,
    )
    from dartbrains_tools.mr_widgets import (
        CompassWidget, NetMagnetizationWidget, PrecessionWidget,
        SpinEnsembleWidget, EncodingWidget, KSpaceWidget, ConvolutionWidget,
    )

    return (
        CompassWidget,
        ConvolutionWidget,
        EncodingWidget,
        GAMMA,
        GAMMA_H,
        KSpaceWidget,
        NetMagnetizationWidget,
        PrecessionWidget,
        SpinEnsembleWidget,
        TISSUE_PROPERTIES,
        compute_spectrum,
        fid_signal,
        go,
        hrf,
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's first start with a short video on the basics of MR physics by Martin Lindquist.

    ---
    # Part 1: Magnetism & Resonance

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
      <iframe
          width="560" height="315"
          src="https://www.youtube.com/embed/XsDXxgjEJVY"
          frameborder="0" allowfullscreen>
      </iframe>
      """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What is a Magnetic Field?

    You've used a compass before -- a tiny magnetized needle that aligns
    with the Earth's magnetic field to point north. This simple device
    illustrates the core idea behind MRI: **magnetic objects tend to align
    with an applied magnetic field.**

    The Earth's magnetic field is weak -- only about 25-65 *micro*Teslas
    (µT). An MRI scanner produces a field that is roughly **100,000 times
    stronger**: typically 1.5 or 3 Tesla (T), with research scanners
    going up to 7T or beyond.

    Let's start with a simulation inspired by the
    [DRCMR Compass MR Simulator](https://www.drcmr.dk/CompassMR/).
    Use the slider below to change the strength of the external magnetic
    field (\(B_0\)) and watch how a compass needle responds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    b0_compass_slider = mo.ui.slider(
        start=0.0, stop=5.0, step=0.1, value=3.0,
        label="B\u2080 field strength (mT)",
        full_width=True,
    )
    return (b0_compass_slider,)


@app.cell(hide_code=True)
def _(CompassWidget, b0_compass_slider, mo):
    _widget = CompassWidget(b0=float(b0_compass_slider.value))
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        _wrapped,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key observations:**
    - With no field (\(B_0 = 0\)), the needle just sits wherever it is -- no preferred direction.
    - As you increase \(B_0\), the needle oscillates back toward alignment **faster**.
    - The **frequency** of oscillation increases with field strength.
    - The oscillation is detected by a nearby coil as a **signal** that decays over time.

    These same principles apply inside an MRI scanner, except instead of compass
    needles, we're working with hydrogen nuclei -- tiny magnets inside your body.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. From Compass to Proton

    We don't have compass needles inside the body, but do have a large number of **hydrogen nuclei** (protons).
    Hydrogen is the most abundant element in the body (it's in every water
    molecule), and each hydrogen nucleus acts like a tiny magnet because of
    a quantum property called **spin**.

    In the absence of an external magnetic field, these tiny magnets point
    in random directions, and their effects cancel out -- there's no net
    magnetization. But when we place them in a strong \(B_0\) field, a
    slight majority align *with* the field rather than *against* it. This
    tiny surplus creates a measurable **net magnetization** vector, called
    \(M_0\), that points along \(B_0\).

    Use the sliders below to see how random spins create a net magnetization
    when a field is applied.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_protons_slider = mo.ui.slider(
        start=10, stop=500, step=10, value=100,
        label="Number of protons",
    )
    b0_on_toggle = mo.ui.switch(label="B\u2080 field ON", value=False)
    return b0_on_toggle, n_protons_slider


@app.cell(hide_code=True)
def _(NetMagnetizationWidget, b0_on_toggle, mo, n_protons_slider):
    _widget = NetMagnetizationWidget(
        n_protons=int(n_protons_slider.value),
        b0_on=bool(b0_on_toggle.value),
    )
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        mo.hstack([n_protons_slider, b0_on_toggle], justify="start", gap=2),
        _wrapped,
        mo.callout(
            mo.md(
                "With the field **OFF**, spins point in random directions and jitter freely -- "
                "the net magnetization (red arrow) is near zero. Toggle B\u2080 **ON** to watch "
                "the spins gradually align, and a net magnetization emerges along z. "
                "**Drag to rotate** the 3D view."
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Precession & the Larmor Frequency

    When a spinning top is tilted, it doesn't just fall over -- it
    **precesses**, tracing a circle as it wobbles around the vertical axis.
    Protons in a magnetic field do the same thing: their spin axes precess
    around the direction of \(B_0\).

    The rate of precession is governed by the **Larmor equation**:

    $$\omega_0 = \gamma \cdot B_0$$

    where:
    - \(\omega_0\) is the Larmor (precession) frequency
    - \(\gamma\) is the **gyromagnetic ratio** -- a constant unique to each nucleus
    - \(B_0\) is the magnetic field strength

    For hydrogen protons, \(\gamma = 42.576\) MHz/T. This means at 3T,
    protons precess at about **127.7 MHz** -- in the radiofrequency (RF)
    range!

    Adjust the field strength below and see how the precession frequency changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    b0_larmor_slider = mo.ui.slider(
        start=0.5, stop=7.0, step=0.1, value=3.0,
        label="B\u2080 (Tesla)",
        full_width=True,
    )
    return (b0_larmor_slider,)


@app.cell(hide_code=True)
def _(GAMMA_H, PrecessionWidget, b0_larmor_slider, mo):
    _b0 = b0_larmor_slider.value
    _larmor_freq = GAMMA_H * _b0

    _widget = PrecessionWidget(b0=_b0, flip_angle=30.0, show_relaxation=False)
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
      b0_larmor_slider,
      _wrapped,
    ])
    return


@app.cell(hide_code=True)
def _(GAMMA, GAMMA_H, b0_larmor_slider, mo):
    _b0 = b0_larmor_slider.value
    _larmor_freq = GAMMA_H * _b0


    mo.vstack([
        mo.callout(
            mo.md(
                f"At **B\u2080 = {_b0:.1f} T**, the Larmor frequency for hydrogen is "
                f"**{_larmor_freq:.1f} MHz** ({_larmor_freq/1e3:.4f} GHz). "
                f"This is in the **radiofrequency** range -- the same part of the "
                f"electromagnetic spectrum used by FM radio!\n\n"
                "**Drag to rotate** the 3D view. The signal traces on the right show "
                "|Mxy| (red) and Mz (teal) in real time."
            ),
            kind="success",
        ),
        mo.md("**Larmor frequencies of different nuclei at this field strength:**"),
        mo.md(
            "| Nucleus | \u03b3 (MHz/T) | Larmor freq at "
            + f"{_b0:.1f}T |\n|---------|-----------|----------|\n"
            + "\n".join(
                f"| {name} | {ratio:.3f} | {ratio * _b0:.2f} MHz |"
                for name, ratio in GAMMA.items()
            )
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. RF Excitation & Resonance

    In an MRI machine, $B_0$ is the strong, static magnetic field that runs along the bore of the scanner (the **longitudinal plane** or z-axis). It aligns hydrogen protons in your body roughly parallel to the field, creating a net magnetization $M_0$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("/Users/lukechang/Github/dartbrains/images/signal_generation/b0.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unfortunately, we can't directly measure magnetization
    along z -- our receiver coils are only sensitive to magnetization
    in the **transverse (xy) plane**.

    To get a signal, we need to **tip** the magnetization away from z.
    We do this by applying a second, much weaker magnetic field called
    \(B_1\), oriented perpendicular to \(B_0\), and oscillating at the
    **Larmor frequency**. This is the radiofrequency (RF) pulse.

    Think of pushing a child on a swing: if you push at the swing's
    natural frequency, each push adds energy and the swing goes higher.
    Push at the wrong frequency, and nothing much happens. This is
    **resonance** -- and it's the "R" in MRI.

    The angle by which the magnetization is tipped is called the **flip
    angle** (\(\alpha\)). A 90° pulse tips \(M\) entirely into the
    xy-plane. A 180° pulse inverts it.

    Try adjusting the flip angle and the B1 frequency below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    flip_angle_slider = mo.ui.slider(
        start=0, stop=180, step=5, value=90,
        label="Flip angle (degrees)",
        full_width=True,
    )
    return (flip_angle_slider,)


@app.cell(hide_code=True)
def _(PrecessionWidget, flip_angle_slider, mo):
    _flip = flip_angle_slider.value

    _widget = PrecessionWidget(b0=3.0, flip_angle=float(_flip), show_relaxation=False)
    _wrapped = mo.ui.anywidget(_widget)

    _mxy = abs(round(float(__import__('math').sin(__import__('math').radians(_flip))), 2))
    _mz = round(float(__import__('math').cos(__import__('math').radians(_flip))), 2)

    mo.vstack([
        flip_angle_slider,
        _wrapped,
        mo.md(
            f"Transverse magnetization |Mxy| = **{_mxy:.2f}** "
            f"(this is our detectable signal strength)\n\n"
            f"Longitudinal magnetization Mz = **{_mz:.2f}**"
        ),
        mo.callout(
            mo.md(
                "**Try this:** Set the flip angle to 90\u00b0 and watch the vector tip "
                "fully into the xy-plane -- maximum signal! At 180\u00b0, the vector "
                "inverts completely. The signal traces on the right show how |Mxy| and "
                "Mz change. **Drag to rotate** the 3D view."
            ),
            kind="info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. The Signal We Measure: Free Induction Decay

    After the RF pulse tips the magnetization into the transverse plane,
    the precessing \(M_{xy}\) component induces a voltage in the receiver
    coil -- just like the oscillating compass needle produced a signal
    in Section 1. This signal is called the **Free Induction Decay (FID)**.

    The FID has two key features:
    1. It **oscillates** at the Larmor frequency (the precessing magnetization)
    2. It **decays** over time as the transverse magnetization dephases
       (characterized by the time constant \(T_2^*\))

    We can extract the frequency content of the FID using a **Fourier
    Transform** (FFT) -- the same mathematical tool introduced in the
    Dartbrains Signal Processing chapter. The FFT reveals a peak at the
    Larmor frequency, confirming that the signal came from precessing protons.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fid_flip_slider = mo.ui.slider(
        start=5, stop=90, step=5, value=90,
        label="Flip angle (\u00b0)",
    )
    fid_t2star_slider = mo.ui.slider(
        start=10, stop=200, step=10, value=50,
        label="T\u2082* (ms)",
    )
    fid_show_fft = mo.ui.switch(label="Show frequency spectrum (FFT)", value=False)
    return fid_flip_slider, fid_show_fft, fid_t2star_slider


@app.cell(hide_code=True)
def _(
    compute_spectrum,
    fid_flip_slider,
    fid_show_fft,
    fid_signal,
    fid_t2star_slider,
    go,
    mo,
    np,
):
    _flip = fid_flip_slider.value
    _t2star = fid_t2star_slider.value
    _show_fft = fid_show_fft.value

    _dt = 0.1
    _t = np.arange(0, 500, _dt)
    _f0 = 0.05
    _sig = fid_signal(_t, _t2star, f0=_f0, flip_angle_deg=_flip)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_t, y=np.real(_sig), mode="lines",
        line=dict(color="#636EFA", width=2), name="FID (real part)",
    ))
    _envelope = np.sin(np.radians(_flip)) * np.exp(-_t / _t2star)
    _fig.add_trace(go.Scatter(
        x=_t, y=_envelope, mode="lines",
        line=dict(color="red", width=1.5, dash="dash"), name="T\u2082* envelope",
    ))
    _fig.add_trace(go.Scatter(
        x=_t, y=-_envelope, mode="lines",
        line=dict(color="red", width=1.5, dash="dash"), showlegend=False,
    ))
    _fig.update_layout(
        title=f"Free Induction Decay (flip={_flip}\u00b0, T\u2082*={_t2star} ms)",
        xaxis_title="Time (ms)", yaxis_title="Signal (a.u.)",
        yaxis=dict(range=[-1.1, 1.1]),
        width=700, height=300, margin=dict(l=60, r=20, t=40, b=40),
    )

    _elements = [
        mo.hstack([fid_flip_slider, fid_t2star_slider, fid_show_fft], justify="start", gap=2),
        _fig,
    ]

    if _show_fft:
        _freqs, _mag = compute_spectrum(_sig, _dt)
        _fft_fig = go.Figure()
        _fft_fig.add_trace(go.Scatter(
            x=_freqs, y=_mag, mode="lines",
            line=dict(color="#00CC96", width=2), name="Magnitude spectrum",
        ))
        _fft_fig.update_layout(
            title="Frequency Spectrum (FFT of FID)",
            xaxis_title="Frequency (kHz)", yaxis_title="Magnitude",
            yaxis=dict(range=[0, 0.55]),
            width=700, height=300, margin=dict(l=60, r=20, t=40, b=40),
        )
        _elements.append(_fft_fig)
        _elements.append(
            mo.md(
                f"The FFT shows a peak at **{_f0*1000:.0f} Hz** -- the precession "
                f"frequency of our protons. The width of the peak is inversely "
                r"related to \(T_2^*\): shorter \(T_2^*\) means faster decay, "
                f"which means a broader spectral peak."
            )
        )

    _elements.append(
        mo.callout(
            mo.md(
                "**Key takeaways:**\n"
                "- The FID amplitude depends on the flip angle (maximum at 90\u00b0)\n"
                "- The decay rate is determined by T\u2082* (we'll explore this next)\n"
                "- The Fourier transform reveals the precession frequency\n"
                "- **This is the fundamental signal that MRI is built upon!**"
            ),
            kind="success",
        )
    )

    mo.vstack(_elements)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 2: Signal & Contrast

    We've learned how an RF pulse tips the net magnetization into the
    transverse plane, producing a Free Induction Decay signal. But
    **why does the signal decay?** And how do MRI scanners produce images
    where gray matter, white matter, and CSF all look different?

    The answers lie in **relaxation** -- the processes by which the
    magnetization returns to equilibrium after excitation. Different
    tissues relax at different rates, and MRI pulse sequences exploit
    these differences to create **contrast**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. T1 Relaxation

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
    ## 7. T2 Relaxation

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
                "become important for fMRI later in this notebook!"
            ),
            kind="warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8. T1 and T2 Together: The Full Picture

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
    ## 9. Tissue Contrast: How TE and TR Create Different Images

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

    **This is the most important interactive plot in this notebook.** Use the
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
    ## 10. Spin Echo & Gradient Echo

    We've been talking about the signal at time TE after excitation. But
    how do we actually *collect* the signal at a specific echo time? There
    are two fundamental approaches:

    ### Spin Echo (SE)
    A **180° refocusing pulse** is applied at time TE/2 after the initial
    90° excitation. This reverses the dephasing caused by static field
    inhomogeneities, causing the spins to **rephase** and form an echo
    at time TE. The spin echo signal depends on **T2** (not T2*), because
    the refocusing pulse undoes the T2' dephasing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("/Users/lukechang/Github/dartbrains/images/signal_generation/spin_echo_pulse_sequence.svg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The diagram shows one full spin echo cycle.

    A **spin echo pulse sequence** begins with a 90° RF pulse that tips the net magnetization from the longitudinal axis into the transverse plane. Immediately after excitation, spins begin to dephase due to local field inhomogeneities, producing a decaying free induction decay (FID).

    A 180° refocusing pulse applied at time TE/2 reverses this dephasing, causing spins to rephase and form an echo at time **TE**.

    The sequence repeats every **TR** (indicated by the faded 90° pulse at the right edge).

    Three spatial encoding gradients operate alongside the RF pulses:

    - **Gss (slice select)** is applied during both RF pulses to restrict excitation to a single imaging slice
    - **Gpe (phase encode)** applies a brief gradient of varying amplitude on each repetition, stepping through k-space line by line
    - **Gro (readout/frequency encode)** is switched on during the echo to spatially encode signal along the remaining axis.

    Together, these three gradients provide the spatial information needed to reconstruct a 2D image from the acquired signal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gradient Echo (GRE)
    Instead of a 180° pulse, a **gradient reversal** is used to form the
    echo. This is faster but does NOT refocus static field inhomogeneities,
    so the signal depends on **T2*** (not T2). Gradient echoes are the
    basis of most fMRI sequences (because fMRI needs T2* sensitivity!).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("/Users/lukechang/Github/dartbrains/images/signal_generation/gradient_echo_pulse_sequence.svg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A **gradient echo pulse sequence** begins with a low flip angle (α) RF pulse — typically much less than 90° — that tips a fraction of the longitudinal magnetization into the transverse plane.

    Unlike the spin echo, there is no 180° refocusing pulse.

    Instead, the echo is formed entirely by gradient manipulation: a negative dephasing lobe on the **readout gradient (Gro)** deliberately dephases the spins, and then a positive lobe of opposite polarity rephases them to produce the gradient echo at time TE.

    Because the 180° pulse is absent, static field inhomogeneities are not corrected, so the signal decays with T2* rather than T2.

    The **slice select gradient (Gss)** is applied during the α pulse to restrict excitation to a single slice, and the **phase encode gradient (Gpe)** steps through different amplitudes on each repetition to fill k-space.

    The combination of a small flip angle and no refocusing pulse allows very short TR and TE, making gradient echo sequences the basis for fast imaging methods such as FLASH, SPGR, and MPRAGE.

    Here is an interactive figure showing this process over time for each sequence type.
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
                "cover next) depends on local magnetic field changes caused "
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
    # Part 3: Imaging & fMRI

    We now know how to generate a signal and how different tissues produce
    different signals. But there's a critical problem: **where did the
    signal come from?**

    If the entire volume inside the scanner experiences the same \(B_0\)
    field, all protons precess at the same frequency, and we can't tell
    whether the signal is coming from your frontal cortex or your cerebellum.
    We need **spatial encoding** -- a way to tag different locations with
    different frequencies or phases so we can reconstruct an image.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
      <iframe
          width="560" height="315"
          src="https://www.youtube.com/embed/PxqDjhO9FUs"
          frameborder="0" allowfullscreen>
      </iframe>
      """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 11. The Localization Problem: Why We Need Gradients

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
    ## 12. Slice Selection

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
    ## 13. Frequency & Phase Encoding

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
    ## 14. K-Space: Where MRI Data Lives
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
      <iframe
          width="560" height="315"
          src="https://www.youtube.com/embed/FI5frNsRTI4"
          frameborder="0" allowfullscreen>
      </iframe>
      """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
        value="Progressive fill (animated)",
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
    ## 15. BOLD & the Hemodynamic Response
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
      <iframe
          width="560" height="315"
          src="https://www.youtube.com/embed/jG2WQpgpnMs"
          frameborder="0" allowfullscreen>
      </iframe>
      """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
        value="Single event",
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
    ## 16. fMRI Pulse Sequences: Echo Planar Imaging (EPI)

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
    ## 17. Putting It All Together: From Proton to Activation Map

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
    ## Summary

    | Topic | Key Concepts |
    |-------|-------------|
    | **Magnetism & Resonance** | B₀ alignment, precession, Larmor equation, RF excitation, FID |
    | **Signal & Contrast** | T₁/T₂ relaxation, Bloch equations, TE/TR contrast, spin echo vs gradient echo |
    | **Imaging & fMRI** | Gradients, spatial encoding, k-space, BOLD signal, HRF, EPI |

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


if __name__ == "__main__":
    app.run()
