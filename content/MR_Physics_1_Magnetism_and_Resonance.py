import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    app_title="MR Physics 1: Magnetism & Resonance",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MR Physics 1: Magnetism & Resonance

    *Written by Luke Chang*

    How do MRI scanners work? And what signal are we actually measuring?
    Understanding the physics underlying MRI is essential for interpreting
    neuroimaging results, yet many people who use MRI don't fully grasp
    how it works -- largely because MR physics can be unintuitive.

    The goal of this notebook is to build your intuition from the ground
    up, starting with things you already know (magnets and compasses) and
    working toward the nuclear magnetic resonance signal that MRI scanners
    detect. Along the way, you'll interact with simulations that let you
    **see** and **feel** how changing physical parameters affects the signal.

    > **Note on simplifications:** We'll use *classical* physics explanations
    > throughout, following the approach advocated by
    > [Lars G. Hanson](https://www.drcmr.dk/bloch). While not quantum-mechanically
    > complete, this classical picture provides strong intuitions that will
    > serve you well. Where the full story differs, we'll flag it in
    > optional "Deep Dive" sections.

    This is **Part 1** of a three-notebook series:
    1. **Magnetism & Resonance** (this notebook)
    2. Signal & Contrast
    3. Imaging & fMRI
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
    from Code.mr_simulations import (
        GAMMA_H, GAMMA, TISSUE_PROPERTIES,
        rotation_x, rotation_y, rotation_z,
        apply_rf_pulse, apply_relaxation, simulate_bloch,
        fid_signal, compute_spectrum,
        plot_magnetization_3d, plot_signal_timeline,
    )
    from Code.mr_widgets import CompassWidget, NetMagnetizationWidget, PrecessionWidget

    return (
        CompassWidget,
        GAMMA,
        GAMMA_H,
        NetMagnetizationWidget,
        PrecessionWidget,
        apply_rf_pulse,
        compute_spectrum,
        fid_signal,
        go,
        mo,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Interactive version:** [Open this notebook in molab](https://molab.marimo.io/github/ljchang/dartbrains/blob/v2-marimo-migration/content/MR_Physics_1_Magnetism_and_Resonance.py) to run code, interact with widgets, and modify examples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
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
def _(b0_compass_slider, CompassWidget, mo):
    _widget = CompassWidget(b0=float(b0_compass_slider.value))
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        b0_compass_slider,
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

    We don't have compass needles inside the body, but we have something
    even better: an **astronomical number of hydrogen nuclei** (protons).
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
def _(n_protons_slider, b0_on_toggle, NetMagnetizationWidget, mo):
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
    mo.accordion(
        {
            "Deep Dive: Why only a tiny surplus?": mo.md(
                r"""
                At body temperature and clinical field strengths, the energy
                difference between "spin up" (aligned with \(B_0\)) and "spin down"
                (anti-aligned) is *minuscule* compared to thermal energy. At 3T,
                only about **10 out of every 2 million** protons contribute to the
                net magnetization. But because there are roughly \(10^{22}\)
                protons per mL of tissue, that tiny surplus still produces a
                measurable signal.

                The equilibrium magnetization is given by:

                $$M_0 = \frac{N \gamma^2 \hbar^2 B_0}{4 k_B T}$$

                where \(N\) is the proton density, \(\gamma\) is the gyromagnetic
                ratio, \(\hbar\) is the reduced Planck constant, \(k_B\) is
                Boltzmann's constant, and \(T\) is temperature.

                In our visualization above, we greatly exaggerate the alignment
                bias so you can see the effect. In reality, you would need
                millions of spins before the net vector becomes visually apparent.
                """
            )
        }
    )
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
def _(GAMMA, GAMMA_H, PrecessionWidget, b0_larmor_slider, mo):
    _b0 = b0_larmor_slider.value
    _larmor_freq = GAMMA_H * _b0

    _widget = PrecessionWidget(b0=_b0, flip_angle=30.0, show_relaxation=False)
    _wrapped = mo.ui.anywidget(_widget)

    mo.vstack([
        b0_larmor_slider,
        _wrapped,
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
    mo.accordion(
        {
            "Deep Dive: The Larmor Equation": mo.md(
                r"""
                The Larmor equation relates the precession frequency to the
                magnetic field:

                $$\omega_0 = \gamma \cdot B_0$$

                or equivalently in terms of frequency (Hz rather than rad/s):

                $$f_0 = \frac{\gamma}{2\pi} \cdot B_0$$

                The gyromagnetic ratio \(\gamma\) is an intrinsic property of
                each nuclear species. For \(^1\)H (protons):

                $$\gamma / 2\pi = 42.576 \text{ MHz/T}$$

                This means:
                - At **1.5T**: \(f_0 = 63.86\) MHz
                - At **3.0T**: \(f_0 = 127.73\) MHz
                - At **7.0T**: \(f_0 = 298.03\) MHz

                The fact that different nuclei precess at different frequencies
                is what allows MRI to be nucleus-specific. By tuning our RF
                transmitter and receiver to the hydrogen Larmor frequency, we
                selectively excite and detect only hydrogen -- ignoring carbon,
                sodium, and everything else.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. RF Excitation & Resonance

    The net magnetization \(M_0\) points along \(B_0\) (the z-axis) at
    equilibrium. Unfortunately, we can't directly measure magnetization
    along z -- our receiver coils are only sensitive to magnetization
    **in the transverse (xy) plane**.

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
def _(PrecessionWidget, flip_angle_slider, GAMMA_H, mo):
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
                "- The decay rate is determined by T\u2082* (we'll explore this in Notebook 2)\n"
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
    ## Summary

    In this notebook, we've traced the MR signal from first principles:

    1. **Magnetic fields** cause magnetic objects to align and oscillate
    2. **Hydrogen protons** in the body act like tiny magnets, creating a net
       magnetization (\(M_0\)) when placed in a strong \(B_0\) field
    3. These protons **precess** at the **Larmor frequency** (\(\omega_0 = \gamma B_0\)),
       which depends on field strength
    4. An **RF pulse** at the Larmor frequency tips the magnetization into the
       transverse plane -- this is **resonance**
    5. The precessing transverse magnetization induces a **Free Induction Decay**
       signal in the receiver coil

    **Next up:** In *MR Physics 2: Signal & Contrast*, we'll explore how the
    signal decays (T1 and T2 relaxation) and how MRI pulse sequences exploit
    these differences to create contrast between tissues.
    """)
    return


if __name__ == "__main__":
    app.run()
