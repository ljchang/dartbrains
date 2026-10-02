# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dartbrains-tools>=0.3.2",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# 
# [tool.grader]
# server = "https://grader.dartbrains.org"
# course = "neuroimaging"
# term = "2026-fall"
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import sin, pi, arange, exp, real, imag
    from numpy.fft import fft, ifft, fftfreq
    from scipy.special import gamma as gamma_func
    from scipy.signal import butter, filtfilt, firwin, freqz

    def glover_hrf(tr_val, oversampling=1):
        _dt = tr_val / oversampling
        _ts = np.arange(0, 32, _dt)
        _peak, _under, _beta = 5.0, 15.0, 0.35
        _h = (
            (_ts / _peak) ** _peak * np.exp(-(_ts - _peak)) / gamma_func(_peak + 1)
            - _beta * (_ts / _under) ** _under * np.exp(-(_ts - _under)) / gamma_func(_under + 1)
        )
        return _h / _h.max()


@app.cell(hide_code=True)
def _():
    from dartbrains_tools.notebook_utils import youtube

    return (youtube,)


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # Signal Processing Basics
    *Written by Luke Chang*

    In this lab, we will cover the basics of convolution, sine waves, and fourier transforms. This lab is largely based on exercises from Mike X Cohen's excellent book, [Analyzing Neural Data Analysis: Theory and Practice](https://www.amazon.com/Analyzing-Neural-Time-Data-Practice/dp/0262019876). If you are interested in learning in more detail about the basics of EEG and time-series analyses I highly recommend his accessible introduction. I also encourage you to watch his accompanying freely available [*lecturelets*](https://www.youtube.com/channel/UCUR_LsXk7IYyueSnXcNextQ) to learn more about each topic introduced in this notebook.

    ## Time Domain

    First we will work on signals in the time domain. This requires measuring a signal at a constant interval over time. The frequency with which we measure a signal is referred to as the sampling frequency. The units of this are typically described in $Hz$ - or the number of cycles per second. It is critical that the sampling frequency is consistent over the entire measurement of the time series.

    ### Dot Product
    To understand convolution, we first need to familiarize ourselves with the dot product. The dot product is simply the sum of the elements of a vector weighted by the elements of another vector. This method is commonly used in signal processing, and also in statistics as a measure of similarity between two vectors. Finally, there is also a geometric interpretation which is a mapping between vectors (i.e., the product of the magnitudes of the two vectors scaled by the cosine of the angle between them). For a more in depth overview of the dot product and its relation to convolution, you can watch this optional [video](https://youtu.be/rea6M1oagmA).

    $$dotproduct_{ab}=\sum\limits_{i=1}^n a_i b_i$$

    Let's create some vectors of random numbers and see how the dot product works. First, the two vectors need to be of the same length.
    """)
    return


@app.cell
def dot_product():
    dot_a = np.random.randint(1, 10, 20)
    _b = np.random.randint(1, 10, 20)
    _fig, _ax = plt.subplots()
    _ax.scatter(dot_a, _b)
    _ax.set_ylabel("b", fontsize=18)
    _ax.set_xlabel("a", fontsize=18)
    _ax.set_title("Two random vectors", fontsize=18)
    plt.close()
    mo.vstack([mo.md(f"**Dot Product:** {np.dot(dot_a, _b)}"), _fig])
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def dot_slider_md():
    mo.md(r"""
    What happens when we make the two vectors more similar? Use the slider to move **b** from unrelated to **a** (0) to identical to **a** (1). Each bar below is one term $a_i b_i$, and the dot product is the sum of all the bars. When the vectors agree, the terms are mostly positive and pile up. When they are unrelated, positive and negative terms cancel out.

    These vectors are centered on zero, so "unrelated" gives a dot product near zero. Standardize both vectors and divide the dot product by $n$, and you get the familiar **correlation** coefficient.
    """)
    return


@app.cell(hide_code=True)
def _make_dot_slider():
    dot_sim_slider = mo.ui.slider(0, 1, value=0.0, step=0.05, label="Similarity of b to a", full_width=True)
    return (dot_sim_slider,)


@app.cell(hide_code=True)
def dot_similarity(dot_sim_slider):
    _rng = np.random.default_rng(3)
    _n = 30
    _a = _rng.standard_normal(_n)
    _w = dot_sim_slider.value
    _b = _w * _a + np.sqrt(1 - _w**2) * _rng.standard_normal(_n)

    _fig, _axes = plt.subplots(ncols=2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.8]})
    _axes[0].scatter(_a, _b)
    _axes[0].set_xlim(-3, 3)
    _axes[0].set_ylim(-3, 3)
    _axes[0].set_aspect("equal")
    _axes[0].set_xlabel("a", fontsize=14)
    _axes[0].set_ylabel("b", fontsize=14)
    _axes[1].bar(np.arange(_n), _a * _b, color=np.where(_a * _b >= 0, "C0", "C3"))
    _axes[1].axhline(0, color="gray", linewidth=0.8)
    _axes[1].set_ylim(-6, 6)
    _axes[1].set_xlabel("element i", fontsize=14)
    _axes[1].set_title(r"$a_i \times b_i$", fontsize=14)
    plt.tight_layout()
    plt.close()
    mo.vstack([
        dot_sim_slider,
        mo.md(f"**Dot product: {np.dot(_a, _b):.1f}** &nbsp; (correlation r = {np.corrcoef(_a, _b)[0, 1]:.2f})"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def conv_intro():
    mo.md(r"""
    ### Convolution
    Convolution in the time domain is an extension of the dot product in which the dot product is computed iteratively over time. One way to think about it is that one signal weights each time point of the other signal and then slides forward over time. Let's call the timeseries variable *signal* and the other vector the *kernel*.

    To gain an intuition of how convolution works, let's play with some data. First, let's create a time series of spikes. Then let's convolve this signal with a boxcar kernel.
    """)
    return


@app.cell
def conv_setup():
    n_samples = 100
    conv_signal = np.zeros(n_samples)
    conv_signal[np.random.randint(0, n_samples, 5)] = 1
    boxcar_kernel = np.zeros(10)
    boxcar_kernel[2:8] = 1
    _fig, _axes = plt.subplots(ncols=2, figsize=(20, 5))
    _axes[0].plot(conv_signal, linewidth=2)
    _axes[0].set_xlabel("Time (samples)", fontsize=18)
    _axes[0].set_ylabel("Signal Intensity", fontsize=18)
    _axes[0].set_title("Signal", fontsize=18)
    _axes[1].plot(boxcar_kernel, linewidth=2, color="red")
    _axes[1].set_xlabel("Time (samples)", fontsize=18)
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_title("Kernel", fontsize=18)
    plt.tight_layout()
    plt.close()
    _fig
    return boxcar_kernel, conv_signal, n_samples


@app.cell(hide_code=True)
def shift_md():
    mo.md("""
    Notice how the kernel is only 10 samples long and the boxcar is 6 samples wide, while the signal is 100 samples long with 5 single pulses.

    Now let's convolve the signal with the kernel by taking the dot product of the kernel with each time point of the signal. This can be illustrated by creating a matrix of the kernel shifted each time point of the signal.
    """)
    return


@app.cell
def shifted_kernels(boxcar_kernel, n_samples):
    shifted_kernel = np.zeros((n_samples, n_samples + len(boxcar_kernel) - 1))
    for _k in range(n_samples):
        shifted_kernel[_k, _k : _k + len(boxcar_kernel)] = boxcar_kernel
    _fig, _ax = plt.subplots(figsize=(8, 8))
    _ax.imshow(shifted_kernel, cmap="Reds")
    _ax.set_xlabel("Time (samples)", fontsize=18)
    _ax.set_ylabel("Kernel shifted to sample", fontsize=18)
    _ax.set_title("Time Shifted Kernels", fontsize=18)
    plt.close()
    _fig
    return (shifted_kernel,)


@app.cell(hide_code=True)
def dotconv_md():
    mo.md("""
    Now, let's take the dot product of the signal with this matrix. Matrix multiplication consists of taking the dot product of the signal vector with each row of this expanded kernel matrix.
    """)
    return


@app.cell
def conv_result(boxcar_kernel, conv_signal, shifted_kernel):
    _result = np.dot(conv_signal, shifted_kernel)
    _fig, _ax = plt.subplots(figsize=(12, 5))
    _ax.plot(_result, linewidth=4, alpha=0.4, label="signal · shifted kernels")
    _ax.plot(np.convolve(conv_signal, boxcar_kernel), linewidth=1.5, linestyle="--", color="k", label="np.convolve(signal, kernel)")
    _ax.set_ylabel("Intensity", fontsize=18)
    _ax.set_xlabel("Time (samples)", fontsize=18)
    _ax.set_title("Signal convolved with boxcar kernel", fontsize=18)
    _ax.legend(fontsize=12)
    plt.close()
    mo.vstack([_fig, mo.md(f"After convolution, each spike has become the shape of the kernel. `np.convolve` (dashed) does exactly this matrix multiplication for us.\n\nSignal: **{len(conv_signal)}**, Kernel: **{len(boxcar_kernel)}**, Convolved: **{len(_result)}** samples (the kernel hangs off the end).")])
    return


@app.cell(hide_code=True)
def step_md():
    mo.md("""
    #### Step Through Convolution
    Use the slider to step through the convolution one timepoint at a time. Watch the kernel (red) slide across the signal and see the output build up.
    """)
    return


@app.cell(hide_code=True)
def _make_conv_step(boxcar_kernel, n_samples):
    conv_step = mo.ui.slider(0, n_samples + len(boxcar_kernel) - 2, value=0, step=1, label="Convolution Step")
    return (conv_step,)


@app.cell(hide_code=True)
def stepthrough(boxcar_kernel, conv_signal, conv_step, n_samples):
    _total = n_samples + len(boxcar_kernel) - 1
    _s = conv_step.value
    _full = np.convolve(conv_signal, boxcar_kernel)
    _fig, _axes = plt.subplots(nrows=2, figsize=(14, 7), sharex=True)
    _axes[0].plot(conv_signal, linewidth=2, label="Signal")
    _kp = np.zeros(_total)
    for _t in range(max(0, _s - len(boxcar_kernel) + 1), min(_s + 1, n_samples)):
        _ki = _s - _t
        if 0 <= _ki < len(boxcar_kernel):
            _kp[_t] = boxcar_kernel[_ki]
    _axes[0].fill_between(range(n_samples), 0, _kp[:n_samples], alpha=0.3, color="red", label="Kernel position")
    _axes[0].axvline(x=min(_s, n_samples - 1), color="red", linestyle="--", alpha=0.5)
    _axes[0].set_ylabel("Intensity", fontsize=14)
    _axes[0].set_title(f"Step {_s}/{_total - 1}", fontsize=14)
    _axes[0].legend(fontsize=12)
    _axes[1].plot(_full, linewidth=1, color="lightgray", label="Full result")
    _axes[1].plot(range(_s + 1), _full[:_s + 1], linewidth=2, color="C0", label="Computed so far")
    _axes[1].scatter([_s], [_full[_s]], color="red", s=80, zorder=5)
    _axes[1].set_ylabel("Intensity", fontsize=14)
    _axes[1].set_xlabel("Time (samples)", fontsize=14)
    _axes[1].legend(fontsize=12)
    plt.tight_layout()
    plt.close()
    mo.vstack([conv_step, _fig])
    return


@app.cell(hide_code=True)
def vary_md():
    mo.md("""
    What happens if the spikes have different intensities?
    """)
    return


@app.cell
def varying(boxcar_kernel, n_samples):
    _sig = np.zeros(n_samples)
    _sig[np.random.randint(0, n_samples, 5)] = np.random.randint(1, 5, 5)
    _fig, _axes = plt.subplots(nrows=2, figsize=(18, 6), sharex=True)
    _axes[0].plot(_sig, linewidth=2)
    _axes[0].set_ylabel("Intensity", fontsize=18)
    _axes[0].set_title("Spikes with varying intensities", fontsize=18)
    _axes[1].plot(np.convolve(_sig, boxcar_kernel), linewidth=2)
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_xlabel("Time (samples)", fontsize=18)
    _axes[1].set_title("Convolved with boxcar kernel", fontsize=18)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def hrf_md():
    mo.md("""
    Now what happens if we switch out the boxcar kernel for a hemodynamic response function (HRF)?

    Here we will use a double gamma hemodynamic function (HRF) developed by Gary Glover.

    Use the sliders to explore how the TR and oversampling affect the HRF shape.

    Oversampling the function will help make it look more smooth. In practice we will want to make sure that the kernel is the correct shape given our sampling resolution. Be sure to set the oversampling to 1. Notice how the function looks more jagged now?
    """)
    return


@app.cell(hide_code=True)
def _make_hrf_sliders():
    tr_slider = mo.ui.slider(0.5, 4, value=2, step=0.5, label="TR (seconds)")
    oversampling_slider = mo.ui.slider(1, 20, value=20, step=1, label="Oversampling")
    return oversampling_slider, tr_slider


@app.cell(hide_code=True)
def hrf_plot(oversampling_slider, tr_slider):
    hrf_kernel = glover_hrf(tr_slider.value, oversampling=oversampling_slider.value)
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _dt = tr_slider.value / oversampling_slider.value
    _ax.plot(np.arange(len(hrf_kernel)) * _dt, hrf_kernel, "-o" if oversampling_slider.value <= 2 else "-", linewidth=2, color="red")
    _ax.set_ylabel("Intensity", fontsize=18)
    _ax.set_xlabel("Time after the event (s)", fontsize=18)
    _ax.set_title(f"HRF (TR={tr_slider.value}s, oversampling={oversampling_slider.value})", fontsize=16)
    _ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.close()
    mo.vstack([mo.hstack([tr_slider, oversampling_slider]), _fig])
    return (hrf_kernel,)


@app.cell(hide_code=True)
def hrfconv_md():
    mo.md("""
    Now let's convolve our event pulses with this HRF kernel.

    If you are interested in a more detailed overview of convolution in the time domain, I encourage you to watch this [video](https://youtu.be/9Hk-RAIzOaw) by Mike X Cohen. For more details about convolution and the HRF function, see this [overview](https://practical-neuroimaging.github.io/on_convolution.html) using python examples.
    """)
    return


@app.cell
def hrf_conv(hrf_kernel, oversampling_slider, tr_slider):
    # Build the events on the same time grid as the HRF (TR / oversampling).
    _dt = tr_slider.value / oversampling_slider.value
    _t = np.arange(0, 200, _dt)
    _sig = np.zeros(len(_t))
    _onsets = np.random.uniform(0, 160, 5)
    _sig[(_onsets / _dt).astype(int)] = np.random.randint(1, 5, 5)
    _fig, _axes = plt.subplots(nrows=2, figsize=(18, 6), sharex=True)
    _axes[0].plot(_t, _sig, linewidth=2)
    _axes[0].set_ylabel("Intensity", fontsize=18)
    _axes[0].set_title("Events with varying intensities", fontsize=18)
    _axes[1].plot(_t, np.convolve(_sig, hrf_kernel)[: len(_t)], linewidth=2)
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_xlabel("Time (s)", fontsize=18)
    _axes[1].set_title("Convolved with the HRF: predicted BOLD response", fontsize=18)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def osc_intro():
    mo.md("""
    ### Oscillations

    Ok, now let’s move on to studying time-varying signals that have the shape of oscillating waves.

    Let’s watch a short video by Mike X Cohen to get some more background on sine waves. Don’t worry too much about the matlab code as we will work through similar Python examples in this notebook.
    """)
    return


@app.cell(hide_code=True)
def sine_vid(youtube):
    youtube("9RvZXZ46FRQ")
    return


@app.cell(hide_code=True)
def osc_math():
    mo.md(r"""
    Oscillations can be described mathematically as:

    $A\sin(2 \pi ft + \theta)$

    where $f$ is frequency or the speed of the oscillation described in the number of cycles per second ($Hz$), Amplitude $A$ refers to the height of the waves, which is half the distance of the peak to the trough. Finally, $\theta$ describes the phase angle offset, which is in radians.

    Here we will plot a simple sine wave. Try playing with the different parameters (i.e., amplitude, frequency, & theta) to gain an intuition of how they each impact the shape of the wave.

    Try the sliders:
    """)
    return


@app.cell(hide_code=True)
def _make_osc_sliders():
    amp_slider = mo.ui.slider(0, 10, value=5, step=0.5, label="Amplitude")
    freq_slider = mo.ui.slider(0.5, 15, value=5, step=0.5, label="Frequency (Hz)")
    theta_slider = mo.ui.slider(-3.14, 3.14, value=0, step=0.1, label="Phase (θ)")
    osc_sf = 500
    return amp_slider, freq_slider, osc_sf, theta_slider


@app.cell(hide_code=True)
def osc_plot(amp_slider, freq_slider, osc_sf, theta_slider):
    _time = arange(0, 2, 1 / osc_sf)
    _sim = amp_slider.value * sin(2 * pi * freq_slider.value * _time + theta_slider.value)
    _fig, _ax = plt.subplots(figsize=(16, 4))
    _ax.plot(_time, _sim, linewidth=2)
    _ax.axhline(0, color="gray", linewidth=0.8)
    _ax.set_ylabel("Amplitude", fontsize=18)
    _ax.set_xlabel("Time (s)", fontsize=18)
    _ax.set_ylim(-12, 12)
    plt.tight_layout()
    plt.close()
    mo.vstack([mo.hstack([amp_slider, freq_slider, theta_slider]), _fig])
    return


@app.cell(hide_code=True)
def multi_md():
    mo.md("""
    Next we will build a more interesting signal by adding together sine waves at five different frequencies, each with its own amplitude and phase.
    """)
    return


@app.cell
def multi_sine():
    multi_freqs = [3, 10, 5, 15, 35]
    multi_amps = [5, 15, 10, 5, 7]
    multi_phases = pi * np.array([1 / 7, 1 / 8, 1, 1 / 2, -1 / 4])
    _time = arange(0, 2, 1 / 500)
    _waves = np.array([multi_amps[_j] * sin(2 * pi * _f * _time + multi_phases[_j]) for _j, _f in enumerate(multi_freqs)])

    _fig, _axes = plt.subplots(nrows=6, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 2]})
    for _j, _f in enumerate(multi_freqs):
        _axes[_j].plot(_time, _waves[_j], linewidth=1.5)
        _axes[_j].set_ylabel(f"{_f} Hz", rotation=0, ha="right", va="center", fontsize=12)
        _axes[_j].set_yticks([])
    _axes[5].plot(_time, _waves.sum(axis=0), linewidth=1.8, color="purple")
    _axes[5].set_ylabel("Sum", rotation=0, ha="right", va="center", fontsize=12)
    _axes[5].set_xlabel("Time (s)", fontsize=14)
    _axes[5].set_xlim(0, 1)
    _axes[0].set_title("Five sine waves (top) and their sum (bottom)", fontsize=16)
    plt.tight_layout()
    plt.close()
    _fig
    return multi_amps, multi_freqs, multi_phases


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    What is the effect of changing the sampling frequency on our ability to measure these oscillations? Try dropping it to be very low (e.g., less than 70 hz.) Notice that signals will alias when the sampling frequency is below the nyquist frequency of a signal. To observe the oscillations, we need to be sampling at least two times for each oscillation cycle. This will result in a jagged view of the data, but we can still theoretically observe the frequency. Practically, higher sampling rates allow us to better observe the underlying signals.
    """)
    return


@app.cell(hide_code=True)
def _make_sf_sliders():
    sf_slider = mo.ui.slider(30, 500, value=500, step=10, label="Sampling Frequency (Hz)")
    noise_slider = mo.ui.slider(0, 20, value=5, step=1, label="Noise Level")
    return noise_slider, sf_slider


@app.cell(hide_code=True)
def combined(multi_amps, multi_freqs, multi_phases, noise_slider, sf_slider):
    _t_true = arange(0, 2, 1 / 2000)
    _t = arange(0, 2, 1 / sf_slider.value)

    _waves_true = np.array([multi_amps[_j] * sin(2 * pi * _f * _t_true + multi_phases[_j]) for _j, _f in enumerate(multi_freqs)])
    _waves = np.array([multi_amps[_j] * sin(2 * pi * _f * _t + multi_phases[_j]) for _j, _f in enumerate(multi_freqs)])

    _sampled = np.sum(_waves, axis=0) + noise_slider.value * np.random.randn(len(_t))
    _combined_true = np.sum(_waves_true, axis=0)

    _nyquist = sf_slider.value / 2

    def _apparent(f, sf):
        folded = f % sf
        return folded if folded <= sf / 2 else sf - folded

    _fig, _axes = plt.subplots(nrows=6, figsize=(12, 11), sharex=True)

    for _j, _f in enumerate(multi_freqs):
        _ax = _axes[_j]
        _ax.plot(_t_true, _waves_true[_j], color="gray", alpha=0.45, linewidth=1, label="true signal")
        _is_aliased = _f > _nyquist
        _color = "tab:red" if _is_aliased else "tab:blue"
        _ax.plot(_t, _waves[_j], "o-", color=_color, linewidth=1.3, markersize=4, label="sampled")
        _ylabel = f"{_f} Hz"
        if _is_aliased:
            _ylabel += f"\n→ looks like\n{_apparent(_f, sf_slider.value):.1f} Hz"
        _ax.set_ylabel(_ylabel, fontsize=10)
        _ax.grid(alpha=0.3)

    _axes[0].set_title(f"Sampling: {sf_slider.value} Hz   |   Nyquist: {_nyquist} Hz", fontsize=16)
    _axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    _axes[5].plot(_t_true, _combined_true, color="gray", alpha=0.45, linewidth=1, label="true sum")
    _axes[5].plot(_t, _sampled, "o-", color="purple", linewidth=1.3, markersize=4, label=f"sampled + noise ({noise_slider.value})")
    _axes[5].set_ylabel("Sum", fontsize=10)
    _axes[5].set_xlabel("Time (s)", fontsize=14)
    _axes[5].legend(loc="upper right", fontsize=8, ncol=2)
    _axes[5].grid(alpha=0.3)

    _axes[0].set_xlim(0, 0.5)
    plt.tight_layout()
    plt.close()

    _aliased = [(f, _apparent(f, sf_slider.value)) for f in multi_freqs if f > _nyquist]
    if _aliased:
        _msg = "**Aliasing!** " + ", ".join(f"{f} Hz → appears as {a:.1f} Hz" for f, a in _aliased) + f"  (Nyquist = {_nyquist} Hz)"
        _warn = mo.callout(mo.md(_msg), kind="warn")
    else:
        _warn = mo.callout(mo.md(f"All frequencies below Nyquist ({_nyquist} Hz) — no aliasing."), kind="success")

    mo.vstack([mo.hstack([sf_slider, noise_slider]), _warn, _fig])
    return


@app.cell(hide_code=True)
def fmri_alias_md():
    mo.md(r"""
    #### Aliasing at fMRI sampling rates

    This matters for fMRI. We sample the brain once per **TR** (repetition time), so the sampling frequency is $f_s = 1/\text{TR}$. A typical TR of 2 s gives $f_s$ = 0.5 Hz and a Nyquist frequency of only **0.25 Hz**. Breathing (~0.3 Hz) and the heartbeat (~1 Hz) are faster than that, so they can't be measured directly. They alias, folding down into slower oscillations.

    Use the slider to change the TR. The gray line is the true physiological signal, and the dots are what the scanner records.
    """)
    return


@app.cell(hide_code=True)
def _make_fmri_tr_slider():
    fmri_tr_slider = mo.ui.slider(0.4, 3.0, value=2.0, step=0.1, label="TR (s)", full_width=True)
    return (fmri_tr_slider,)


@app.cell(hide_code=True)
def fmri_aliasing(fmri_tr_slider):
    def _apparent(f, sf):
        folded = f % sf
        return folded if folded <= sf / 2 else sf - folded

    _tr = fmri_tr_slider.value
    _fs = 1 / _tr
    _nyq = _fs / 2
    _t_true = np.arange(0, 60, 0.02)
    _t = np.arange(0, 60, _tr)

    _fig, _axes = plt.subplots(nrows=2, figsize=(14, 5), sharex=True)
    for _ax, (_name, _f) in zip(_axes, [("Respiration", 0.3), ("Cardiac", 1.1)]):
        _aliased = _f > _nyq
        _app = _apparent(_f, _fs)
        _ax.plot(_t_true, np.sin(2 * pi * _f * _t_true), color="gray", alpha=0.35, linewidth=1)
        _ax.plot(_t, np.sin(2 * pi * _f * _t), "o-", color="tab:red" if _aliased else "tab:blue", linewidth=1.6, markersize=5)
        _ax.set_ylabel(f"{_name}\n{_f} Hz", rotation=0, ha="right", va="center", fontsize=12)
        _ax.set_yticks([])
        if _aliased and _app > 0:
            _note = f"aliased: appears at {_app:.3f} Hz (one cycle every {1 / _app:.0f} s)"
        elif _aliased:
            _note = "aliased: appears as a constant"
        else:
            _note = "sampled fast enough"
        _ax.set_title(_note, fontsize=12, loc="right", color="tab:red" if _aliased else "tab:blue")
    _axes[-1].set_xlabel("Time (s)", fontsize=14)
    plt.tight_layout()
    plt.close()

    mo.vstack([
        fmri_tr_slider,
        mo.md(f"TR = {_tr:g} s → $f_s$ = {_fs:.2f} Hz → Nyquist = **{_nyq:.2f} Hz**"),
        _fig,
        mo.callout(mo.md(
            "Task designs typically put their signal around **0.01–0.1 Hz**. At a 2 s TR, an aliased heartbeat "
            "can land right in that band, where no filter can separate it from the task. This is one reason "
            "fast multiband sequences (TR < 1 s) and physiological recordings are used to model cardiac and "
            "respiratory noise."
        ), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def tf_intro():
    mo.md("""
    ## Time & Frequency Domains
    We have seen above how to represent signals in the time domain. However, these signals can also be represented in the frequency domain.

    Let’s get started by watching a short video by Mike X Cohen to get an overview of how a signal can be represented in both of these different domains.
    """)
    return


@app.cell(hide_code=True)
def tf_video(youtube):
    youtube("fYtVHhk3xJ0")
    return


@app.cell(hide_code=True)
def signal_md():
    mo.md(r"""
    For the rest of this chapter we'll analyze one fixed version of our five-sine-wave signal: sampled at **500 Hz** (well above Nyquist for all five components) for exactly **2 seconds**, plus some noise. Every spectrum and filter below uses this same signal.

    Why exactly 2 s? Each component then completes a whole number of cycles in the recording. If a wave stops partway through a cycle, its energy smears into neighboring frequencies in the spectrum (called *spectral leakage*), and its peak comes out lower than its true amplitude.
    """)
    return


@app.cell
def freq_signal(multi_amps, multi_freqs, multi_phases):
    signal_fs = 500
    signal_time = arange(0, 2, 1 / signal_fs)
    combined_signal = sum(
        _a * sin(2 * pi * _f * signal_time + _p) for _f, _a, _p in zip(multi_freqs, multi_amps, multi_phases)
    ) + 5 * np.random.default_rng(0).standard_normal(len(signal_time))

    _fig, _ax = plt.subplots(figsize=(16, 3.5))
    _ax.plot(signal_time, combined_signal, linewidth=1.2)
    _ax.set_xlabel("Time (s)", fontsize=14)
    _ax.set_ylabel("Amplitude", fontsize=14)
    _ax.set_title("Our signal: five sine waves + noise, 500 Hz for 2 s", fontsize=16)
    plt.tight_layout()
    plt.close()
    _fig
    return combined_signal, signal_fs, signal_time


@app.cell(hide_code=True)
def freq_intro():
    mo.md(r"""
    ## Frequency Domain

    In the previous example, we generated a complex signal composed of multiple sine waves oscillating at different frequencies. Typically in data analysis, we only observe the signal and are trying to uncover the generative processes that gave rise to the signal. In this section, we will introduce the frequency domain and how we can identify if there are any frequencies oscillating at a consistent frequency in our signal using the fourier transform. The fourier transform convolves different frequencies of sine waves with our data to identify oscillatory components.

    One important assumption: **stationarity** — the generative processes don't vary over time.

    See this [video](https://youtu.be/rea6M1oagmA) or a more in depth discussion on stationarity. In practice, this assumption is rarely true. Often it can be useful to use other techniques such as wavelets to look at time x frequency representations. We will not be covering wavelets here, but see this series of [videos](https://www.youtube.com/watch?v=7ahrcB5HL0k) for more information.

    ### Discrete Time Fourier Transform
    We will gain an intution of how the fourier transform works by building our own discrete time fourier transform.

    Let’s watch this short video about the fourier transform by Mike X Cohen. Don’t worry too much about the details of the discussion on the matlab code as we will be exploring these concepts in python below.
    """)
    return


@app.cell(hide_code=True)
def dft_vid(youtube):
    youtube("_htCsieA0_U")
    return


@app.cell(hide_code=True)
def dft_math():
    mo.md(r"""
    The discrete Fourier transform of variable $x$ at frequency
    $f$ can be defined as:

    $$X_f = \sum\limits_{k=0}^{n-1} x_k \cdot e^{\frac{-i2\pi fk}{n}}$$

    where $n$ refers to the number of data points in vector $x$, and the capital letter $X_f$ is the fourier coefficient of time series variable $x$ at frequency $f$.

    Essentially, we create a bank of complex sine waves at different frequencies that are linearly spaced. The zero frequency component reflects the mean offset over the entire signal and will simply be zero in our example.

    #### Complex Sine Waves
    You may have noticed that we are computing complex sine waves using the `np.exp` function instead of the `np.sin` function.

    $$e^{i(2\pi ft + \theta)}$$

    We will not spend too much time on the details, but basically complex sine waves have three components: time, a real part of the sine wave, and the imaginary part of the sine wave, which are basically phase shifted by $\frac{\pi}{2}$. $1j$ is how we can specify a complex number in python. We can extract the real components using `np.real` or the imaginary using `np.imag`.

    We can visualize complex sine waves in three dimensions. For more information, watch this [video](https://www.youtube.com/watch?v=iZCDOuzfsY0). If you need a refresher on complex numbers, you may want to watch this [video](https://www.youtube.com/watch?v=fNfXKiIIufY).

    In this plot, we show this complex signal in 3 dimensions and also project on two dimensional planes to show that the real and imaginary create a unit circle, and are phase offset by $\frac{\pi}{2}$ with respect to time.
    """)
    return


@app.cell(hide_code=True)
def _make_cx_sliders():
    cx_freq_slider = mo.ui.slider(1, 15, value=5, step=1, label="Frequency (Hz)")
    cx_theta_slider = mo.ui.slider(-3.14, 3.14, value=0, step=0.1, label="Phase (θ)")
    return cx_freq_slider, cx_theta_slider


@app.cell(hide_code=True)
def cx_3d(cx_freq_slider, cx_theta_slider, osc_sf):
    from matplotlib.collections import LineCollection
    import matplotlib.patches as mpatches

    _time = arange(-1, 1 + 1 / osc_sf, 1 / osc_sf)
    _z = exp(1j * (2 * pi * cx_freq_slider.value * _time + cx_theta_slider.value))
    _phi = cx_theta_slider.value
    _t0 = int(np.argmin(np.abs(_time)))
    _x0, _y0 = real(_z)[_t0], imag(_z)[_t0]

    _fig = plt.figure(figsize=(15, 10))

    _ax1 = _fig.add_subplot(2, 2, 1, projection="3d")
    _ax1.plot(np.arange(len(_time)) / osc_sf, real(_z), imag(_z))
    _ax1.set_xlabel("Time (sec)", fontsize=14)
    _ax1.set_ylabel("Real(z)", fontsize=14)
    _ax1.set_zlabel("Imaginary(z)", fontsize=14)
    _ax1.set_title("Complex Sine Wave", fontsize=16)
    _ax1.view_init(15, 250)

    _ax2 = _fig.add_subplot(2, 2, 2)
    _pts = np.column_stack([real(_z), imag(_z)])
    _segs = np.concatenate([_pts[:-1, None, :], _pts[1:, None, :]], axis=1)
    _lc = LineCollection(_segs, cmap="viridis", array=_time[:-1], linewidth=1.8)
    _ax2.add_collection(_lc)
    _u = np.linspace(0, 2 * pi, 200)
    _ax2.plot(np.cos(_u), np.sin(_u), color="gray", alpha=0.25, linestyle="--", linewidth=1)
    _ax2.axhline(0, color="gray", alpha=0.3, linewidth=0.6)
    _ax2.axvline(0, color="gray", alpha=0.3, linewidth=0.6)
    _ax2.plot([0, _x0], [0, _y0], color="red", linewidth=2.2, label="phasor at t=0")
    _ax2.scatter([_x0], [_y0], color="red", s=50, zorder=5)
    _phi_deg = np.degrees(_phi)
    _theta1, _theta2 = (0, _phi_deg) if _phi_deg >= 0 else (_phi_deg, 0)
    _arc = mpatches.Arc((0, 0), 0.6, 0.6, theta1=_theta1, theta2=_theta2, color="red", linewidth=1.8)
    _ax2.add_patch(_arc)
    _ax2.annotate(rf"$\theta={_phi:.2f}$",
                  xy=(0.42 * np.cos(_phi / 2), 0.42 * np.sin(_phi / 2)),
                  color="red", fontsize=12, ha="center", va="center")
    _ax2.set_xlim(-1.35, 1.35)
    _ax2.set_ylim(-1.35, 1.35)
    _ax2.set_aspect("equal")
    _ax2.set_xlabel("Real(z)", fontsize=14)
    _ax2.set_ylabel("Imaginary(z)", fontsize=14)
    _ax2.set_title(rf"Phase plane  ($\theta={_phi:.2f}$ rad)", fontsize=16)
    _ax2.legend(loc="upper right", fontsize=9)
    plt.colorbar(_lc, ax=_ax2, fraction=0.046, pad=0.04, label="Time (s)")

    _ax3 = _fig.add_subplot(2, 2, 3)
    _ax3.plot(np.arange(len(_time)) / osc_sf, real(_z))
    _ax3.set_xlabel("Time (sec)", fontsize=14)
    _ax3.set_ylabel("Real(z)", fontsize=14)
    _ax3.set_title("Real vs Time", fontsize=16)

    _ax4 = _fig.add_subplot(2, 2, 4)
    _ax4.plot(np.arange(len(_time)) / osc_sf, imag(_z))
    _ax4.set_xlabel("Time (sec)", fontsize=14)
    _ax4.set_ylabel("Imaginary(z)", fontsize=14)
    _ax4.set_title("Imaginary vs Time", fontsize=16)

    plt.tight_layout()
    plt.close()
    mo.vstack([mo.hstack([cx_freq_slider, cx_theta_slider]), _fig])
    return


@app.cell(hide_code=True)
def fb_md():
    mo.md("""
    #### Create a filter bank
    Ok, now let’s create a bank of n-1 linearly spaced complex sine waves and the plot first 5 waves to see their frequencies.

    Remember the first basis function is zero frequency (DC) component and reflects the mean offset over the entire signal.
    """)
    return


@app.cell
def filter_bank(combined_signal):
    _t = np.arange(len(combined_signal)) / len(combined_signal)
    sine_bank = np.array([exp(-1j * 2 * pi * _k * _t) for _k in range(len(combined_signal))])
    _fig, _axes = plt.subplots(nrows=5, figsize=(12, 8), sharex=True)
    for _j in range(5):
        _axes[_j].plot(np.real(sine_bank[_j, :]), linewidth=2)
        _axes[_j].set_ylabel(f"f = {_j}", rotation=0, ha="right", va="center", fontsize=12)
    _axes[0].set_title("First 5 waves in the bank (real part): 0, 1, 2, 3 and 4 cycles", fontsize=16)
    _axes[4].set_xlabel("Time (samples)", fontsize=14)
    plt.tight_layout()
    plt.close()
    _fig
    return (sine_bank,)


@app.cell(hide_code=True)
def hm_md():
    mo.md("""
    We can visualize a whole bank of sine waves at once using a heatmap representation. Our bank has one row per sample, too many to draw clearly, so the heatmap shows a smaller bank of 64 waves built the same way. Each row is a different sine wave, and columns reflect time. The intensity of the value is like if the sine wave was coming towards and away rather than up and down. Notice how it looks like that the second half of the sine waves appear to be a mirror image of the first half. This is because the first half contain the positive frequencies, while the second half contains the negative frequencies. Negative frequencies capture sine waves that travel in reverse order around the complex plane compared to that travel forward. This becomes more relevant with the hilbert transform, but for the purposes of this tutorial we will be ignoring the negative frequencies.
    """)
    return


@app.cell
def fb_heatmap():
    # A 64-point bank: the real one (one row per sample of our signal) is too
    # dense to draw. Squeezed into a few hundred pixels it aliases into a moire
    # pattern -- the same undersampling problem we just saw with sine waves.
    _k = np.arange(64)
    _small_bank = exp(-1j * 2 * pi * np.outer(_k, _k) / 64)
    _fig, _ax = plt.subplots(figsize=(8, 8))
    _ax.imshow(np.real(_small_bank), cmap="RdBu_r", interpolation="nearest")
    _ax.axhline(31.5, color="k", linestyle="--", linewidth=1)
    _ax.set_ylabel("Frequency", fontsize=18)
    _ax.set_xlabel("Time", fontsize=18)
    _ax.set_title("A bank of 64 complex sine waves (real part)", fontsize=16)
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def fc_md():
    mo.md("""
    #### Estimate Fourier Coefficients
    Now let’s take the dot product of each of the sine wave basis set with our signal to get the fourier coefficients.

    We can scale the coefficients to be more interpretable by dividing by the number of time points and multiplying by 2. Watch this [video](https://youtu.be/Ee9btm3tros) if you’re interested in a more detailed explanation. Basically, this only needs to be done if you want the amplitude to be in the same units as the original data. In practice, this scaling factor will not change your interpretation of the spectrum.
    """)
    return


@app.cell
def dft_compute(combined_signal, multi_amps, multi_freqs, signal_fs, sine_bank):
    fourier_coeffs = 2 * np.dot(combined_signal, sine_bank) / len(combined_signal)
    dft_freq_axis = fftfreq(len(combined_signal), 1 / signal_fs)
    _n_pos = len(combined_signal) // 2
    _fig, _ax = plt.subplots(figsize=(14, 5))
    _ax.plot(dft_freq_axis[:_n_pos], np.abs(fourier_coeffs[:_n_pos]), linewidth=2)
    _ax.scatter(multi_freqs, multi_amps, color="red", s=60, zorder=5, label="true amplitudes")
    _ax.set_xlim(0, 50)
    _ax.set_xlabel("Frequency (Hz)", fontsize=14)
    _ax.set_ylabel("Amplitude", fontsize=14)
    _ax.set_title("Amplitude spectrum of our signal (0 to 50 Hz; nothing above)", fontsize=16)
    _ax.legend(fontsize=12)
    plt.tight_layout()
    plt.close()
    _same = np.allclose(fourier_coeffs, 2 * fft(combined_signal) / len(combined_signal))
    mo.vstack([
        mo.md(f"Recall: `freq = {multi_freqs}`, `amplitude = {multi_amps}`. Each peak lands on (or, because of the noise, very near) its true amplitude, shown as red dots. The small bumps everywhere else are the noise."),
        _fig,
        mo.md(f"Each coefficient belongs to one frequency: coefficient $k$ is $k$ cycles per recording, so with 2 s of data it sits at $k / 2$ Hz. In practice you'd use `np.fft.fft`, which computes exactly these coefficients (`np.allclose` → **{_same}**) much faster: $O(n \\log n)$ operations instead of $O(n^2)$. That's the *fast* Fourier transform."),
    ])
    return (fourier_coeffs,)


@app.cell(hide_code=True)
def fft_details(youtube):
    mo.vstack([mo.md("Let's learn a few more important details about the DFT:"), youtube("RHjqvcKVopg")])
    return


@app.cell(hide_code=True)
def ifft_md():
    mo.md(r"""
    ### Inverse Fourier Transform

    The fourier transform allows you to represent a time series in the frequency domain. This is a lossless operation, meaning that no information in the original signal is lost by the transform. This means that we can reconstruct the original signal by inverting the operation. Thus, we can create a time series with only the frequency domain information using the inverse fourier transform. Watch this [video](https://youtu.be/HFacSL--vps)  if you would like a more in depth explanation.

    $$x_k = \sum\limits_{k=0}^{n-1} X_f \cdot e^\frac{i2\pi fk}{n}$$

    Notice that we are computing the dot product between the complex sine wave and the fourier coefficients $X$ instead of the time series data $x$.
    """)
    return


@app.cell
def ifft_plot(combined_signal, fourier_coeffs, signal_time, sine_bank):
    _recon = np.real(np.dot(fourier_coeffs, np.conj(sine_bank))) / 2
    _fig, _ax = plt.subplots(figsize=(14, 4))
    _ax.plot(signal_time, combined_signal, linewidth=4, alpha=0.35, label="original")
    _ax.plot(signal_time, _recon, linewidth=1.2, linestyle="--", color="k", label="reconstructed from the coefficients")
    _ax.set_xlim(0, 1)
    _ax.set_xlabel("Time (s)", fontsize=14)
    _ax.set_ylabel("Amplitude", fontsize=14)
    _ax.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.close()
    mo.vstack([_fig, mo.md(f"Largest difference from the original: **{np.max(np.abs(_recon - combined_signal)):.1e}**, i.e. rounding error.")])
    return


@app.cell(hide_code=True)
def phase_md():
    mo.md(r"""
    ### Phase Spectrum

    Each Fourier coefficient is a complex number, so it carries two things: its size (the **amplitude**, plotted above) and its angle (the **phase**). The phase says where in its cycle each wave is at time zero.

    Phase is only meaningful at frequencies where there is a real oscillation; elsewhere it is just the phase of the noise. So below we read off the phase at our five frequencies and compare it to the phases we used to build the signal. The FFT measures phase relative to a cosine, and a sine is a cosine shifted by $\pi/2$, so we add $\pi/2$ before comparing.
    """)
    return


@app.cell
def phase_spectra(combined_signal, multi_freqs, multi_phases, signal_fs):
    _F = fft(combined_signal)
    _freqs = fftfreq(len(combined_signal), 1 / signal_fs)
    _bins = [int(np.argmin(np.abs(_freqs - _f))) for _f in multi_freqs]
    _true = np.angle(np.exp(1j * np.asarray(multi_phases)))  # wrap to (-pi, pi]
    _recovered = np.angle(_F[_bins] * np.exp(1j * pi / 2))
    # pi and -pi are the same angle; draw each recovered phase on the same side
    # as its true value so the bars are comparable.
    _recovered = _true + np.angle(np.exp(1j * (_recovered - _true)))
    _x = np.arange(len(multi_freqs))
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.bar(_x - 0.18, _true, width=0.36, label="phase we used", color="gray")
    _ax.bar(_x + 0.18, _recovered, width=0.36, label="phase from the FFT", color="tab:orange")
    _ax.axhline(0, color="k", linewidth=0.8)
    _ax.set_xticks(_x, [f"{_f} Hz" for _f in multi_freqs])
    _ax.set_yticks([-pi, -pi / 2, 0, pi / 2, pi], [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    _ax.set_xlabel("Frequency", fontsize=14)
    _ax.set_ylabel("Phase (radians)", fontsize=14)
    _ax.legend(fontsize=12)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def ct_md():
    mo.md(r"""
    ### Convolution Theorem
    Convolution in the time domain is the same as multiplication in the frequency domain. This means that time domain convolution computations can be performed much more efficiently in the frequency domain via simple multiplication. (The opposite is also true: multiplication in the time domain is the same as convolution in the frequency domain.) Watch this [video](https://youtu.be/hj7j4Q8T3Ck) for an overview of the convolution theorem and convolution in the frequency domain.

    Below, the top row convolves our signal with a short smoothing kernel in time. The bottom row shows the same three things in the frequency domain: the signal's spectrum, times the kernel's spectrum, gives the spectrum of the result.
    """)
    return


@app.cell
def conv_theorem(combined_signal, signal_fs, signal_time):
    # A 20 ms Gaussian smoothing kernel, sampled at the signal's own rate.
    _kt = arange(-0.05, 0.05 + 1 / signal_fs, 1 / signal_fs)
    _kernel = exp(-0.5 * (_kt / 0.01) ** 2)
    _kernel = _kernel / _kernel.sum()
    _n = len(combined_signal) + len(_kernel) - 1
    _in_time = np.convolve(combined_signal, _kernel)
    _in_freq = ifft(fft(combined_signal, n=_n) * fft(_kernel, n=_n)).real
    _diff = np.max(np.abs(_in_time - _in_freq))
    _half = len(_kernel) // 2
    _result = _in_time[_half : _half + len(combined_signal)]  # re-centered on the signal

    # Plot the spectra on the signal's own 2 s grid. Zero-padding to the full
    # convolution length (as the check above must) would move the components off
    # their frequency bins and shrink the peaks.
    _N = len(combined_signal)
    _f = fftfreq(_N, 1 / signal_fs)[: _N // 2]
    _S = 2 * np.abs(fft(combined_signal))[: _N // 2] / _N
    _K = np.abs(fft(_kernel, n=_N))[: _N // 2]
    _R = 2 * np.abs(fft(_result))[: _N // 2] / _N

    _fig, _axes = plt.subplots(2, 3, figsize=(18, 7))
    _axes[0, 0].plot(signal_time, combined_signal, linewidth=1)
    _axes[0, 0].set_title("signal", fontsize=14)
    _axes[0, 1].plot(_kt * 1000, _kernel, linewidth=2, color="tab:red")
    _axes[0, 1].set_title("∗  kernel", fontsize=14)
    _axes[0, 1].set_xlabel("Time from kernel center (ms)", fontsize=12)
    _axes[0, 2].plot(signal_time, _result, linewidth=1.5, color="tab:green")
    _axes[0, 2].set_title("=  smoothed signal", fontsize=14)
    for _ax in (_axes[0, 0], _axes[0, 2]):
        _ax.set_xlim(0, 1)
        _ax.set_xlabel("Time (s)", fontsize=12)
    _axes[1, 0].plot(_f, _S, linewidth=1.5)
    _axes[1, 0].set_title("spectrum of signal", fontsize=14)
    _axes[1, 1].plot(_f, _K, linewidth=2, color="tab:red")
    _axes[1, 1].set_title("×  spectrum of kernel (its gain)", fontsize=14)
    _axes[1, 2].plot(_f, _R, linewidth=1.5, color="tab:green")
    _axes[1, 2].set_title("=  spectrum of smoothed signal", fontsize=14)
    for _ax in _axes[1]:
        _ax.set_xlim(0, 50)
        _ax.set_xlabel("Frequency (Hz)", fontsize=12)
    _axes[0, 0].set_ylabel("Time domain", fontsize=14)
    _axes[1, 0].set_ylabel("Frequency domain", fontsize=14)
    plt.tight_layout()
    plt.close()
    mo.vstack([
        _fig,
        mo.callout(mo.md(f"Convolving in time and multiplying in frequency give the same result: the largest difference is **{_diff:.1e}**, i.e. rounding error. Notice the kernel's spectrum is a low-pass gain, so smoothing is low-pass filtering."), kind="success"),
    ])
    return


@app.cell(hide_code=True)
def filt_intro():
    mo.md(r"""
    ## Filters

    A filter changes how much of each frequency survives. It can be described in two equivalent ways:

    - In the **frequency domain**, by its **gain**: a number for every frequency, from 1 (keep it) to 0 (remove it).
    - In the **time domain**, by its **kernel**, also called its *impulse response*: the output you get when you feed the filter a single spike.

    These are two views of the same object. The convolution theorem says the gain is the Fourier transform of the kernel. So there are also two ways to *apply* a filter: convolve the signal with the kernel, or multiply the signal's Fourier transform by the gain. Both give the same answer.

    Filters can be classified as finite impulse response (FIR) or infinite impulse response (IIR). These terms describe how a filter responds to a single input impulse. FIR filters have a response that ends at a discrete point in time, while IIR filters have a response that continues indefinitely.

    Filters are designed in the frequency domain and have several properties that need to be considered.

    - ripple in the pass-band
    - attenuation in the stop-band
    - steepness of roll-off
    - filter order (i.e., length for FIR filters)
    - time-domain ringing

    In general, there is a frequency by time tradeoff. The sharper something is in frequency, the broader it is in time, and vice versa.

    ### Two ways to apply the same filter

    Let's apply one low-pass filter both ways. The signal has two components, a slow 5 Hz wave and a fast 35 Hz wave, and the filter should keep everything below 15 Hz. We'll use a simple FIR filter from `scipy.signal.firwin`, whose kernel is just a list of 101 weights.
    """)
    return


@app.cell
def two_way_setup():
    fs_demo = 500
    t_demo = arange(0, 2, 1 / fs_demo)
    x_demo = 10 * sin(2 * pi * 5 * t_demo) + 5 * sin(2 * pi * 35 * t_demo)
    lp_kernel = firwin(101, 15, fs=fs_demo)  # low-pass: keep below 15 Hz
    return fs_demo, lp_kernel, t_demo, x_demo


@app.cell(hide_code=True)
def two_way_time_md():
    mo.md(r"""
    #### Way 1: in the time domain, convolve with the kernel

    This is exactly the convolution from the start of this chapter. Slide the kernel along the signal and take a dot product at every time point. The kernel is a smooth bump, so each output sample is a weighted average of its neighbors. Averaging over ~100 ms smooths away the fast 35 Hz wiggles but barely touches the slow 5 Hz wave.
    """)
    return


@app.cell
def two_way_time(fs_demo, lp_kernel, t_demo, x_demo):
    y_time = np.convolve(x_demo, lp_kernel, mode="same")

    _fig, _axes = plt.subplots(ncols=3, figsize=(18, 4), gridspec_kw={"width_ratios": [2, 1, 2]})
    _axes[0].plot(t_demo, x_demo, linewidth=1.2)
    _axes[0].set_title("signal: 5 Hz + 35 Hz", fontsize=14)
    _axes[1].plot((np.arange(len(lp_kernel)) - len(lp_kernel) // 2) / fs_demo * 1000, lp_kernel, "o-", markersize=3, color="tab:red")
    _axes[1].set_title("∗  low-pass kernel (101 weights)", fontsize=14)
    _axes[1].set_xlabel("Time from kernel center (ms)", fontsize=12)
    _axes[2].plot(t_demo, y_time, linewidth=2, color="tab:green")
    _axes[2].set_title("=  filtered: only the 5 Hz wave is left", fontsize=14)
    for _ax in (_axes[0], _axes[2]):
        _ax.set_xlim(0, 1)
        _ax.set_ylim(-16, 16)
        _ax.set_xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.close()
    _fig
    return (y_time,)


@app.cell(hide_code=True)
def two_way_freq_md():
    mo.md(r"""
    #### Way 2: in the frequency domain, multiply by the gain

    1. Take the FFT of the signal. Remember that the FFT returns **positive and negative** frequencies: the second half of the array is a mirror image of the first. The plots below show only the positive half.
    2. Build the filter's gain at **every** FFT frequency, the negative ones included, so the mirror image stays intact. Here we take the FFT of the kernel, padded to the length of the signal, which takes care of both halves. Because the kernel is symmetric, its gain is real-valued: about 1 below 15 Hz and about 0 above.
    3. Multiply the two, frequency by frequency.
    4. Take the inverse FFT to get back to time. The result should be real; `.real` drops the leftover rounding error.
    """)
    return


@app.cell
def two_way_freq(fs_demo, lp_kernel, t_demo, x_demo, y_time):
    _n = len(x_demo)
    X = fft(x_demo)
    freqs_demo = fftfreq(_n, 1 / fs_demo)

    # Pad the kernel to the signal's length and center it on sample 0, so the
    # filter doesn't shift the signal in time.
    _padded = np.zeros(_n)
    _padded[: len(lp_kernel)] = lp_kernel
    _padded = np.roll(_padded, -(len(lp_kernel) // 2))
    gain = fft(_padded)

    Y = X * gain
    y_freq = ifft(Y).real

    # The two versions differ only within half a kernel of the ends: there,
    # convolution runs off the edge of the signal, while the FFT wraps around.
    _inner = slice(len(lp_kernel), _n - len(lp_kernel))
    _max_diff = np.max(np.abs(y_freq[_inner] - y_time[_inner]))

    _pos = slice(0, _n // 2)
    _fig, _axes = plt.subplots(nrows=3, figsize=(16, 11))
    _axes[0].plot(freqs_demo[_pos], 2 * np.abs(X[_pos]) / _n, linewidth=2)
    _axes[0].set_ylabel("Amplitude", fontsize=14, color="tab:blue")
    _ax_gain = _axes[0].twinx()
    _ax_gain.plot(freqs_demo[_pos], np.real(gain[_pos]), color="tab:red", linewidth=2)
    _ax_gain.set_ylim(-0.05, 1.1)
    _ax_gain.set_ylabel("Gain", fontsize=14, color="tab:red")
    _axes[0].set_xlim(0, 60)
    _axes[0].set_xlabel("Frequency (Hz)", fontsize=12)
    _axes[0].set_title("Steps 1-2: the signal's spectrum (blue) and the filter's gain (red)", fontsize=14)
    _axes[1].plot(freqs_demo[_pos], 2 * np.abs(Y[_pos]) / _n, linewidth=2, color="tab:green")
    _axes[1].set_xlim(0, 60)
    _axes[1].set_xlabel("Frequency (Hz)", fontsize=12)
    _axes[1].set_ylabel("Amplitude", fontsize=14)
    _axes[1].set_title("Step 3: spectrum × gain — the 35 Hz peak is gone", fontsize=14)
    _axes[2].plot(t_demo, y_time, linewidth=4, alpha=0.4, label="Way 1: convolve in time")
    _axes[2].plot(t_demo, y_freq, linewidth=1.5, linestyle="--", color="k", label="Way 2: multiply in frequency")
    _axes[2].set_xlim(0, 1)
    _axes[2].set_xlabel("Time (s)", fontsize=14)
    _axes[2].set_title("Step 4: inverse FFT — the same filtered signal both ways", fontsize=14)
    _axes[2].legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.close()
    mo.vstack([
        _fig,
        mo.md(f"Away from the edges, the two results differ by at most **{_max_diff:.1e}**, i.e. only by rounding error."),
    ])
    return


@app.cell(hide_code=True)
def two_way_why_md():
    mo.md(r"""
    #### Which one is used in practice?

    Both, and you'll meet each:

    - **Frequency domain**: you see exactly what the filter does to every frequency, and for long signals the FFT makes it fast. Watch the **edges**, though: the FFT treats the signal as if it wraps around in a loop, so the end leaks into the beginning.
    - **Time domain**: works on a stream, one sample at a time. Real-time audio does this. The equalizer later in this chapter filters your music in the time domain, with small IIR filters running sample by sample, while its gray spectrum display is an FFT.
    - `scipy.signal.filtfilt`, which we use for the Butterworth filters below, is also a time-domain method. Butterworth filters are IIR, so instead of convolving with a fixed kernel, each output sample is computed from recent inputs *and recent outputs*. `filtfilt` runs the filter forward and then backward, which cancels the time shift a one-way filter would add.

    #### Two easy mistakes in the frequency domain

    1. **Forgetting the negative frequencies.** If you set the gain at +35 Hz to 0 but not at −35 Hz, the spectrum is no longer mirror-symmetric, and the inverse FFT comes back complex instead of real.
    2. **A brick-wall gain.** Jumping straight from 1 to 0 at the cutoff is the sharpest filter possible in frequency, so by the frequency × time trade-off it rings in time. You can see this around sharp edges, like the onsets of a block design.
    """)
    return


@app.cell
def two_way_mistakes(fs_demo, x_demo):
    _n = len(x_demo)
    _freqs = fftfreq(_n, 1 / fs_demo)
    _X = fft(x_demo)

    # Mistake 1: zero +35 Hz only.
    _one_sided = _X.copy()
    _one_sided[(_freqs > 15)] = 0
    _imag = np.max(np.abs(ifft(_one_sided).imag))

    # Mistake 2: brick-wall vs smooth low-pass on a block design.
    _t = arange(0, 2, 1 / fs_demo)
    _blocks = ((_t * 2) % 1 < 0.5).astype(float)
    _B = fft(_blocks)
    _brick = ifft(_B * (np.abs(_freqs) <= 15)).real
    _smooth = filtfilt(*butter(4, 15, fs=fs_demo), _blocks)

    _fig, _ax = plt.subplots(figsize=(16, 4))
    _ax.plot(_t, _blocks, color="gray", linewidth=1, label="block design")
    _ax.plot(_t, _brick, linewidth=2, label="brick-wall gain (rings)")
    _ax.plot(_t, _smooth, linewidth=2, label="smooth roll-off (Butterworth)")
    _ax.set_xlim(0, 1)
    _ax.set_xlabel("Time (s)", fontsize=14)
    _ax.set_title("Mistake 2: a brick-wall cutoff rings around sharp edges", fontsize=14)
    _ax.legend(fontsize=12, loc="upper right")
    plt.close()
    mo.vstack([
        mo.md(f"Mistake 1: zeroing only the positive frequencies above 15 Hz leaves an imaginary part as large as **{_imag:.2f}** after the inverse FFT. Zero the negative ones too (`np.abs(freqs) > 15`) and it drops to rounding error."),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def hp_md():
    mo.md(r"""
    ### Four common filters

    From here on we'll use IIR Butterworth filters, applied in the time domain with `filtfilt`. There are four basic types, named for what they let through:

    - **High-pass** keeps frequencies above a cutoff and removes slow ones (like scanner drift).
    - **Low-pass** keeps frequencies below a cutoff and removes fast ones.
    - **Band-pass** keeps only a range of frequencies. A Morlet wavelet, for example, is a band-pass filter centered on one frequency.
    - **Band-stop** removes a range of frequencies and keeps everything else.

    Each row below shows one filter: its gain in the frequency domain (left; the red lines mark our signal's five components) and what it does to our signal in the time domain (right).
    """)
    return


@app.cell
def four_filters(combined_signal, multi_freqs, signal_fs, signal_time):
    _filters = [
        ("High-pass, 25 Hz", butter(3, 25, btype="highpass", fs=signal_fs)),
        ("Low-pass, 7 Hz", butter(2, 7, btype="lowpass", fs=signal_fs)),
        ("Band-pass, 12-18 Hz", butter(2, [12, 18], btype="bandpass", fs=signal_fs)),
        ("Band-stop, 4-6 Hz", butter(2, [4, 6], btype="bandstop", fs=signal_fs)),
    ]
    _fig, _axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 13), gridspec_kw={"width_ratios": [1, 2]})
    for _row, (_name, (_b, _a)) in zip(_axes, _filters):
        _w, _h = freqz(_b, _a, worN=2048, fs=signal_fs)
        _row[0].plot(_w, np.abs(_h), linewidth=2.5)
        for _f in multi_freqs:
            _row[0].axvline(_f, color="red", linestyle="--", alpha=0.4)
        _row[0].set_xlim(0, 50)
        _row[0].set_ylim(-0.05, 1.1)
        _row[0].set_ylabel("Gain", fontsize=12)
        _row[0].set_title(_name, fontsize=14, loc="left")
        _row[1].plot(signal_time, combined_signal, linewidth=1, alpha=0.4, label="original")
        _row[1].plot(signal_time, filtfilt(_b, _a, combined_signal), linewidth=1.8, label="filtered")
        _row[1].set_xlim(0, 1)
        _row[1].legend(fontsize=10, loc="upper right")
    _axes[-1, 0].set_xlabel("Frequency (Hz)", fontsize=14)
    _axes[-1, 1].set_xlabel("Time (s)", fontsize=14)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def temporal_md():
    mo.md("""
    What does a filter look like in the time domain? Feed it a single spike, and the output is its impulse response. The **order** sets how sharp the cutoff is. Compare a high-pass filter of order 2 and order 8 below: the sharper gain rings for longer in time. That is the frequency × time trade-off again.

    Notice that the response starts *before* the spike. `filtfilt` runs the filter forward and then backward over the whole recording, so its response is symmetric around the spike. That's what keeps it from shifting the signal in time, and it's possible because we filter a recording after the fact, not a live stream.
    """)
    return


@app.cell
def filt_temporal(signal_fs):
    _impulse = np.zeros(400)
    _impulse[200] = 1
    _t_ms = (np.arange(400) - 200) / signal_fs * 1000
    _fig, _axes = plt.subplots(ncols=2, figsize=(16, 4))
    for _order in (2, 8):
        _b, _a = butter(_order, 25, btype="highpass", fs=signal_fs)
        _w, _h = freqz(_b, _a, worN=1024, fs=signal_fs)
        _axes[0].plot(_w, np.abs(_h), linewidth=2.5, label=f"order {_order}")
        _axes[1].plot(_t_ms, filtfilt(_b, _a, _impulse), linewidth=2, label=f"order {_order}")
    _axes[0].set_xlim(0, 100)
    _axes[0].set_xlabel("Frequency (Hz)", fontsize=14)
    _axes[0].set_ylabel("Gain", fontsize=14)
    _axes[0].set_title("Frequency domain: gain", fontsize=16)
    _axes[0].legend(fontsize=12)
    _axes[1].set_xlim(-150, 150)
    _axes[1].set_xlabel("Time relative to the spike (ms)", fontsize=14)
    _axes[1].set_title("Time domain: response to a single spike", fontsize=16)
    _axes[1].legend(fontsize=12)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def explorer_md():
    mo.md("""
    ### Interactive Filter Explorer
    Explore all filter types interactively:
    """)
    return


@app.cell(hide_code=True)
def _make_filt_controls():
    filter_type = mo.ui.dropdown(options=["highpass", "lowpass", "bandpass", "bandstop"], value="highpass", label="Filter Type")
    order_slider = mo.ui.slider(1, 10, value=3, step=1, label="Filter Order")
    cutoff_slider = mo.ui.slider(1, 100, value=25, step=1, label="Cutoff (Hz)")
    cutoff2_slider = mo.ui.slider(1, 100, value=40, step=1, label="Upper Cutoff (Hz)")
    return cutoff2_slider, cutoff_slider, filter_type, order_slider


@app.cell(hide_code=True)
def filter_explorer(
    combined_signal,
    cutoff2_slider,
    cutoff_slider,
    filter_type,
    multi_freqs,
    order_slider,
    signal_fs,
    signal_time,
):
    if filter_type.value in ("bandpass", "bandstop"):
        _lo, _hi = min(cutoff_slider.value, cutoff2_slider.value), max(cutoff_slider.value, cutoff2_slider.value)
        if _lo == _hi:
            _hi = _lo + 1
        _b, _a = butter(order_slider.value, [_lo, _hi], btype=filter_type.value, fs=signal_fs)
    else:
        _b, _a = butter(order_slider.value, cutoff_slider.value, btype=filter_type.value, fs=signal_fs)
    _w, _h = freqz(_b, _a, worN=1024, fs=signal_fs)
    _impulse = np.zeros(400)
    _impulse[200] = 1

    _fig = plt.figure(figsize=(16, 9))
    _gs = _fig.add_gridspec(2, 2)
    _ax0 = _fig.add_subplot(_gs[0, 0])
    _ax0.plot(_w, np.abs(_h), linewidth=2)
    for _f in multi_freqs:
        _ax0.axvline(_f, color="red", linestyle="--", alpha=0.5)
    _ax0.set_xlim(0, 100)
    _ax0.set_ylim(-0.05, 1.1)
    _ax0.set_ylabel("Gain", fontsize=14)
    _ax0.set_xlabel("Frequency (Hz)", fontsize=14)
    _ax0.set_title(f"{filter_type.value.title()} (order {order_slider.value}): gain; red lines are our signal's components", fontsize=13)
    _ax1 = _fig.add_subplot(_gs[0, 1])
    _ax1.plot((np.arange(400) - 200) / signal_fs * 1000, filtfilt(_b, _a, _impulse), linewidth=2, color="green")
    _ax1.set_xlim(-150, 150)
    _ax1.set_xlabel("Time relative to the spike (ms)", fontsize=14)
    _ax1.set_title("Response to a single spike: raise the order and it rings longer", fontsize=13)
    _ax2 = _fig.add_subplot(_gs[1, :])
    _ax2.plot(signal_time, combined_signal, linewidth=1.5, alpha=0.4, label="original")
    _ax2.plot(signal_time, filtfilt(_b, _a, combined_signal), linewidth=2, label="filtered")
    _ax2.set_xlim(0, 1)
    _ax2.set_xlabel("Time (s)", fontsize=14)
    _ax2.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.close()
    mo.vstack([
        mo.hstack([filter_type, order_slider]),
        mo.hstack([cutoff_slider, cutoff2_slider]),
        mo.md("*Upper cutoff only used for bandpass/bandstop*"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def hp_fmri_md():
    mo.md(r"""
    ### High-pass filtering fMRI data

    The most common filter in fMRI preprocessing is a **high-pass** filter. It removes slow scanner drift, which otherwise dwarfs the effects we care about. The simulated voxel below has a block design (20 s on, 20 s off, so the task repeats every 40 s, or 0.025 Hz) sampled at TR = 2 s, plus slow drift and noise.

    The cutoff is usually written as a **period**, in seconds per cycle, rather than a frequency. Use the slider to choose it. Everything slower than that period is removed (the red band in the spectrum). What happens when the cutoff period is shorter than the 40 s task cycle?
    """)
    return


@app.cell(hide_code=True)
def _make_hp_cutoff_slider():
    hp_cutoff_slider = mo.ui.slider(16, 256, value=128, step=8, label="High-pass cutoff (s per cycle)", full_width=True)
    return (hp_cutoff_slider,)


@app.cell(hide_code=True)
def hp_fmri(hp_cutoff_slider):
    _tr, _nvol, _block = 2.0, 240, 20.0
    _t = np.arange(_nvol) * _tr
    _task = np.convolve(((_t // _block) % 2 == 1).astype(float), glover_hrf(_tr))[:_nvol]
    _drift = 3.0 * (_t / _t[-1]) ** 2 - 2.0 * (_t / _t[-1]) + 0.8 * sin(2 * pi * _t / 300)
    _y = _task + _drift + 0.35 * np.random.default_rng(11).standard_normal(_nvol)

    _period = hp_cutoff_slider.value
    _b, _a = butter(2, 1 / _period, btype="highpass", fs=1 / _tr)
    _y_filtered = filtfilt(_b, _a, _y)
    _task_kept = np.std(filtfilt(_b, _a, _task)) / np.std(_task)
    _r_raw = np.corrcoef(_y, _task)[0, 1]
    _r_filtered = np.corrcoef(_y_filtered, _task)[0, 1]

    _freqs = np.fft.rfftfreq(_nvol, _tr)
    _fig, _axes = plt.subplots(ncols=2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    _axes[0].plot(_t, _y - _y.mean(), color="gray", linewidth=1, alpha=0.7, label="raw")
    _axes[0].plot(_t, _y_filtered, linewidth=1.6, label="high-passed")
    _axes[0].plot(_t, _task - _task.mean(), color="k", linestyle="--", linewidth=1.2, label="true task signal")
    _axes[0].set_xlabel("Time (s)", fontsize=14)
    _axes[0].legend(loc="upper left", fontsize=10, ncol=3)
    _axes[1].plot(_freqs, np.abs(np.fft.rfft(_y - _y.mean())), color="gray", linewidth=1.4)
    _axes[1].axvspan(0, 1 / _period, color="tab:red", alpha=0.15, label="removed")
    _axes[1].axvline(1 / (2 * _block), color="k", linestyle="--", linewidth=1.2, label="task (0.025 Hz)")
    _axes[1].set_xlim(0, 0.1)
    _axes[1].set_xlabel("Frequency (Hz)", fontsize=14)
    _axes[1].set_title("Spectrum of the raw signal", fontsize=14)
    _axes[1].legend(fontsize=10)
    plt.tight_layout()
    plt.close()

    if _period < 2 * _block:
        _msg = mo.callout(mo.md(f"The cutoff ({_period} s) is shorter than the task cycle ({2 * _block:g} s), so the filter is removing the **task** too. Only {_task_kept:.0%} of it survives."), kind="danger")
    else:
        _msg = mo.callout(mo.md(f"Correlation with the true task: **{_r_raw:.2f}** raw → **{_r_filtered:.2f}** filtered ({_task_kept:.0%} of the task signal kept)."), kind="success")
    mo.vstack([
        hp_cutoff_slider,
        _fig,
        _msg,
        mo.md("SPM's default cutoff is **128 s** (≈ 0.008 Hz). SPM and nilearn implement it as a set of slow cosine regressors in the GLM rather than a Butterworth filter, but the idea is the same. Low-pass filtering is rarely used for fMRI: it removes little noise and adds temporal autocorrelation that the GLM then has to model."),
    ])
    return


@app.cell(hide_code=True)
def hearing_md():
    mo.md(r"""
    ### Hearing filters

    Filters are easier to understand when you can *hear* them. Music is a mixture of many frequencies at once: the kick drum and bass live below ~250 Hz, voices and most instruments between ~250 Hz and 4 kHz, and the hiss of cymbals and the "air" of a recording above ~4 kHz.

    Drop any song (mp3, wav, …) onto the player below. It loops, and every sound passes through a bank of filters before it reaches your speakers. Each handle sets the **gain** at one frequency, from 1 (pass it through untouched) to 0 (remove it), with low frequencies on the left — exactly like the filter plots above. The dark curve is the filter's actual frequency response, and the gray shading is the spectrum of what you are hearing right now. Your file never leaves your browser.
    """)
    return


@app.cell(hide_code=True)
def _():
    from dartbrains_tools.mr_widgets import EqualizerWidget

    return (EqualizerWidget,)


@app.cell(hide_code=True)
def equalizer(EqualizerWidget):
    eq_view = mo.ui.anywidget(EqualizerWidget())
    eq_view
    return


@app.cell(hide_code=True)
def hearing_prompts_md():
    mo.md(r"""
    Try these:

    1. Click **Low-pass**. Which instruments disappear, and which survive? Now drag the 500 Hz and 1k handles back up — how does moving the cutoff change the sound?
    2. Click **High-pass**. What happens to the bass and drums? Why does the song sound "thin"?
    3. Click **Band-pass**, which keeps only ~500 Hz–1 kHz. Why does it sound like an old telephone or AM radio?
    4. Click **Band-stop** and listen to what is missing. Compare the gray spectrum to the one you see with **Flat**.
    5. Look at the curve between handles. Even a sharp step on the sliders becomes a smooth roll-off — a real filter can't jump from 1 to 0 at a single frequency. How does that relate to the filter order you explored above?
    """)
    return


@app.cell(hide_code=True)
def ex_md():
    mo.md("""
    ## Exercises

    ### Exercise 1: Create a simulated time series with 7 different frequencies with noise
    """)
    return


@app.cell
def exercise_1():
    # Your code here
    ...
    return


@app.cell(hide_code=True)
def ex2_md():
    mo.md("""
    ### Exercise 2: Show that you can identify each signal using a FFT
    """)
    return


@app.cell
def exercise_2():
    # Your code here
    ...
    return


@app.cell(hide_code=True)
def ex3_md():
    mo.md("""
    ### Exercise 3: Remove one frequency with a bandstop filter
    """)
    return


@app.cell
def exercise_3():
    # Your code here
    ...
    return


@app.cell(hide_code=True)
def ex4_md():
    mo.md("""
    ### Exercise 4: Remove frequency with a bandstop filter in the frequency domain and reconstruct the signal in the time domain with the frequency removed and compare it to the original
    """)
    return


@app.cell
def exercise_4():
    # Your code here
    ...
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ### Graded assignment

    The graded version of these exercises is the **Signal Processing assignment** at the
    end of this page, which opens in a drawer from the **Assignment** button in the header.
    Open it from that page (in molab or by download), sign in with your Dartmouth account
    inside the notebook, and submit each question when you are ready.
    """)
    return


@app.cell(hide_code=True)
def _():
    from dartbrains_tools.notebook_utils import assignment_card

    assignment_card("signal-processing")
    return


if __name__ == "__main__":
    app.run()
