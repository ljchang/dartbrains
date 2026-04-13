import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import sin, pi, arange, exp, real, imag
    from numpy.fft import fft, ifft, fftfreq
    from scipy.special import gamma as gamma_func
    from scipy.signal import butter, filtfilt, freqz, sosfreqz

    def youtube(video_id):
        return mo.Html(f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>')

    def glover_hrf(tr_val, oversampling=1):
        _dt = tr_val / oversampling
        _ts = np.arange(0, 32, _dt)
        _peak, _under, _beta = 5.0, 15.0, 0.35
        _h = (
            (_ts / _peak) ** _peak * np.exp(-(_ts - _peak)) / gamma_func(_peak + 1)
            - _beta * (_ts / _under) ** _under * np.exp(-(_ts - _under)) / gamma_func(_under + 1)
        )
        return _h / _h.max()

    def rad_to_hz(w, fs):
        return (w * fs) / (2 * np.pi)


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # Signal Processing Basics
    *Written by Luke Chang*

    In this lab, we will cover the basics of convolution, sine waves, and fourier transforms. This lab is largely based on exercises from Mike X Cohen's excellent book, [Analyzing Neural Data Analysis: Theory and Practice](https://www.amazon.com/Analyzing-Neural-Time-Data-Practice/dp/0262019876). If you are interested in learning in more detail about the basics of EEG and time-series analyses I highly recommend his accessible introduction. I also encourage you to watch his accompanying freely available [*lecturelets*](https://www.youtube.com/channel/UCUR_LsXk7IYyueSnXcNextQ) to learn more about each topic introduced in this notebook.

    > **Interactive version:** [Open this notebook in molab](https://molab.marimo.io/github/ljchang/dartbrains/blob/v2-marimo-migration/content/Signal_Processing.py) to run code, interact with widgets, and modify examples.

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
    _ax.set_ylabel("B", fontsize=18)
    _ax.set_xlabel("A", fontsize=18)
    _ax.set_title("Scatterplot", fontsize=18)
    mo.vstack([mo.md(f"**Dot Product:** {np.dot(dot_a, _b)}"), _fig])

    return (dot_a,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def dot_md2():
    mo.md("""
    What happens when we make the two variables more similar? In the next example we add gaussian noise on top of one of the vectors. What happens to the dot product?
    """)
    return


@app.cell
def dot_product_similar(dot_a):
    _b2 = dot_a + np.random.randn(20)
    _fig, _ax = plt.subplots()
    _ax.scatter(dot_a, _b2)
    _ax.set_ylabel("B", fontsize=18)
    _ax.set_xlabel("A", fontsize=18)
    _ax.set_title("Scatterplot", fontsize=18)
    mo.vstack([mo.md(f"**Dot Product:** {np.dot(dot_a, _b2):.2f}"), _fig])
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
    _axes[0].set_xlabel("Time", fontsize=18)
    _axes[0].set_ylabel("Signal Intensity", fontsize=18)
    _axes[0].set_title("Signal", fontsize=18)
    _axes[1].plot(boxcar_kernel, linewidth=2, color="red")
    _axes[1].set_xlabel("Time", fontsize=18)
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_title("Kernel", fontsize=18)
    plt.tight_layout()
    _fig
    return boxcar_kernel, conv_signal, n_samples


@app.cell(hide_code=True)
def shift_md():
    mo.md("""
    Notice how the kernel is only 10 samples long and the boxcar width is about 6 seconds, while the signal is 100 samples long with 5 single pulses.

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
    _ax.set_xlabel("Time", fontsize=18)
    _ax.set_ylabel("Time", fontsize=18)
    _ax.set_title("Time Shifted Kernels", fontsize=18)
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
    _ax.plot(_result, linewidth=2)
    _ax.set_ylabel("Intensity", fontsize=18)
    _ax.set_xlabel("Time", fontsize=18)
    _ax.set_title("Signal convolved with boxcar kernel", fontsize=18)
    mo.vstack([_fig, mo.md(f"You can see that after convolution, each spike has now become the shape of the kernel.\n\nSignal: **{len(conv_signal)}**, Kernel: **{len(boxcar_kernel)}**, Convolved: **{len(_result)}** samples")])
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
    _axes[1].set_xlabel("Time", fontsize=14)
    _axes[1].legend(fontsize=12)
    plt.tight_layout()
    plt.close()
    mo.vstack([conv_step, _fig])
    return


@app.cell(hide_code=True)
def npconv_md():
    mo.md("""
    This process can be performed using `np.convolve`:
    """)
    return


@app.cell
def np_convolve(boxcar_kernel, conv_signal):
    _fig, _ax = plt.subplots(figsize=(12, 5))
    _ax.plot(np.convolve(conv_signal, boxcar_kernel), linewidth=2)
    _ax.set_ylabel("Intensity", fontsize=18)
    _ax.set_xlabel("Time", fontsize=18)
    _ax.set_title("Signal convolved with boxcar kernel", fontsize=18)
    _fig
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
    _axes[1].set_xlabel("Time", fontsize=18)
    _axes[1].set_title("Convolved with boxcar kernel", fontsize=18)
    plt.tight_layout()
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
    _ax.plot(hrf_kernel, linewidth=2, color="red")
    _ax.set_ylabel("Intensity", fontsize=18)
    _ax.set_xlabel("Time", fontsize=18)
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
def hrf_conv(hrf_kernel, n_samples):
    _sig = np.zeros(n_samples)
    _sig[np.random.randint(0, n_samples, 5)] = np.random.randint(1, 5, 5)
    _fig, _axes = plt.subplots(nrows=2, figsize=(18, 6), sharex=True)
    _axes[0].plot(_sig, linewidth=2)
    _axes[0].set_ylabel("Intensity", fontsize=18)
    _axes[0].set_title("Spikes with varying intensities", fontsize=18)
    _axes[1].plot(np.convolve(_sig, hrf_kernel), linewidth=2)
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_xlabel("Time", fontsize=18)
    _axes[1].set_title("Convolved with HRF kernel", fontsize=18)
    plt.tight_layout()
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
def sine_vid():
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
    _time = arange(-1, 1 + 1 / osc_sf, 1 / osc_sf)
    _sim = amp_slider.value * sin(2 * pi * freq_slider.value * _time + theta_slider.value)
    _fig = plt.figure(figsize=(20, 4))
    _gs = plt.GridSpec(1, 6, left=0.05, right=0.95, wspace=0.3)
    _ax1 = _fig.add_subplot(_gs[0, :4])
    _ax1.plot(_time, _sim, linewidth=2)
    _ax1.set_ylabel("Amplitude", fontsize=18)
    _ax1.set_xlabel("Time", fontsize=18)
    _ax1.set_ylim(-12, 12)
    _ax2 = _fig.add_subplot(_gs[0, 5:], polar=True)
    _ax2.plot(real(_sim), imag(_sim))
    plt.tight_layout()
    mo.vstack([mo.hstack([amp_slider, freq_slider, theta_slider]), _fig])
    return


@app.cell(hide_code=True)
def multi_md():
    mo.md("""
    Next we will generate a simulation combining multiple sine waves. Try dropping the sampling frequency below 70 Hz to see aliasing. Add noise to make it more realistic.
    """)
    return


@app.cell
def multi_sine():
    multi_freqs = [3, 10, 5, 15, 35]
    multi_amps = [5, 15, 10, 5, 7]
    multi_phases = pi * np.array([1 / 7, 1 / 8, 1, 1 / 2, -1 / 4])
    time = arange(-1, 1 + 1 / 500, 1 / 500)
    waves = np.array([multi_amps[_j] * sin(2 * pi * _f * time + multi_phases[_j]) for _j, _f in enumerate(multi_freqs)])
    _fig, _axes = plt.subplots(nrows=5, figsize=(12, 5), sharex=True)
    for _j in range(5):
        _axes[_j].plot(waves[_j, :], linewidth=2)
    _axes[0].set_title("Sine waves at different frequencies", fontsize=18)
    _axes[4].set_xlabel("Time", fontsize=18)
    plt.tight_layout()
    _fig
    return multi_amps, multi_freqs, multi_phases, waves


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let’s add all of those signals together to get a more interesting signal.
    """)
    return


@app.cell
def _(waves):
    _fig = plt.figure(figsize=(12,3))
    plt.plot(np.sum(waves, axis=0), linewidth=2)
    plt.xlabel('Time', fontsize=18)
    plt.title("Sum of all of the sine waves", fontsize=18)
    plt.xlabel("Time", fontsize=18)
    plt.tight_layout()
    _fig
    return


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
    _t_true = arange(-1, 1, 1 / 1000)
    _t = arange(-1, 1 + 1 / sf_slider.value, 1 / sf_slider.value)

    _waves_true = np.array([multi_amps[_j] * sin(2 * pi * _f * _t_true + multi_phases[_j]) for _j, _f in enumerate(multi_freqs)])
    _waves = np.array([multi_amps[_j] * sin(2 * pi * _f * _t + multi_phases[_j]) for _j, _f in enumerate(multi_freqs)])

    combined_signal = np.sum(_waves, axis=0) + noise_slider.value * np.random.randn(len(_t))
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
    _axes[5].plot(_t, combined_signal, "o-", color="purple", linewidth=1.3, markersize=4, label=f"sampled + noise ({noise_slider.value})")
    _axes[5].set_ylabel("Sum", fontsize=10)
    _axes[5].set_xlabel("Time (s)", fontsize=14)
    _axes[5].legend(loc="upper right", fontsize=8, ncol=2)
    _axes[5].grid(alpha=0.3)

    _axes[0].set_xlim(-0.25, 0.25)
    plt.tight_layout()
    plt.close()

    _aliased = [(f, _apparent(f, sf_slider.value)) for f in multi_freqs if f > _nyquist]
    if _aliased:
        _msg = "**Aliasing!** " + ", ".join(f"{f} Hz → appears as {a:.1f} Hz" for f, a in _aliased) + f"  (Nyquist = {_nyquist} Hz)"
        _warn = mo.callout(mo.md(_msg), kind="warn")
    else:
        _warn = mo.callout(mo.md(f"All frequencies below Nyquist ({_nyquist} Hz) — no aliasing."), kind="success")

    mo.vstack([mo.hstack([sf_slider, noise_slider]), _warn, _fig])
    return (combined_signal,)


@app.cell(hide_code=True)
def tf_intro():
    mo.md("""
    ## Time & Frequency Domains
    We have seen above how to represent signals in the time domain. However, these signals can also be represented in the frequency domain.

    Let’s get started by watching a short video by Mike X Cohen to get an overview of how a signal can be represented in both of these different domains.
    """)
    return


@app.cell(hide_code=True)
def tf_video():
    youtube("fYtVHhk3xJ0")
    return


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
def dft_vid():
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
        _axes[_j].plot(sine_bank[_j, :], linewidth=2)
    _axes[0].set_title("Bank of sine waves", fontsize=18)
    _axes[4].set_xlabel("Time", fontsize=18)
    plt.tight_layout()
    plt.close()
    _fig
    return (sine_bank,)


@app.cell(hide_code=True)
def hm_md():
    mo.md("""
    We can visualize all of the sine waves simultaneously using a heatmap representation. Each row is a different sine wave, and columns reflect time. The intensity of the value is like if the sine wave was coming towards and away rather than up and down. Notice how it looks like that the second half of the sine waves appear to be a mirror image of the first half. This is because the first half contain the positive frequencies, while the second half contains the negative frequencies. Negative frequencies capture sine waves that travel in reverse order around the complex plane compared to that travel forward. This becomes more relevant with the hilbert transform, but for the purposes of this tutorial we will be ignoring the negative frequencies.
    """)
    return


@app.cell
def fb_heatmap(sine_bank):
    _fig, _ax = plt.subplots(figsize=(8, 8))
    _ax.imshow(np.real(sine_bank))
    _ax.set_ylabel("Frequency", fontsize=18)
    _ax.set_xlabel("Time", fontsize=18)
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
def dft_compute(
    combined_signal,
    multi_amps,
    multi_freqs,
    sf_slider,
    sine_bank,
):
    fourier_coeffs = 2 * np.dot(combined_signal, sine_bank) / len(combined_signal)
    dft_freq_axis = fftfreq(len(combined_signal), 1 / sf_slider.value)
    _n_pos = len(combined_signal) // 2
    _pos_freqs = dft_freq_axis[:_n_pos]
    _pos_amps = np.abs(fourier_coeffs[:_n_pos])
    _fig, _axes = plt.subplots(nrows=2, figsize=(12, 8))
    _axes[0].plot(_pos_amps, linewidth=2)
    _axes[0].set_xlabel("Frequency (index)", fontsize=18)
    _axes[0].set_ylabel("Amplitude", fontsize=18)
    _axes[0].set_title("Power spectrum (DFT)", fontsize=18)
    _zoom = min(80, _n_pos)
    _axes[1].plot(_pos_freqs[:_zoom], _pos_amps[:_zoom], linewidth=2)
    _axes[1].set_xlabel("Frequency (Hz)", fontsize=18)
    _axes[1].set_ylabel("Amplitude", fontsize=18)
    _axes[1].set_title("Power spectrum (zoomed)", fontsize=18)
    for _f in multi_freqs:
        _axes[1].axvline(x=_f, color="red", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.close()
    mo.vstack([mo.md(f"Recall: `freq = {multi_freqs}`, `amplitude = {multi_amps}`"), _fig])
    return dft_freq_axis, fourier_coeffs


@app.cell(hide_code=True)
def fft_details():
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
def ifft_plot(fourier_coeffs, sine_bank):
    _fig, _ax = plt.subplots(figsize=(12, 5))
    _ax.plot(np.dot(fourier_coeffs, sine_bank) / 2)
    _ax.set_ylabel("Intensity", fontsize=18)
    _ax.set_xlabel("Time", fontsize=18)
    _ax.set_title("Reconstructed Time Series Signal", fontsize=18)
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def phase_md():
    mo.md("""
    ### Phase Spectrum

    The FFT provides frequency, amplitude, AND phase. Below we show both spectra. Phase carries timing/structural info.
    """)
    return


@app.cell
def phase_spectra(combined_signal, multi_freqs, sf_slider):
    _fft = fft(combined_signal)
    _freqs = fftfreq(len(combined_signal), 1 / sf_slider.value)
    _n = len(combined_signal) // 2
    _fig, _axes = plt.subplots(nrows=2, figsize=(14, 8))
    _axes[0].plot(_freqs[:_n], 2 * np.abs(_fft[:_n]) / len(combined_signal), linewidth=2)
    _axes[0].set_ylabel("Amplitude", fontsize=14)
    _axes[0].set_title("Power Spectrum", fontsize=16)
    _axes[0].set_xlim(0, 50)
    _axes[1].plot(_freqs[:_n], np.angle(_fft[:_n]), linewidth=2, color="orange")
    _axes[1].set_ylabel("Phase (radians)", fontsize=14)
    _axes[1].set_xlabel("Frequency (Hz)", fontsize=14)
    _axes[1].set_title("Phase Spectrum", fontsize=16)
    _axes[1].set_xlim(0, 50)
    for _f in multi_freqs:
        _axes[0].axvline(x=_f, color="red", alpha=0.3, linestyle="--")
        _axes[1].axvline(x=_f, color="red", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def scramble_md():
    mo.md("""
    #### Phase Scrambling
    Keep the power spectrum but randomize phases. The signal looks completely different — phase carries the structure!
    """)
    return


@app.cell
def phase_scramble(combined_signal):
    _fft = fft(combined_signal)
    _scrambled = np.abs(_fft) * exp(1j * np.random.uniform(-pi, pi, len(_fft)))
    _recon = ifft(_scrambled).real
    _n = len(combined_signal) // 2
    _fig, _axes = plt.subplots(nrows=3, figsize=(14, 10))
    _axes[0].plot(combined_signal, linewidth=2)
    _axes[0].set_title("Original Signal", fontsize=16)
    _axes[1].plot(_recon, linewidth=2, color="orange")
    _axes[1].set_title("Phase-Scrambled (same power spectrum!)", fontsize=16)
    _axes[2].plot(np.abs(_fft[:_n]), linewidth=2, alpha=0.7, label="Original")
    _axes[2].plot(np.abs(_scrambled[:_n]), linewidth=2, alpha=0.7, linestyle="--", label="Scrambled")
    _axes[2].set_title("Power Spectra (identical!)", fontsize=16)
    _axes[2].set_xlabel("Frequency", fontsize=14)
    _axes[2].legend(fontsize=12)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def fft_intro():
    mo.md("""
    ### Fast Fourier Transform
    In practice, `np.fft.fft` is used. Scale by dividing by N.
    """)
    return


@app.cell
def fft_section(combined_signal, dft_freq_axis):
    _result = fft(combined_signal)
    _n_pos = len(combined_signal) // 2
    _pos_freqs = dft_freq_axis[:_n_pos]
    _zoom = min(80, _n_pos)
    _fig, _axes = plt.subplots(nrows=2, figsize=(12, 8))
    _axes[0].plot(_pos_freqs[:_zoom], 2 * np.abs(_result[:_zoom]) / len(combined_signal), linewidth=2)
    _axes[0].set_ylabel("Amplitude", fontsize=18)
    _axes[0].set_xlabel("Frequency (Hz)", fontsize=18)
    _axes[0].set_title("FFT Power Spectrum", fontsize=18)
    _axes[1].plot(ifft(_result).real, linewidth=2)
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_xlabel("Time", fontsize=18)
    _axes[1].set_title("Reconstructed (Inverse FFT)", fontsize=18)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def ct_md():
    mo.md(r"""
    ### Convolution Theorem

    Convolution in the time domain is the same as multiplication in the frequency domain. This means that time domain convolution computations can be performed much more efficiently in the frequency domain via simple multiplication. (The opposite is also true that multiplication in the time domain is the same as convolution in the frequency domain. Watch this [Video](https://youtu.be/hj7j4Q8T3Ck) for an overview of the convolution theorem and convolution in the frequency domain.

    ![ConvolutionTheorem.png](../images/signal_processing/ConvolutionTheorem.png)

    Let's prove it:
    """)
    return


@app.cell
def conv_theorem(combined_signal, hrf_kernel):
    _conv_time = np.convolve(combined_signal, hrf_kernel)
    _n = len(combined_signal) + len(hrf_kernel) - 1
    _conv_freq = ifft(fft(combined_signal, n=_n) * fft(hrf_kernel, n=_n)).real
    _diff = np.max(np.abs(_conv_time - _conv_freq))
    _fig, _axes = plt.subplots(nrows=2, figsize=(14, 8))
    _axes[0].plot(_conv_time, linewidth=2, label="Time domain (np.convolve)")
    _axes[0].plot(_conv_freq, linewidth=2, linestyle="--", label="Freq domain (FFT multiply)")
    _axes[0].set_title("Both methods produce the same result", fontsize=16)
    _axes[0].legend(fontsize=12)
    _axes[1].plot(np.abs(_conv_time - _conv_freq), linewidth=2, color="red")
    _axes[1].set_title(f"Difference (max = {_diff:.2e})", fontsize=16)
    _axes[1].set_xlabel("Time", fontsize=14)
    plt.tight_layout()
    plt.close()
    mo.vstack([_fig, mo.callout(mo.md(f"**Convolution theorem confirmed!** Max difference: {_diff:.2e}"), kind="success")])
    return


@app.cell(hide_code=True)
def filt_intro():
    mo.md(r"""
    ## Filters

    Filters can be classified as finite impulse response (FIR) or infinite impulse response (IIR). These terms describe how a filter responds to a single input impulse. FIR filters have a response that ends at a discrete point in time, while IIR filters have a response that continues indefinitely.

    Filters are constructed in the frequency domain and have several properties that need to be considered.

    - ripple in the pass-band
    - attenuation in the stop-band
    - steepness of roll-off
    - filter order (i.e., length for FIR filters)
    - time-domain ringing

    In general, there is a frequency by time tradeoff. The sharper something is in frequency, the broader it is in time, and vice versa.

    Here we will use IIR butterworth filters as an example.


    ### High Pass
    High pass filters only allow high frequency signals to remain, effectively removing any low frequency information.

    Here we will construct a high pass butterworth filter and plot it in frequency space.
    """)
    return


@app.cell
def highpass(combined_signal, multi_freqs):
    _b, _a = butter(3, 25, btype="high", output="ba", fs=500)
    _w, _h = freqz(_b, _a, worN=1024, whole=False)
    _filtered = filtfilt(_b, _a, combined_signal)
    _fig, _axes = plt.subplots(nrows=2, figsize=(20, 8))
    _axes[0].plot(rad_to_hz(_w, 500), abs(_h), linewidth=3)
    for _f in multi_freqs:
        _axes[0].axvline(_f, color="red", linestyle="--", alpha=0.5)
    _axes[0].set_ylabel("Gain", fontsize=18)
    _axes[0].set_xlabel("Frequency (Hz)", fontsize=18)
    _axes[0].set_title("High Pass (order=3, cutoff=25 Hz) — red lines: signal components", fontsize=18)
    _axes[1].plot(combined_signal, linewidth=2, label="Original")
    _axes[1].plot(_filtered, linewidth=2, label="Filtered")
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_xlabel("Time", fontsize=18)
    _axes[1].legend(fontsize=18)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def temporal_md():
    mo.md("""
    Notice how the gain scales from [0,1]? Filters can be multiplied by the FFT of a signal to apply the filter in the frequency domain. When the resulting signal is transformed back in the time domain using the inverse FFT, the new signal will be filtered. This can be much faster than applying filters in the time domain.

    The filter_order parameter adjusts the sharpness of the cutoff in the frequency domain. Try playing with different values to see how it changes the filter plot.

    What does the filter look like in the temporal domain? Let’s take the inverse FFT and plot it to see what it looks like as a kernel in the temporal domain. Notice how changing the filter order adds more ripples in the time domain.
    """)
    return


@app.cell
def filt_temporal():
    _sos = butter(8, 25, btype="high", output="sos", fs=500)
    _w, _h = sosfreqz(_sos)
    _fig, _ax = plt.subplots(figsize=(12, 4))
    _ax.plot(ifft(_h).real[:100], linewidth=3)
    _ax.set_ylabel("Amplitude", fontsize=18)
    _ax.set_xlabel("Time", fontsize=18)
    _ax.set_title("High pass filter kernel (order=8)", fontsize=18)
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def lp_md():
    mo.md("""
    ### Low Pass
    Low pass filters only retain low frequency signals, which removes any high frequency information. We use `filtfilt` for zero-phase distortion.
    """)
    return


@app.cell
def lowpass(combined_signal, multi_freqs):
    _b, _a = butter(2, 7, btype="low", output="ba", fs=500)
    _w, _h = freqz(_b, _a, worN=1024, whole=False)
    _filtered = filtfilt(_b, _a, combined_signal)
    _fig, _axes = plt.subplots(nrows=2, figsize=(20, 8))
    _axes[0].plot(rad_to_hz(_w, 500), abs(_h), linewidth=3)
    for _f in multi_freqs:
        _axes[0].axvline(_f, color="red", linestyle="--", alpha=0.5)
    _axes[0].set_ylabel("Gain", fontsize=18)
    _axes[0].set_xlabel("Frequency (Hz)", fontsize=18)
    _axes[0].set_title("Low Pass (order=2, cutoff=7 Hz) — red lines: signal components", fontsize=18)
    _axes[1].plot(combined_signal, linewidth=2, label="Original")
    _axes[1].plot(_filtered, linewidth=2, label="Filtered")
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_xlabel("Time", fontsize=18)
    _axes[1].legend(fontsize=18)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def bp_md():
    mo.md("""
    ### Bandpass
    Bandpass filters permit retaining only a specific frequency. Morlet wavelets are an example of a bandpass filter. or example a Morlet wavelet is a gaussian with the peak frequency at the center of a bandpass filter.

    Let’s try selecting removing specific frequencies
    """)
    return


@app.cell
def bandpass(combined_signal, multi_freqs):
    _b, _a = butter(2, [12, 18], btype="bandpass", output="ba", fs=500)
    _w, _h = freqz(_b, _a, worN=1024, whole=False)
    _filtered = filtfilt(_b, _a, combined_signal)
    _fig, _axes = plt.subplots(nrows=2, figsize=(20, 8))
    _axes[0].plot(rad_to_hz(_w, 500), abs(_h), linewidth=3)
    for _f in multi_freqs:
        _axes[0].axvline(_f, color="red", linestyle="--", alpha=0.5)
    _axes[0].set_ylabel("Gain", fontsize=18)
    _axes[0].set_xlabel("Frequency (Hz)", fontsize=18)
    _axes[0].set_title("Bandpass (order=2, 12-18 Hz) — red lines: signal components", fontsize=18)
    _axes[1].plot(combined_signal, linewidth=2, label="Original")
    _axes[1].plot(_filtered, linewidth=2, label="Filtered")
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_xlabel("Time", fontsize=18)
    _axes[1].legend(fontsize=18)
    plt.tight_layout()
    plt.close()
    _fig
    return


@app.cell(hide_code=True)
def bs_md():
    mo.md("""
    ### Band-Stop
    Bandstop filters remove a specific frequency from the signal
    """)
    return


@app.cell
def bandstop(combined_signal, multi_freqs):
    _b, _a = butter(2, [4, 6], btype="bandstop", output="ba", fs=500)
    _w, _h = freqz(_b, _a, worN=1024, whole=False)
    _fig, _axes = plt.subplots(nrows=2, figsize=(20, 8))
    _axes[0].plot(rad_to_hz(_w, 500), abs(_h), linewidth=3)
    for _f in multi_freqs:
        _axes[0].axvline(_f, color="red", linestyle="--", alpha=0.5)
    _axes[0].set_ylabel("Gain", fontsize=18)
    _axes[0].set_xlabel("Frequency (Hz)", fontsize=18)
    _axes[0].set_title("Band-Stop (order=2, 4-6 Hz) — red lines: signal components", fontsize=18)
    _axes[1].plot(combined_signal, linewidth=2, label="Original")
    _axes[1].plot(filtfilt(_b, _a, combined_signal), linewidth=2, label="Filtered")
    _axes[1].set_ylabel("Intensity", fontsize=18)
    _axes[1].set_xlabel("Time", fontsize=18)
    _axes[1].legend(fontsize=18)
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
):
    _fs = 500
    if filter_type.value in ("bandpass", "bandstop"):
        _lo, _hi = min(cutoff_slider.value, cutoff2_slider.value), max(cutoff_slider.value, cutoff2_slider.value)
        if _lo == _hi:
            _hi = _lo + 1
        _b, _a = butter(order_slider.value, [_lo, _hi], btype=filter_type.value, output="ba", fs=_fs)
    else:
        _b, _a = butter(order_slider.value, cutoff_slider.value, btype=filter_type.value, output="ba", fs=_fs)
    _w, _h = freqz(_b, _a, worN=1024, whole=False)
    _fig, _axes = plt.subplots(nrows=3, figsize=(14, 12))
    _axes[0].plot(rad_to_hz(_w, _fs), abs(_h), linewidth=2)
    for _f in multi_freqs:
        _axes[0].axvline(_f, color="red", linestyle="--", alpha=0.5)
    _axes[0].set_ylabel("Gain", fontsize=14)
    _axes[0].set_xlabel("Frequency (Hz)", fontsize=14)
    _axes[0].set_title(f"{filter_type.value.title()} Filter (order={order_slider.value}) — red lines: signal components", fontsize=16)
    _axes[0].set_ylim(-0.05, 1.1)
    _axes[1].plot(ifft(_h).real[:100], linewidth=2)
    _axes[1].set_ylabel("Amplitude", fontsize=14)
    _axes[1].set_xlabel("Time", fontsize=14)
    _axes[1].set_title("Filter kernel (time domain)", fontsize=14)
    _axes[2].plot(combined_signal, linewidth=1.5, alpha=0.5, label="Original")
    _axes[2].plot(filtfilt(_b, _a, combined_signal), linewidth=2, label="Filtered")
    _axes[2].set_ylabel("Intensity", fontsize=14)
    _axes[2].set_xlabel("Time", fontsize=14)
    _axes[2].set_title("Filtered Signal", fontsize=14)
    _axes[2].legend(fontsize=12)
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


if __name__ == "__main__":
    app.run()
