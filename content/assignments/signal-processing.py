# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "scipy", "matplotlib", "marimo-grader-client", "mograder"]
# mograder-cell-hashes = "75142515,fe9f4ef6,8af6cf8a,625df5dc,67a54988,ca2329d2,e348968a,9f2cc442,406a3c2b,c153bf14,c9ec193b,3aa0c3f6,a8fb810c,edd5928b,f95a95b8,52b7212a,d16afb54,8feeb704,5314c3dc,d283430f,286fc629,9a2b1022,294a7272,bd3d1a89,3023e359,4838bc52"
# grader-server = "https://grader.dartbrains.org"
# grader-course = "neuroimaging"
# grader-term = "2026-fall"
# grader-offering-id = "a8e72d80-495a-4f26-a006-b9798bf9b306"
# grader-assignment = "signal-processing"
# grader-assignment-id = "51e589a9-69f3-4ae5-911d-5cdb60354f96"
# grader-assignment-version = "8143aecf-1f9b-4a52-878a-628318add7ef"
# grader-version = "6"
# ///
"""DartBrains assignment: Signal Processing."""

import marimo

__generated_with = "0.24.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.fft import fft, ifft, fftfreq
    from scipy.signal import butter, filtfilt, freqz

    from marimo_grader_client import Grader

    g = Grader()  # reads server / offering / assignment / version from this file's PEP 723 block
    return butter, fft, fftfreq, filtfilt, freqz, g, ifft, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Assignment: Signal Processing

        This assignment works through the **Exercises** at the end of the
        [Signal Processing chapter](https://dartbrains.org/content/Signal_Processing.html) on
        dartbrains.org. You will build a signal out of sine waves, recover its components with the
        FFT, and then remove one component two different ways: with a Butterworth bandstop filter
        in the time domain and by editing the spectrum directly in the frequency domain.

        Each question has a cell for your code, a **Check** cell that runs locally and gives
        hints, and a **Submit** button that records an attempt on the server. Sign in once at
        the top. Checks award partial credit; the final interpretation question is graded by hand.
        """
    )
    return


@app.cell
def _(g):
    g.signin_button()
    return


@app.cell
def _():
    # === MOGRADER: MARKS ===
    _marks = {"sp-q01": 5, "sp-q02": 5, "sp-q03": 5, "sp-q04": 5, "sp-q05": 5}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup

        As in the chapter, we sample at `sf = 500` Hz. `time` holds two seconds of sample times
        (1000 samples), so the FFT's frequency resolution is 0.5 Hz and any integer frequency
        lands exactly on a bin. The Nyquist frequency is `sf / 2 = 250` Hz.
        """
    )
    return


@app.cell
def _(np):
    sf = 500  # sampling frequency (Hz)
    time = np.arange(-1, 1, 1 / sf)  # two seconds of samples
    n_samples = len(time)
    nyquist = sf / 2
    return n_samples, nyquist, sf, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q1. Simulate a time series with 7 frequencies plus noise

        Build a signal from **seven** sine waves, following the chapter's `multi_sine` example.
        Choose seven *different integer* frequencies between 1 and 100 Hz (well below Nyquist) and
        keep neighbouring frequencies at least 5 Hz apart: you will filter one of them out later.
        Give each wave an amplitude of at least 3 and its own phase. Store them as `freqs`, `amps`
        and `phases`; store the noise-free sum sampled on `time` as `clean_signal`; then add
        Gaussian noise with standard deviation 5
        (`5 * np.random.default_rng(0).standard_normal(n_samples)`) to get `signal`.
        Plot the individual waves and the noisy sum.
        """
    )
    return


@app.cell
def _(n_samples, np, plt, time):
    amps = ...
    clean_signal = ...
    freqs = ...
    phases = ...
    signal = ...
    # YOUR CODE HERE
    pass
    return amps, clean_signal, freqs, phases, signal


@app.cell
def _(amps, clean_signal, fft, fftfreq, freqs, g, mo, n_samples, np, nyquist, sf, signal, time):
    mo.stop(
        any(v is ... for v in (freqs, amps, clean_signal, signal)),
        mo.md("**Complete the code cell above first.** This check runs once `freqs`, `amps`, `clean_signal` and `signal` are defined."),
    )

    # HIDDEN TESTS
    g.check(
        "sp-q01: simulated multi-frequency signal",
        [
            (
                len(freqs) == 7 and len(set(freqs)) == 7,
                "freqs should hold 7 distinct frequencies",
                1,
            ),
            (
                all(0 < _f < nyquist for _f in freqs),
                "every frequency must be below the Nyquist frequency (sf / 2)",
                1,
            ),
            (
                isinstance(clean_signal, np.ndarray)
                and clean_signal.shape == time.shape
                and np.shape(signal) == time.shape,
                "clean_signal and signal should be sampled on `time` (one value per sample)",
                1,
            ),
            (
                np.std(signal - clean_signal) > 1,
                "signal should be clean_signal plus Gaussian noise",
                1,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("sp-q01")
    return


@app.cell
def _(g):
    g.feedback("sp-q01")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q2. Identify each frequency with an FFT

        Show that you can recover every component of `signal` with a fast Fourier transform.
        Compute `spectrum = fft(signal)`, convert it to a single-sided amplitude spectrum
        `amplitude = 2 * np.abs(spectrum) / n_samples`, and build the matching frequency axis
        `freq_axis = fftfreq(n_samples, 1 / sf)`. Then find the seven *positive* frequencies with
        the largest amplitude and store them, sorted, in `detected_freqs`. Plot the amplitude
        spectrum for positive frequencies (zoom in to 0-100 Hz) with a marker at each detected
        frequency.
        """
    )
    return


@app.cell
def _(fft, fftfreq, freqs, n_samples, np, plt, sf, signal):
    amplitude = ...
    detected_freqs = ...
    freq_axis = ...
    spectrum = ...
    # YOUR CODE HERE
    pass
    return amplitude, detected_freqs, freq_axis, spectrum


@app.cell
def _(amplitude, amps, detected_freqs, freq_axis, freqs, g, mo, np, signal):
    mo.stop(
        any(v is ... for v in (freq_axis, amplitude, detected_freqs, freqs, amps, signal)),
        mo.md(
            "**Complete the code cell above first.** This check runs once `freq_axis`, `amplitude` and `detected_freqs` "
            "are defined (it also needs `freqs`, `amps` and `signal` from the earlier questions)."
        ),
    )

    g.check(
        "sp-q02: recover frequencies with the FFT",
        [
            (
                np.shape(freq_axis) == np.shape(signal),
                "freq_axis should have one frequency per FFT bin (same length as signal)",
                1,
            ),
            (
                np.shape(amplitude) == np.shape(signal) and bool(np.all(amplitude >= 0)),
                "amplitude should be a non-negative spectrum with one value per FFT bin",
                1,
            ),
            (
                np.array_equal(np.sort(np.round(detected_freqs)), np.sort(freqs)),
                "detected_freqs should match the seven frequencies you simulated",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("sp-q02")
    return


@app.cell
def _(g):
    g.feedback("sp-q02")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q3. Remove one frequency with a bandstop filter

        Pick one of your frequencies to remove and store it in `target_freq`. Build a Butterworth
        bandstop filter with `butter(order, [target_freq - 3, target_freq + 3], btype="bandstop",
        fs=sf)` (an order of 2-4 works well), apply it to `signal` with zero-phase `filtfilt`, and
        store the result in `filtered`. Plot the filter's gain with `freqz` (mark your
        frequencies) above the original and filtered signals, as in the chapter's bandstop
        example.
        """
    )
    return


@app.cell
def _(butter, filtfilt, freqs, freqz, plt, sf, signal, time):
    filtered = ...
    target_freq = ...
    # YOUR CODE HERE
    pass
    return filtered, target_freq


@app.cell
def _(fft, fftfreq, filtered, freqs, g, mo, n_samples, np, sf, signal, target_freq):
    mo.stop(
        any(v is ... for v in (filtered, target_freq, freqs, signal)),
        mo.md(
            "**Complete the code cell above first.** This check runs once `filtered` and `target_freq` "
            "are defined (it also needs `freqs` and `signal` from the earlier questions)."
        ),
    )

    _freq_axis = fftfreq(n_samples, 1 / sf)
    _amp_orig = 2 * np.abs(fft(signal)) / n_samples
    _amp_filt = 2 * np.abs(fft(filtered)) / n_samples

    def _bin(f):
        return int(np.argmin(np.abs(_freq_axis - f)))

    g.check(
        "sp-q03: time-domain bandstop filter",
        [
            (
                np.shape(filtered) == np.shape(signal),
                "filtered should be the same length as signal",
                1,
            ),
            (
                target_freq in list(freqs),
                "target_freq should be one of your simulated frequencies",
                1,
            ),
            (
                _amp_filt[_bin(target_freq)] < 0.2 * _amp_orig[_bin(target_freq)],
                "the target frequency should be attenuated by at least 80%",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("sp-q03")
    return


@app.cell
def _(g):
    g.feedback("sp-q03")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q4. Remove the same frequency in the frequency domain

        Now remove `target_freq` without a time-domain filter. Take the FFT of `signal`, make a
        copy called `spectrum_masked`, and set to zero every bin whose frequency is within 1 Hz of
        `target_freq`. Remember the spectrum is two-sided, so the bins at `-target_freq` must be
        zeroed too (`np.abs(np.abs(freq_axis) - target_freq) <= 1` selects both). Transform back
        with `ifft` and keep the real part as `reconstructed`. Plot `signal`, `reconstructed` and
        `filtered` from Q3 on the same axes, plus the difference `signal - reconstructed`
        (which should look like the sine wave you removed).
        """
    )
    return


@app.cell
def _(fft, filtered, freq_axis, ifft, np, plt, signal, target_freq, time):
    reconstructed = ...
    spectrum_masked = ...
    # YOUR CODE HERE
    pass
    return reconstructed, spectrum_masked


@app.cell
def _(fft, fftfreq, freqs, g, ifft, mo, n_samples, np, reconstructed, sf, signal, spectrum_masked, target_freq):
    mo.stop(
        any(v is ... for v in (reconstructed, spectrum_masked, freqs, signal, target_freq)),
        mo.md(
            "**Complete the code cell above first.** This check runs once `reconstructed` and `spectrum_masked` "
            "are defined (it also needs `freqs`, `signal` and `target_freq` from the earlier questions)."
        ),
    )

    _freq_axis = fftfreq(n_samples, 1 / sf)
    _amp_orig = 2 * np.abs(fft(signal)) / n_samples
    _amp_recon = 2 * np.abs(fft(reconstructed)) / n_samples

    def _bin(f):
        return int(np.argmin(np.abs(_freq_axis - f)))

    # HIDDEN TESTS
    g.check(
        "sp-q04: frequency-domain removal and reconstruction",
        [
            (
                np.shape(reconstructed) == np.shape(signal) and np.isrealobj(reconstructed),
                "reconstructed should be a real-valued array the same length as signal",
                1,
            ),
            (
                np.shape(spectrum_masked) == np.shape(signal) and np.iscomplexobj(spectrum_masked),
                "spectrum_masked should be the (complex) FFT of signal with the target bins zeroed",
                1,
            ),
            (
                _amp_recon[_bin(target_freq)] < 0.05 * _amp_orig[_bin(target_freq)],
                "the target frequency should be (almost) completely removed",
                2,
            ),
            # HIDDEN TESTS
        ],
    )
    return


@app.cell
def _(g):
    g.submit_button("sp-q04")
    return


@app.cell
def _(g):
    g.feedback("sp-q04")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Q5. Interpretation

        Q3 and Q4 removed the same frequency in two different ways. In two or three sentences,
        compare the two results (look at the frequencies next to `target_freq` and at the edges
        of the signal), and use the convolution theorem to explain why a Butterworth bandstop
        applied with `filtfilt` and zeroing FFT bins are related but not identical operations.
        """
    )
    return


@app.cell
def _(mo):
    answer = mo.ui.text_area(placeholder="Your answer...", full_width=True)
    answer
    return (answer,)


@app.cell
def _(answer, g):
    # UI element values are not part of the notebook file, so pass them as outputs;
    # they are stored with the attempt and shown to the grader next to the notebook.
    g.submit_button("sp-q05", outputs={"answer": answer.value})
    return


@app.cell
def _(g):
    g.feedback("sp-q05")
    return


if __name__ == "__main__":
    app.run()
