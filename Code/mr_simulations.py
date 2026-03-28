"""
MR Physics Simulation Module for Dartbrains Course
===================================================

Pure numpy implementations of Bloch equation simulations, signal generators,
and Plotly visualization helpers for interactive MR physics notebooks.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# Tissue Constants
# ============================================================================

# T1 (ms), T2 (ms), PD (relative) at 1.5T and 3T
# Sources: Wansapura et al. 1999, Stanisz et al. 2005, de Bazelaire et al. 2004
TISSUE_PROPERTIES = {
    "1.5T": {
        "Gray Matter":  {"T1": 900,  "T2": 100, "PD": 0.82},
        "White Matter": {"T1": 600,  "T2": 80,  "PD": 0.71},
        "CSF":          {"T1": 4000, "T2": 2000,"PD": 1.00},
        "Fat":          {"T1": 250,  "T2": 60,  "PD": 1.00},
        "Muscle":       {"T1": 900,  "T2": 50,  "PD": 1.00},
    },
    "3T": {
        "Gray Matter":  {"T1": 1300, "T2": 80,  "PD": 0.82},
        "White Matter": {"T1": 830,  "T2": 60,  "PD": 0.71},
        "CSF":          {"T1": 4000, "T2": 2000,"PD": 1.00},
        "Fat":          {"T1": 370,  "T2": 50,  "PD": 1.00},
        "Muscle":       {"T1": 1400, "T2": 40,  "PD": 1.00},
    },
}

# Gyromagnetic ratios (MHz/T)
GAMMA = {
    "1H":   42.576,   # Hydrogen (proton)
    "13C":  10.708,   # Carbon-13
    "23Na": 11.262,   # Sodium-23
    "31P":  17.235,   # Phosphorus-31
}

GAMMA_H = GAMMA["1H"]  # Most commonly used

# ============================================================================
# Bloch Equation Solver
# ============================================================================


def rotation_x(angle):
    """Rotation matrix about the x-axis by `angle` (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


def rotation_y(angle):
    """Rotation matrix about the y-axis by `angle` (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def rotation_z(angle):
    """Rotation matrix about the z-axis by `angle` (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def apply_relaxation(M, dt, T1, T2, M0=1.0):
    """Apply T1 recovery and T2 decay over time step dt (ms)."""
    E1 = np.exp(-dt / T1) if T1 > 0 else 0.0
    E2 = np.exp(-dt / T2) if T2 > 0 else 0.0
    Mx = M[0] * E2
    My = M[1] * E2
    Mz = M[2] * E1 + M0 * (1 - E1)
    return np.array([Mx, My, Mz])


def apply_rf_pulse(M, flip_angle_deg, phase_deg=0):
    """Apply an instantaneous RF pulse with given flip angle and phase.

    flip_angle_deg: flip angle in degrees (rotation about axis in xy-plane)
    phase_deg: phase of the B1 field (0 = x-axis, 90 = y-axis)
    """
    flip = np.radians(flip_angle_deg)
    phase = np.radians(phase_deg)
    # Rotate into the frame of the RF pulse, apply rotation, rotate back
    Rz = rotation_z(-phase)
    Rz_inv = rotation_z(phase)
    Rx = rotation_x(flip)
    R = Rz_inv @ Rx @ Rz
    return R @ M


def simulate_bloch(M0_vec, T1, T2, B0, dt, n_steps, rf_events=None, M0_eq=1.0):
    """Simulate magnetization evolution using the Bloch equations.

    Parameters
    ----------
    M0_vec : array-like, shape (3,)
        Initial magnetization vector [Mx, My, Mz].
    T1, T2 : float
        Relaxation times in ms.
    B0 : float
        Static field strength in Tesla (determines precession frequency).
    dt : float
        Time step in ms.
    n_steps : int
        Number of time steps.
    rf_events : list of dict, optional
        Each dict has keys: 'time_step' (int), 'flip_angle' (degrees), 'phase' (degrees).
    M0_eq : float
        Equilibrium magnetization magnitude.

    Returns
    -------
    t : ndarray, shape (n_steps,)
        Time points in ms.
    M : ndarray, shape (n_steps, 3)
        Magnetization trajectory [Mx, My, Mz] at each time point.
    """
    M = np.zeros((n_steps, 3))
    M[0] = np.array(M0_vec, dtype=float)
    t = np.arange(n_steps) * dt

    # Precession angle per time step
    omega = 2 * np.pi * GAMMA_H * B0  # rad/ms (since gamma is MHz/T = rad/(us*T))
    # Actually gamma in MHz/T means f = gamma * B0 in MHz
    # omega = 2*pi*f = 2*pi*gamma*B0 in Mrad/s
    # Per ms: dphi = 2*pi*gamma*B0 * dt * 1e-3... let's be careful
    # gamma = 42.576 MHz/T, so f = 42.576 * B0 MHz
    # omega = 2*pi*42.576*B0 * 1e6 rad/s
    # per dt ms: dphi = omega * dt * 1e-3 = 2*pi*42.576*B0 * 1e6 * dt * 1e-3
    # = 2*pi*42.576*B0*dt * 1e3 rad
    # That's way too fast to visualize. For visualization, we'll use a
    # slow "effective" precession rate so students can see the rotation.
    # The actual Larmor frequency at 3T is ~128 MHz -- impossible to animate.

    # For visualization we use a normalized precession: one full rotation
    # per "period" that scales with B0 but is visually tractable.
    # f_vis = B0 * scale_factor (e.g., 1 rotation per 10 ms at 1T)
    f_vis = B0 * 0.1  # rotations per ms (adjustable)
    dphi = 2 * np.pi * f_vis * dt

    rf_dict = {}
    if rf_events:
        for ev in rf_events:
            rf_dict[ev["time_step"]] = ev

    for i in range(1, n_steps):
        # Check for RF pulse at this time step
        if i in rf_dict:
            ev = rf_dict[i]
            M[i - 1] = apply_rf_pulse(M[i - 1], ev["flip_angle"], ev.get("phase", 0))

        # Precession (rotation about z)
        Rz = rotation_z(dphi)
        M_rot = Rz @ M[i - 1]

        # Relaxation
        M[i] = apply_relaxation(M_rot, dt, T1, T2, M0_eq)

    return t, M


# ============================================================================
# Signal Generators
# ============================================================================


def fid_signal(t, T2_star, f0=0.0, M0=1.0, flip_angle_deg=90):
    """Generate a Free Induction Decay signal.

    Parameters
    ----------
    t : ndarray
        Time points in ms.
    T2_star : float
        Effective T2* decay time in ms.
    f0 : float
        Off-resonance frequency offset in kHz (for visualization).
    M0 : float
        Initial magnetization.
    flip_angle_deg : float
        Flip angle in degrees.

    Returns
    -------
    signal : ndarray (complex)
        Complex FID signal.
    """
    amplitude = M0 * np.sin(np.radians(flip_angle_deg))
    decay = np.exp(-t / T2_star)
    oscillation = np.exp(1j * 2 * np.pi * f0 * t)
    return amplitude * decay * oscillation


def spin_echo_signal(TE, TR, T1, T2, PD=1.0):
    """Calculate spin echo signal intensity for given parameters.

    Uses the spin echo signal equation:
    S = PD * (1 - exp(-TR/T1)) * exp(-TE/T2)

    Parameters
    ----------
    TE, TR : float or ndarray
        Echo time and repetition time in ms.
    T1, T2 : float
        Relaxation times in ms.
    PD : float
        Proton density (relative).

    Returns
    -------
    signal : float or ndarray
        Signal intensity.
    """
    return PD * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)


def gradient_echo_signal(TE, TR, T1, T2_star, flip_angle_deg, PD=1.0):
    """Calculate gradient echo (SPGR) signal intensity.

    S = PD * sin(a) * (1 - exp(-TR/T1)) / (1 - cos(a)*exp(-TR/T1)) * exp(-TE/T2*)

    Parameters
    ----------
    TE, TR : float
        Echo time and repetition time in ms.
    T1 : float
        T1 relaxation time in ms.
    T2_star : float
        T2* decay time in ms.
    flip_angle_deg : float
        Flip angle in degrees.
    PD : float
        Proton density.

    Returns
    -------
    signal : float
        Signal intensity.
    """
    a = np.radians(flip_angle_deg)
    E1 = np.exp(-TR / T1)
    return PD * np.sin(a) * (1 - E1) / (1 - np.cos(a) * E1) * np.exp(-TE / T2_star)


def t1_recovery(t, T1, M0=1.0):
    """T1 recovery curve: Mz(t) = M0 * (1 - exp(-t/T1))."""
    return M0 * (1 - np.exp(-t / T1))


def t2_decay(t, T2, M0=1.0):
    """T2 decay curve: Mxy(t) = M0 * exp(-t/T2)."""
    return M0 * np.exp(-t / T2)


def hrf(t, peak_time=6.0, undershoot_time=16.0, peak_amplitude=1.0,
        undershoot_ratio=0.167, peak_width=1.0, undershoot_width=1.0):
    """Double-gamma hemodynamic response function.

    Parameters
    ----------
    t : ndarray
        Time points in seconds.
    peak_time : float
        Time of peak response in seconds.
    undershoot_time : float
        Time of undershoot in seconds.
    peak_amplitude : float
        Amplitude of peak.
    undershoot_ratio : float
        Ratio of undershoot to peak amplitude.
    peak_width, undershoot_width : float
        Width parameters for peak and undershoot.

    Returns
    -------
    h : ndarray
        HRF values at each time point.
    """
    from scipy.stats import gamma as gamma_dist
    peak = gamma_dist.pdf(t, peak_time / peak_width, scale=peak_width)
    undershoot = gamma_dist.pdf(t, undershoot_time / undershoot_width, scale=undershoot_width)
    h = peak_amplitude * peak - undershoot_ratio * peak_amplitude * undershoot
    # Normalize so peak = peak_amplitude
    if np.max(np.abs(h)) > 0:
        h = h / np.max(h) * peak_amplitude
    return h


# ============================================================================
# Plotly Visualization Helpers
# ============================================================================


def plot_magnetization_3d(M, title="Magnetization Vector", show_trajectory=True,
                          show_endpoint=True, show_axes=True):
    """Create a 3D Plotly figure of the magnetization vector trajectory.

    Parameters
    ----------
    M : ndarray, shape (n, 3) or (3,)
        Magnetization trajectory or single vector.
    title : str
        Figure title.
    show_trajectory : bool
        Show the path of the magnetization vector.
    show_endpoint : bool
        Highlight the current (final) position.
    show_axes : bool
        Show x, y, z axis lines.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if M.ndim == 1:
        M = M.reshape(1, 3)

    fig = go.Figure()

    # Reference sphere (unit sphere wireframe)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.05, colorscale=[[0, "lightblue"], [1, "lightblue"]],
        showscale=False, name="Unit sphere"
    ))

    if show_axes:
        axis_len = 1.3
        for ax, color, label in [
            ([axis_len, 0, 0], "red", "x"),
            ([0, axis_len, 0], "green", "y"),
            ([0, 0, axis_len], "blue", "z (B₀)")
        ]:
            fig.add_trace(go.Scatter3d(
                x=[0, ax[0]], y=[0, ax[1]], z=[0, ax[2]],
                mode="lines+text",
                line=dict(color=color, width=2, dash="dash"),
                text=["", label],
                textposition="top center",
                showlegend=False
            ))

    if show_trajectory and len(M) > 1:
        fig.add_trace(go.Scatter3d(
            x=M[:, 0], y=M[:, 1], z=M[:, 2],
            mode="lines",
            line=dict(color="orange", width=3),
            name="Trajectory",
            opacity=0.6
        ))

    # Current magnetization vector as a cone/arrow
    end = M[-1]
    fig.add_trace(go.Cone(
        x=[0], y=[0], z=[0],
        u=[end[0]], v=[end[1]], w=[end[2]],
        sizemode="absolute", sizeref=0.15,
        colorscale=[[0, "red"], [1, "red"]],
        showscale=False, name="M"
    ))

    if show_endpoint:
        fig.add_trace(go.Scatter3d(
            x=[end[0]], y=[end[1]], z=[end[2]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name=f"M = [{end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f}]"
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Mx",
            yaxis_title="My",
            zaxis_title="Mz",
            aspectmode="cube",
            xaxis=dict(range=[-1.3, 1.3]),
            yaxis=dict(range=[-1.3, 1.3]),
            zaxis=dict(range=[-1.3, 1.3]),
        ),
        width=550, height=500,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def plot_signal_timeline(t, signals, labels=None, colors=None,
                         title="Signal", xlabel="Time (ms)", ylabel="Signal"):
    """Plot one or more signal time courses.

    Parameters
    ----------
    t : ndarray
        Time axis.
    signals : list of ndarray or single ndarray
        Signal(s) to plot.
    labels : list of str, optional
        Legend labels.
    colors : list of str, optional
        Line colors.
    title, xlabel, ylabel : str
        Axis labels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]
    if labels is None:
        labels = [f"Signal {i}" for i in range(len(signals))]
    if colors is None:
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    fig = go.Figure()
    for i, sig in enumerate(signals):
        fig.add_trace(go.Scatter(
            x=t, y=sig,
            mode="lines",
            name=labels[i],
            line=dict(color=colors[i % len(colors)], width=2.5),
        ))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=700, height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(x=0.7, y=0.95),
    )
    return fig


def plot_contrast_bars(tissues, signals, title="Tissue Signal Intensity"):
    """Bar chart of signal intensities for different tissues.

    Parameters
    ----------
    tissues : list of str
        Tissue names.
    signals : list of float
        Signal intensities.
    title : str
        Figure title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    colors = {
        "Gray Matter": "#808080",
        "White Matter": "#F5F5DC",
        "CSF": "#4169E1",
        "Fat": "#FFD700",
        "Muscle": "#CD5C5C",
    }
    bar_colors = [colors.get(t, "#636EFA") for t in tissues]

    fig = go.Figure(go.Bar(
        x=tissues, y=signals,
        marker_color=bar_colors,
        text=[f"{s:.2f}" for s in signals],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        yaxis_title="Signal Intensity (a.u.)",
        yaxis=dict(range=[0, 1.1]),
        width=500, height=400,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def plot_pulse_sequence(events, total_time, title="Pulse Sequence Diagram"):
    """Draw a basic pulse sequence timing diagram.

    Parameters
    ----------
    events : dict
        Keys are channel names ('RF', 'Gx', 'Gy', 'Gz', 'Signal', 'ADC').
        Values are lists of dicts with 'start', 'end', 'amplitude', and
        optionally 'label', 'style' ('rect' or 'sinc').
    total_time : float
        Total time extent of the diagram (ms).
    title : str
        Figure title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    channels = ["RF", "Gz", "Gy", "Gx", "Signal", "ADC"]
    channel_colors = {
        "RF": "#EF553B", "Gz": "#00CC96", "Gy": "#AB63FA",
        "Gx": "#636EFA", "Signal": "#FFA15A", "ADC": "#FF6692"
    }

    fig = make_subplots(
        rows=len(channels), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_titles=channels,
    )

    for i, ch in enumerate(channels):
        row = i + 1
        # Baseline
        fig.add_trace(go.Scatter(
            x=[0, total_time], y=[0, 0],
            mode="lines", line=dict(color="gray", width=1, dash="dot"),
            showlegend=False
        ), row=row, col=1)

        if ch in events:
            for ev in events[ch]:
                t_start = ev["start"]
                t_end = ev["end"]
                amp = ev["amplitude"]
                t_arr = np.linspace(t_start, t_end, 100)

                style = ev.get("style", "rect")
                if style == "sinc":
                    mid = (t_start + t_end) / 2
                    width = (t_end - t_start) / 2
                    x_norm = (t_arr - mid) / width * 3 * np.pi
                    y_arr = amp * np.sinc(x_norm / np.pi)
                elif style == "trapezoid":
                    ramp = (t_end - t_start) * 0.1
                    y_arr = np.piecewise(
                        t_arr,
                        [t_arr < t_start + ramp,
                         (t_arr >= t_start + ramp) & (t_arr <= t_end - ramp),
                         t_arr > t_end - ramp],
                        [lambda t: amp * (t - t_start) / ramp,
                         amp,
                         lambda t: amp * (t_end - t) / ramp]
                    )
                else:  # rect
                    y_arr = np.full_like(t_arr, amp)

                fig.add_trace(go.Scatter(
                    x=t_arr, y=y_arr,
                    mode="lines", fill="tozeroy",
                    line=dict(color=channel_colors.get(ch, "#636EFA"), width=2),
                    fillcolor=channel_colors.get(ch, "#636EFA"),
                    opacity=0.7,
                    showlegend=False
                ), row=row, col=1)

                if "label" in ev:
                    fig.add_annotation(
                        x=(t_start + t_end) / 2, y=amp * 1.1,
                        text=ev["label"], showarrow=False,
                        row=row, col=1, font=dict(size=10)
                    )

    fig.update_layout(
        title=title,
        height=120 * len(channels),
        width=700,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=30),
    )
    fig.update_xaxes(title_text="Time (ms)", row=len(channels), col=1)
    return fig


def plot_kspace_and_image(kspace, title="K-Space and Image"):
    """Side-by-side display of k-space magnitude and reconstructed image.

    Parameters
    ----------
    kspace : ndarray, shape (N, M), complex
        K-space data (2D Fourier domain).
    title : str
        Figure title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    image = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))
    kspace_mag = np.log1p(np.abs(kspace))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["K-Space (log magnitude)", "Reconstructed Image"])

    fig.add_trace(go.Heatmap(
        z=kspace_mag, colorscale="Viridis", showscale=False
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=image, colorscale="Gray", showscale=False
    ), row=1, col=2)

    fig.update_layout(
        title=title,
        width=800, height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# ============================================================================
# Fourier Utilities
# ============================================================================


def compute_spectrum(signal, dt):
    """Compute magnitude spectrum of a 1D signal.

    Parameters
    ----------
    signal : ndarray
        Time-domain signal (real or complex).
    dt : float
        Time step in ms.

    Returns
    -------
    freqs : ndarray
        Frequency axis in kHz.
    magnitude : ndarray
        Magnitude spectrum.
    """
    n = len(signal)
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dt))  # in 1/ms = kHz
    return freqs, np.abs(spectrum) / n


def image_to_kspace(image):
    """Convert a 2D image to centered k-space."""
    return np.fft.fftshift(np.fft.fft2(image))


def kspace_to_image(kspace):
    """Convert centered k-space back to image."""
    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))


def mask_kspace(kspace, mask_type="center", radius_fraction=0.2):
    """Apply a mask to k-space for educational demonstrations.

    Parameters
    ----------
    kspace : ndarray, complex
        Centered k-space data.
    mask_type : str
        One of: 'center', 'periphery', 'horizontal_lines', 'random'
    radius_fraction : float
        Fraction of k-space radius to use for center/periphery masks.

    Returns
    -------
    masked_kspace : ndarray, complex
        Masked k-space data.
    mask : ndarray, bool
        The mask applied.
    """
    ny, nx = kspace.shape
    cy, cx = ny // 2, nx // 2
    Y, X = np.ogrid[:ny, :nx]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    r = radius_fraction * max_r

    if mask_type == "center":
        mask = dist <= r
    elif mask_type == "periphery":
        mask = dist > r
    elif mask_type == "horizontal_lines":
        # Keep every 4th line
        mask = np.zeros_like(kspace, dtype=bool)
        mask[::4, :] = True
    elif mask_type == "random":
        rng = np.random.default_rng(42)
        mask = rng.random(kspace.shape) > 0.7
        # Always keep center
        center_mask = dist <= r * 0.5
        mask = mask | center_mask
    else:
        mask = np.ones_like(kspace, dtype=bool)

    return kspace * mask, mask
