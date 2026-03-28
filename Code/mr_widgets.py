"""
MR Physics Interactive Widgets (anywidget)
==========================================

Custom animated widgets for MR physics teaching notebooks.
Uses Three.js (WebGL) for 3D and Canvas 2D for 2D animations.
"""

import anywidget
import traitlets
from pathlib import Path

_JS_DIR = Path(__file__).parent / "js"


class PrecessionWidget(anywidget.AnyWidget):
    """3D magnetization precession animation using Three.js."""
    _esm = _JS_DIR / "precession_widget.js"

    b0 = traitlets.Float(3.0).tag(sync=True)
    flip_angle = traitlets.Float(90.0).tag(sync=True)
    t1 = traitlets.Float(0.0).tag(sync=True)
    t2 = traitlets.Float(0.0).tag(sync=True)
    show_relaxation = traitlets.Bool(False).tag(sync=True)
    paused = traitlets.Bool(False).tag(sync=True)


class SpinEnsembleWidget(anywidget.AnyWidget):
    """2D spin ensemble dephasing/rephasing animation."""
    _esm = _JS_DIR / "spin_ensemble_widget.js"

    sequence_type = traitlets.Unicode("spin_echo").tag(sync=True)
    speed = traitlets.Float(1.0).tag(sync=True)
    paused = traitlets.Bool(False).tag(sync=True)


class CompassWidget(anywidget.AnyWidget):
    """Animated compass needle in a magnetic field."""
    _esm = _JS_DIR / "compass_widget.js"

    b0 = traitlets.Float(3.0).tag(sync=True)


class NetMagnetizationWidget(anywidget.AnyWidget):
    """3D proton ensemble showing net magnetization emergence."""
    _esm = _JS_DIR / "net_magnetization_widget.js"

    n_protons = traitlets.Int(100).tag(sync=True)
    b0_on = traitlets.Bool(False).tag(sync=True)


class KSpaceWidget(anywidget.AnyWidget):
    """Progressive k-space filling with real-time reconstruction."""
    _esm = _JS_DIR / "kspace_widget.js"

    mask_type = traitlets.Unicode("progressive").tag(sync=True)
    radius_fraction = traitlets.Float(0.2).tag(sync=True)
    speed = traitlets.Float(1.0).tag(sync=True)


class ConvolutionWidget(anywidget.AnyWidget):
    """Animated convolution of stimulus events with HRF."""
    _esm = _JS_DIR / "convolution_widget.js"

    pattern = traitlets.Unicode("single").tag(sync=True)
    speed = traitlets.Float(1.0).tag(sync=True)


class EncodingWidget(anywidget.AnyWidget):
    """Animated frequency and phase encoding demonstration."""
    _esm = _JS_DIR / "encoding_widget.js"

    speed = traitlets.Float(1.0).tag(sync=True)


class TransformCubeWidget(anywidget.AnyWidget):
    """3D rigid body / affine transformation on a cube."""
    _esm = _JS_DIR / "transform_cube_widget.js"

    trans_x = traitlets.Float(0.0).tag(sync=True)
    trans_y = traitlets.Float(0.0).tag(sync=True)
    trans_z = traitlets.Float(0.0).tag(sync=True)
    rot_x = traitlets.Float(0.0).tag(sync=True)
    rot_y = traitlets.Float(0.0).tag(sync=True)
    rot_z = traitlets.Float(0.0).tag(sync=True)
    scale_x = traitlets.Float(1.0).tag(sync=True)
    scale_y = traitlets.Float(1.0).tag(sync=True)
    scale_z = traitlets.Float(1.0).tag(sync=True)


class CostFunctionWidget(anywidget.AnyWidget):
    """Interactive image registration cost function demo."""
    _esm = _JS_DIR / "cost_function_widget.js"

    trans_x = traitlets.Float(0.0).tag(sync=True)
    trans_y = traitlets.Float(0.0).tag(sync=True)


class SmoothingWidget(anywidget.AnyWidget):
    """Interactive Gaussian spatial smoothing on a brain phantom."""
    _esm = _JS_DIR / "smoothing_widget.js"

    fwhm = traitlets.Float(0.0).tag(sync=True)
