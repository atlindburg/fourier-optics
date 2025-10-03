from .grid import coords_2D, freqs_2D, coords_1D, freqs_1D   # import 1D versions if you have them
from .utils import fft2, ifft2, fft1, ifft1, energy
from .propagation import (
    fraunhofer_2D, fresnel_tf_2D, angular_spectrum_2D,
    fraunhofer_1D, fresnel_tf_1D, angular_spectrum_1D,
    z_step_1D
)
from .masks import pinhole_1D, double_slit_1D, rectangular_grating_1D,sinusoidal_grating_1D, holographic_mask
from .sources import gaussian_beam_1D
from .focusing import thin_lens_1D

__all__ = [
    # grids
    "coords", "freqs", "coords_1D", "freqs_1D",
    # utils
    "fft2", "ifft2", "fft1", "ifft1", "energy",
    # 2D propagators
    "fraunhofer_2D", "fresnel_tf_2D", "angular_spectrum_2D",
    # 1D propagators
    "fraunhofer_1D", "fresnel_tf_1D", "angular_spectrum_1D",
]

__version__ = "0.1.0"

