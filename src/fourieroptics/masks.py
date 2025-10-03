import numpy as np

def pinhole_1D(x, width):
    """
    Generate a 1D pinhole (slit) aperture field.

    Parameters
    ----------
    x : ndarray
        1D spatial grid (meters).
    width : float
        Width of the pinhole (meters).

    Returns
    -------
    aperture : 1D ndarray (complex)
        Aperture field: 1 inside the pinhole, 0 outside.
    """
    aperture = np.zeros_like(x, dtype=complex)
    aperture[np.abs(x) < width/2] = 1.0
    return aperture

def double_slit_1D(x, slit_width, separation):
    """
    1D double-slit aperture field.
    """
    aperture = np.zeros_like(x, dtype=complex)
    # First slit
    aperture[np.abs(x + separation/2) < slit_width/2] = 1.0
    # Second slit
    aperture[np.abs(x - separation/2) < slit_width/2] = 1.0
    return aperture

def square_grating_1D(x, period, duty_cycle=0.5, phase_shift=0.0, binary_phase=False):
    """
    1D square grating mask using sine construction.

    Parameters
    ----------
    x : ndarray
        Spatial coordinate grid (meters).
    period : float
        Grating period in meters.
    duty_cycle : float, optional
        Fraction of period that is "open" (default=0.5).
        Implemented by thresholding sine wave.
    phase_shift : float, optional
        Phase shift applied to the "open" regions (radians).
        Only relevant if binary_phase=True.
    binary_phase : bool, optional
        If False → binary amplitude grating (open=1, closed=0).
        If True → binary phase grating (open=exp(i*phase_shift), closed=1).

    Returns
    -------
    mask : ndarray (complex)
        Complex transmission function of the grating.
    """
    # sine-based square wave in [-1, 1]
    sq_wave = np.sign(np.sin(2 * np.pi * x / period))

    # convert to [0,1] mask with adjustable duty cycle
    mask = (sq_wave > np.cos(np.pi * duty_cycle)).astype(float)

    if binary_phase:
        out = np.ones_like(x, dtype=complex)
        out[mask == 1] = np.exp(1j * phase_shift)
        return out
    else:
        return mask.astype(complex)

