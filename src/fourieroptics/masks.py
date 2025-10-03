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

def rectangular_grating_1D(x, period, duty_cycle=0.5, phase_shift=0.0, binary_phase=False):
    """
    1D rectangular grating mask with adjustable duty cycle.
    """
    # position within each period, 0 <= t < 1
    t = (x % period) / period
    
    # open where t < duty_cycle
    mask = np.zeros_like(x, dtype=float)
    mask[t < duty_cycle] = 1.0
    
    if binary_phase:
        out = np.ones_like(x, dtype=complex)
        out[mask == 1] = np.exp(1j * phase_shift)
        return out
    else:
        return mask.astype(complex)

