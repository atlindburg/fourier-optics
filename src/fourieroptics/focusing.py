import numpy as np

def thin_lens_1D(x, focal_length, wavelength):
    """
    Generate a 1D thin-lens phase mask.

    Parameters
    ----------
    x : ndarray
        1D spatial grid (meters).
    focal_length : float
        Lens focal length (meters). Positive for a converging lens.
    wavelength : float
        Wavelength (meters).

    Returns
    -------
    lens : 1D ndarray (complex)
        Complex phase mask representing the lens: exp(-1j * k * x^2 / (2 f)).
    """
    k = 2 * np.pi / wavelength
    # thin lens quadratic phase (thin lens approximation)
    return np.exp(-1j * k * (x**2) / (2.0 * focal_length))
