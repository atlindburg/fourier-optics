import numpy as np
from .utils import fft2, ifft2, fft1, ifft1, energy
from .grid import freqs_2D, freqs_1D

def fresnel_tf_1D(u0, dx, wavelength, z, aperture_size=None):
    """
    1D Fresnel propagation using transfer function method.

    Parameters
    ----------
    u0 : ndarray
        Input 1D field.
    dx : float
        Grid spacing (meters).
    wavelength : float
        Wavelength (meters).
    z : float
        Propagation distance (meters).
    aperture_size : float, optional
        Physical size of the aperture for Fresnel number check.
        If provided, will raise ValueError if Fresnel number is too large.
    """
    FX = freqs_1D(u0.shape[0], dx)
    k = 2 * np.pi / wavelength
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2))
    U0 = fft1(u0)
    return ifft1(U0 * H)

def angular_spectrum_1D(u0, dx, wavelength, z):
    """
    1D Angular spectrum propagation.
    """
    FX = freqs_1D(u0.shape[0], dx)
    k = 2 * np.pi / wavelength
    kz = np.sqrt((k**2 - (2*np.pi*FX)**2).astype(complex))
    H = np.exp(1j * kz * z)
    U0 = fft1(u0)
    return ifft1(U0 * H)

def fresnel_tf_2D(u0, dx, wavelength, z):
    """Fresnel propagation using transfer function method."""
    FX, FY = freqs_2D(u0.shape, dx)
    k = 2*np.pi / wavelength
    H = np.exp(1j * k * z) * np.exp(-1j*np.pi*wavelength*z*(FX**2 + FY**2))
    U0 = fft2(u0)
    return ifft2(U0 * H)

def angular_spectrum_2D(u0, dx, wavelength, z):
    """Angular spectrum propagation."""
    FX, FY = freqs_2D(u0.shape, dx)
    k = 2*np.pi / wavelength
    kz = np.sqrt((k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2).astype(complex))
    H = np.exp(1j * kz * z)
    U0 = fft2(u0)
    return ifft2(U0 * H)

def z_step_1D(E0, x, propagator, wavelength, z_range, n_points=500):
    """
    Compute intensity along z for a 1D field using a given propagator.

    Parameters
    ----------
    E0 : ndarray
        Input 1D field at z=0.
    x : ndarray
        1D coordinate grid corresponding to E0 (meters).
    propagator : callable
        Function with signature propagator(E0, dx, wavelength, z) -> propagated field.
    wavelength : float
        Wavelength in meters.
    z_range : float
        Maximum propagation distance in meters.
    n_points : int
        Number of z-steps.

    Returns
    -------
    intensity_map : ndarray
        2D array of normalized intensity. Shape: (len(E0), n_points)
    z_steps : ndarray
        Array of z positions corresponding to intensity_map columns.
    """
    dx = x[1] - x[0]  # spacing from the input grid
    z_steps = np.linspace(0, z_range, n_points)
    intensity_map = np.zeros((E0.size, n_points))

    for idx, z_val in enumerate(z_steps):
        E_z = propagator(E0, dx, wavelength, z_val)
        # Some propagators return tuple (field, dx_out)
        if isinstance(E_z, tuple):
            E_z = E_z[0]
        intensity_map[:, idx] = np.abs(E_z)**2
        intensity_map[:, idx] /= np.max(intensity_map[:, idx])

    return intensity_map, z_steps