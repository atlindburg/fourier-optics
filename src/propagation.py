import numpy as np
from .utils import fft2, ifft2
from .grid import freqs

def fraunhofer_1D(u0, dx, wavelength, z):
    """
    1D Fraunhofer (far-field) propagation.

    Parameters
    ----------
    u0 : 1D ndarray (complex)
        Input field.
    dx : float
        Sampling interval in meters.
    wavelength : float
        Wavelength in meters.
    z : float
        Propagation distance in meters.

    Returns
    -------
    u1 : 1D ndarray (complex)
        Output field in the far-field plane (scaled Fourier transform).
    out_dx : float
        Sampling pitch at the observation plane.
    """
    n = u0.shape[0]
    U0 = fft1(u0)

    # Output sampling interval
    out_dx = wavelength * z / (n * dx)

    # Normalization (energy conservation convention)
    u1 = (np.exp(1j * 2*np.pi / wavelength * z) /
          (1j * wavelength * z)) * U0

    return u1, out_dx

def fresnel_tf_1D(u0, dx, wavelength, z):
    """
    1D Fresnel propagation using transfer function method.
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

def fraunhofer_2D(u0, dx, wavelength, z):
    """Fraunhofer (far-field) propagation."""
    ny, nx = u0.shape
    FX, FY = freqs(u0.shape, dx)
    U0 = fft2(u0)
    # output sampling
    out_dx = wavelength * z / (nx * dx)
    # phase factor (optional, depends on convention)
    return U0, out_dx

def fresnel_tf_2D(u0, dx, wavelength, z):
    """Fresnel propagation using transfer function method."""
    FX, FY = freqs(u0.shape, dx)
    k = 2*np.pi / wavelength
    H = np.exp(1j * k * z) * np.exp(-1j*np.pi*wavelength*z*(FX**2 + FY**2))
    U0 = fft2(u0)
    return ifft2(U0 * H)

def angular_spectrum_2D(u0, dx, wavelength, z):
    """Angular spectrum propagation."""
    FX, FY = freqs(u0.shape, dx)
    k = 2*np.pi / wavelength
    kz = np.sqrt((k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2).astype(complex))
    H = np.exp(1j * kz * z)
    U0 = fft2(u0)
    return ifft2(U0 * H)