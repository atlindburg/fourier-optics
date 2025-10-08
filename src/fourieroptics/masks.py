import numpy as np
from .propagation import angular_spectrum_1D

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

def rectangular_grating_1D(x, period, duty_cycle=0.5, phase_shift=0.0, binary_phase=False, amplitude=1.0):
    """
    1D rectangular grating mask with adjustable duty cycle.
    """
    # position within each period, 0 <= t < 1
    t = (x % period) / period
    
    # open where t < duty_cycle
    mask = np.zeros_like(x, dtype=float)
    mask[t < duty_cycle] = amplitude
    
    if binary_phase:
        out = np.ones_like(x, dtype=complex)
        out[mask == 1] = np.exp(1j * phase_shift)
        return out
    else:
        return mask.astype(complex)

def sinusoidal_grating_1D(x, period, amplitude=1.0, phase=0.0, offset=0.0):
    """
    Generate a 1D sinusoidal grating.

    Parameters
    ----------
    x : ndarray
        1D spatial coordinate array (meters).
    period : float
        Grating period (meters).
    amplitude : float, optional
        Peak-to-peak amplitude of the grating (default=1.0).
    phase : float, optional
        Phase offset of the sine wave in radians (default=0.0).
    offset : float, optional
        Constant offset added to the grating (default=0.0).

    Returns
    -------
    grating : ndarray
        1D sinusoidal grating values (same shape as x).
    """
    grating = amplitude * np.sin(2 * np.pi * x / period + phase) + offset
    return grating

def holographic_mask(E_source, E_target, dx, wavelength, z, phase_only=True):
    """
    Generate a 1D holographic mask that transforms a source into a target at distance z.

    Parameters
    ----------
    E_source : ndarray (complex)
        Source field at initial plane.
    E_target : ndarray (complex)
        Desired target field at reconstruction plane.
    dx : float
        Spatial sampling.
    wavelength : float
        Wavelength of light.
    z : float
        Distance from mask to target plane.
    phase_only : bool
        If True, return phase-only hologram.

    Returns
    -------
    H : ndarray (complex)
        Holographic mask to apply to source field.
    """
    
    epsilon = 1e-12

    # Backward propagate target to hologram plane
    E_target_back = angular_spectrum_1D(E_target, dx, wavelength, -z)
    
    # Forward propagate source to hologram plane
    E_source_forward = angular_spectrum_1D(E_source, dx, wavelength, z)
    
    # Compute hologram mask
    H = E_target_back / (E_source_forward + epsilon)
    
    if phase_only:
        H = np.exp(1j * np.angle(H))
    else:
        H /= np.max(np.abs(H))
    
    return H

def blazed_grating_1D(x, period, blaze_depth=2*np.pi):
    """
    Generate a 1D blazed (sawtooth) phase grating.

    Parameters
    ----------
    x : ndarray
        1D spatial coordinate array (meters).
    period : float
        Grating period (meters).
    blaze_depth : float
        Maximum phase shift per period (default 2π, i.e., full blaze).

    Returns
    -------
    grating : ndarray (complex)
        Complex phase transmission function: exp(i * φ(x)).
    """
    phi = np.mod(blaze_depth * (x / period), blaze_depth)
    grating = np.exp(1j * phi)
    grating = np.angle(grating)
    grating = grating / np.max(np.abs(grating))  # Normalize to [-1, 1]
    return grating

def sinusoidal_phase_grating_1D(x, period, phase_depth=np.pi, phase_offset=0.0):
    """
    Generate a 1D sinusoidal *phase* grating.

    Parameters
    ----------
    x : ndarray
        1D spatial coordinate array (meters).
    period : float
        Grating period (meters).
    phase_depth : float, optional
        Maximum phase modulation depth (radians), default is π.
    phase_offset : float, optional
        Constant phase offset (radians), default is 0.

    Returns
    -------
    grating : ndarray (complex)
        Complex transmission function representing the phase grating:
        exp(i * phase_depth * sin(2πx / period + phase_offset))
    """
    spatial_phase = 2 * np.pi * x / period + phase_offset
    grating = np.exp(1j * phase_depth * np.sin(spatial_phase))
    return grating

def vls_grating_1D(x, g0, b2=0.0, b3=0.0, b4=0.0, threshold=0.5):
    """
    Generate a 1D Variable Line Spacing (VLS) grating mask.

    Parameters
    ----------
    x : ndarray
        1D spatial coordinate array (in meters).
    g0 : float
        Nominal groove density in lines per meter.
    b2, b3, b4 : float
        VLS polynomial coefficients (default 0). Units:
            b2 [m^-1], b3 [m^-2], b4 [m^-3]
    threshold : float, optional
        Duty cycle threshold (default 0.5).

    Returns
    -------
    grating : ndarray
        1D binary array (0/1) representing the VLS grating pattern.
    """
    # Local groove density variation g(x)
    g_of_x = g0 + 2*b2*x + 3*b3*x**2 + 4*b4*x**3

    # Cumulative line number integral (discretized)
    dx = x[1] - x[0]
    line_numbers = np.cumsum(g_of_x) * dx

    # Binary grating mask based on fractional line position
    grating = np.where(np.mod(line_numbers, 1) < threshold, 1.0, 0.0)
    return grating
