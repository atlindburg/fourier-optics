import numpy as np

def gaussian_beam_1D(x, w0, x0=0):
    """
    1D Gaussian beam field at z=0.
    """
    return np.exp(-((x - x0)**2) / (w0**2))