import numpy as np

def coords(shape, dx):
    """Return coordinate grids (X,Y) in meters, centered."""
    ny, nx = shape
    x = (np.arange(nx) - nx//2) * dx
    y = (np.arange(ny) - ny//2) * dx
    return np.meshgrid(x, y)

def freqs(shape, dx):
    """Return frequency grids (FX,FY) in cycles/m, fftshifted."""
    ny, nx = shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    return np.meshgrid(fx, fy)