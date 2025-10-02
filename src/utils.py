import numpy as np

def freqs_1D(n, dx):
    """Return frequency grid (cycles/m), fftshifted."""
    return np.fft.fftshift(np.fft.fftfreq(n, d=dx))

def fft1(u):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(u)))

def ifft1(U):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(U)))

def fft2(u):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))

def ifft2(U):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U)))

def energy(u):
    return np.sum(np.abs(u)**2)