import sys
import os

# Add the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import matplotlib.pyplot as plt
import src as fo

# Parameters
wavelength = 633e-9     # 633 nm
dx = 10e-6              # pixel size (10 µm)
shape = (512, 512)
z = 0.1                 # 10 cm propagation

# Generate coordinate grid
X, Y = fo.coords(shape, dx)
r = np.sqrt(X**2 + Y**2)

# Input field: circular aperture of radius 0.5 mm
ap_radius = 0.5e-3
u0 = np.zeros(shape, dtype=complex)
u0[r < ap_radius] = 1.0

# --- Propagate with different methods ---
u_fraunhofer, out_dx = fo.fraunhofer_2D(u0, dx, wavelength, z)
u_fresnel = fo.fresnel_tf_2D(u0, dx, wavelength, z)
u_angspec = fo.angular_spectrum_2D(u0, dx, wavelength, z)

# --- Plot results ---
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].imshow(np.abs(u0)**2, cmap="inferno", extent=[X.min(), X.max(), Y.min(), Y.max()])
axes[0].set_title("Input aperture")

axes[1].imshow(np.abs(u_fraunhofer)**2, cmap="inferno")
axes[1].set_title("Fraunhofer intensity")

axes[2].imshow(np.abs(u_fresnel)**2, cmap="inferno")
axes[2].set_title("Fresnel intensity")

axes[3].imshow(np.abs(u_angspec)**2, cmap="inferno")
axes[3].set_title("Angular Spectrum intensity")

for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.show()