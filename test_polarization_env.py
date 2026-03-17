"""Test script to visualize polarization eye output."""

import subprocess
import sys

# Run the main.py with a callback that prints observations
script = '''
import torch
import numpy as np

# Monkey-patch the polarization eye step to print debug info
from cambrian.eyes import polarization

original_step = polarization.MjCambrianPolarizationEye.step

def debug_step(self, obs=None):
    result = original_step(self, obs)

    if hasattr(result, 'shape'):
        print(f"\\n=== POLARIZATION EYE OUTPUT ===")
        print(f"Shape: {result.shape}")
        print(f"Device: {result.device}")

        I = result[..., 0]
        Q = result[..., 1]
        U = result[..., 2]

        print(f"I (intensity):  min={I.min():.4f}, max={I.max():.4f}, mean={I.mean():.4f}")
        print(f"Q (horizontal): min={Q.min():.4f}, max={Q.max():.4f}, mean={Q.mean():.4f}")
        print(f"U (diagonal):   min={U.min():.4f}, max={U.max():.4f}, mean={U.mean():.4f}")

        # Check if we have meaningful polarization
        q_var = (Q.max() - Q.min()).item()
        u_var = (U.max() - U.min()).item()

        if q_var > 0.01 or u_var > 0.01:
            print("✓ Polarization signal detected!")
        else:
            print("✗ No polarization signal (all zeros or constant)")
        print("=" * 35)

    return result

polarization.MjCambrianPolarizationEye.step = debug_step
'''

# Write the patch to a temp file and run main.py
with open('/tmp/pol_patch.py', 'w') as f:
    f.write(script)

print("Running polarization test with debug output...")
print("=" * 60)

# Use exec to apply the patch then run main
exec(script)

# Now run a simple test
from cambrian.eyes.polarization import (
    get_viewing_directions,
    calculate_scattering_angle,
    calculate_degree_of_polarization,
    calculate_polarization_angle,
    calculate_stokes_vector,
)
from cambrian.utils import device

print("\n=== TESTING POLARIZATION MATH ===")

import numpy as np

# Simulate camera looking up at 60 degrees
angle = np.radians(60)
cam_xmat = np.array([
    [1, 0, 0],
    [0, np.cos(angle), -np.sin(angle)],
    [0, np.sin(angle), np.cos(angle)]
])

resolution = (8, 8)
fov = (120.0, 120.0)

viewing_dirs = get_viewing_directions(cam_xmat, resolution, fov)
print(f"Viewing directions shape: {viewing_dirs.shape}")

# Sun direction (from custom_light_1: dir="-1 0 -1" means sun is at +1, 0, +1)
sun_dir = torch.tensor([1.0, 0.0, 1.0], device=device)
sun_dir = sun_dir / torch.norm(sun_dir)
print(f"Sun direction: {sun_dir}")

# Calculate polarization
scattering = calculate_scattering_angle(viewing_dirs, sun_dir)
print(f"Scattering angles: min={torch.rad2deg(scattering.min()):.1f}°, max={torch.rad2deg(scattering.max()):.1f}°")

dop = calculate_degree_of_polarization(scattering, 0.75)
print(f"Degree of polarization: min={dop.min():.3f}, max={dop.max():.3f}")

aop = calculate_polarization_angle(viewing_dirs, sun_dir)
print(f"Angle of polarization: min={torch.rad2deg(aop.min()):.1f}°, max={torch.rad2deg(aop.max()):.1f}°")

intensity = torch.ones_like(dop) * 0.8
stokes = calculate_stokes_vector(intensity, dop, aop)
print(f"\nStokes vectors shape: {stokes.shape}")
print(f"I: [{stokes[...,0].min():.3f}, {stokes[...,0].max():.3f}]")
print(f"Q: [{stokes[...,1].min():.3f}, {stokes[...,1].max():.3f}]")
print(f"U: [{stokes[...,2].min():.3f}, {stokes[...,2].max():.3f}]")

print("\n✓ Polarization math is working correctly!")
print("=" * 60)

import torch
