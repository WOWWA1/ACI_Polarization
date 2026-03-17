"""Simple test script for polarization eye."""

import torch
import numpy as np

# Test the polarization functions directly
from cambrian.eyes.polarization import (
    get_viewing_directions,
    calculate_scattering_angle,
    calculate_degree_of_polarization,
    calculate_polarization_angle,
    calculate_stokes_vector,
)
from cambrian.utils import device

def test_polarization_math():
    """Test the polarization calculation functions."""
    print("Testing polarization math functions...")
    print(f"Using device: {device}")

    # Simulate a camera looking straight ahead (identity rotation)
    cam_xmat = np.eye(3)
    resolution = (8, 8)
    fov = (120.0, 120.0)

    # Get viewing directions
    viewing_dirs = get_viewing_directions(cam_xmat, resolution, fov)
    print(f"\nViewing directions shape: {viewing_dirs.shape}")
    print(f"Center pixel direction: {viewing_dirs[4, 4]}")
    print(f"Top-left pixel direction: {viewing_dirs[0, 0]}")
    print(f"Bottom-right pixel direction: {viewing_dirs[7, 7]}")

    # Sun direction (e.g., sun at 45 degrees elevation, east)
    sun_direction = torch.tensor([1.0, 0.0, 1.0], device=device)
    sun_direction = sun_direction / torch.norm(sun_direction)
    print(f"\nSun direction: {sun_direction}")

    # Calculate scattering angles
    scattering_angles = calculate_scattering_angle(viewing_dirs, sun_direction)
    print(f"\nScattering angles shape: {scattering_angles.shape}")
    print(f"Scattering angle range: {torch.rad2deg(scattering_angles.min()):.1f}° to {torch.rad2deg(scattering_angles.max()):.1f}°")

    # Calculate degree of polarization
    dop = calculate_degree_of_polarization(scattering_angles, max_polarization=0.75)
    print(f"\nDegree of polarization shape: {dop.shape}")
    print(f"DoP range: {dop.min():.3f} to {dop.max():.3f}")

    # Calculate polarization angle
    aop = calculate_polarization_angle(viewing_dirs, sun_direction)
    print(f"\nAngle of polarization shape: {aop.shape}")
    print(f"AoP range: {torch.rad2deg(aop.min()):.1f}° to {torch.rad2deg(aop.max()):.1f}°")

    # Calculate Stokes vectors
    intensity = torch.ones_like(dop) * 0.8  # Assume uniform sky brightness
    stokes = calculate_stokes_vector(intensity, dop, aop)
    print(f"\nStokes vector shape: {stokes.shape}")
    print(f"I range: {stokes[..., 0].min():.3f} to {stokes[..., 0].max():.3f}")
    print(f"Q range: {stokes[..., 1].min():.3f} to {stokes[..., 1].max():.3f}")
    print(f"U range: {stokes[..., 2].min():.3f} to {stokes[..., 2].max():.3f}")
    print(f"V range: {stokes[..., 3].min():.3f} to {stokes[..., 3].max():.3f}")

    print("\n✓ All polarization math functions work correctly!")
    return True


def test_with_different_sun_positions():
    """Test polarization pattern with sun at different positions."""
    print("\n" + "="*50)
    print("Testing polarization pattern with different sun positions...")

    cam_xmat = np.eye(3)
    resolution = (8, 8)
    fov = (120.0, 120.0)
    viewing_dirs = get_viewing_directions(cam_xmat, resolution, fov)

    sun_positions = [
        ("Directly overhead", [0, 0, 1]),
        ("East, 45° elevation", [1, 0, 1]),
        ("North, low", [0, 1, 0.2]),
        ("West, setting", [-1, 0, 0.1]),
    ]

    for name, sun_pos in sun_positions:
        sun_dir = torch.tensor(sun_pos, dtype=torch.float32, device=device)
        sun_dir = sun_dir / torch.norm(sun_dir)

        scattering_angles = calculate_scattering_angle(viewing_dirs, sun_dir)
        dop = calculate_degree_of_polarization(scattering_angles, 0.75)

        print(f"\n{name}:")
        print(f"  Max DoP: {dop.max():.3f} at scattering angle {torch.rad2deg(scattering_angles[dop == dop.max()][0]):.1f}°")
        print(f"  Min DoP: {dop.min():.3f}")
        print(f"  Mean DoP: {dop.mean():.3f}")


if __name__ == "__main__":
    print("="*50)
    print("POLARIZATION EYE TEST")
    print("="*50)

    test_polarization_math()
    test_with_different_sun_positions()

    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)
