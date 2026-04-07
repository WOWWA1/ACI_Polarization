"""DoP and AoP heatmap visualization for polarization compass environment.

This module generates Degree of Polarization (DoP) and Angle of Polarization (AoP)
heatmaps for the visible sky hemisphere, computed dynamically based on sun position.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np


def compute_polarization_heatmap(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    resolution: int = 256,
    max_polarization: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DoP and AoP heatmaps for entire upper hemisphere.

    Uses the Rayleigh sky model to compute polarization patterns based on
    the scattering angle from the sun position.

    Args:
        sun_azimuth_deg: Sun azimuth in degrees (0 = +X, 90 = +Y).
        sun_elevation_deg: Sun elevation in degrees (0 = horizon, 90 = zenith).
        resolution: Output image resolution (square image).
        max_polarization: Maximum degree of polarization (Rayleigh max ~0.75-0.8).

    Returns:
        dop_heatmap: (resolution, resolution) float array [0, 1]
        aop_heatmap: (resolution, resolution) float array [0, 2*pi]
    """
    # Create pixel grid
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)

    # Convert to hemisphere coordinates using azimuthal equidistant projection
    theta, phi = image_to_hemisphere(xx, yy)

    # Compute viewing directions for each pixel
    view_dirs = hemisphere_to_direction(theta, phi)

    # Compute sun direction
    sun_az_rad = np.radians(sun_azimuth_deg)
    sun_el_rad = np.radians(sun_elevation_deg)
    sun_dir = np.array([
        np.cos(sun_el_rad) * np.cos(sun_az_rad),
        np.cos(sun_el_rad) * np.sin(sun_az_rad),
        np.sin(sun_el_rad),
    ])

    # Calculate scattering angle
    scattering_angle = calculate_scattering_angle(view_dirs, sun_dir)

    # Calculate DoP using Rayleigh formula
    dop = calculate_degree_of_polarization(scattering_angle, max_polarization)

    # Calculate AoP
    aop = calculate_angle_of_polarization(view_dirs, sun_dir)

    # Mask out pixels outside the hemisphere (r > 1)
    r = np.sqrt(xx**2 + yy**2)
    mask = r > 1
    dop[mask] = np.nan
    aop[mask] = np.nan

    return dop, aop


def image_to_hemisphere(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert image coordinates to hemisphere coordinates.

    Uses azimuthal equidistant projection (fisheye-like) where:
    - Image center (0, 0) maps to zenith (theta = pi/2)
    - Image edge (r = 1) maps to horizon (theta = 0)

    Args:
        x: X coordinates in [-1, 1] range
        y: Y coordinates in [-1, 1] range

    Returns:
        theta: Elevation angles (0 = horizon, pi/2 = zenith)
        phi: Azimuth angles (0 = +X, pi/2 = +Y)
    """
    r = np.sqrt(x**2 + y**2)
    r = np.clip(r, 0, 1)

    # Elevation: r=0 -> theta=pi/2 (zenith), r=1 -> theta=0 (horizon)
    theta = (1 - r) * (np.pi / 2)

    # Azimuth: angle from +X axis
    phi = np.arctan2(y, x)

    return theta, phi


def hemisphere_to_direction(
    theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """Convert hemisphere coordinates to unit direction vectors.

    Args:
        theta: Elevation angles (0 = horizon, pi/2 = zenith)
        phi: Azimuth angles (0 = +X, pi/2 = +Y)

    Returns:
        Viewing direction vectors with shape (*theta.shape, 3)
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # For elevation from horizon: x,y use cos(theta), z uses sin(theta)
    dx = cos_theta * np.cos(phi)
    dy = cos_theta * np.sin(phi)
    dz = sin_theta

    return np.stack([dx, dy, dz], axis=-1)


def calculate_scattering_angle(
    view_dirs: np.ndarray, sun_dir: np.ndarray
) -> np.ndarray:
    """Calculate scattering angle between viewing directions and sun.

    Args:
        view_dirs: Per-pixel viewing directions, shape (..., 3)
        sun_dir: Unit vector pointing toward sun, shape (3,)

    Returns:
        Scattering angles in radians
    """
    cos_gamma = np.sum(view_dirs * sun_dir, axis=-1)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    return np.arccos(cos_gamma)


def calculate_degree_of_polarization(
    scattering_angle: np.ndarray, max_polarization: float = 0.8
) -> np.ndarray:
    """Calculate degree of polarization using Rayleigh formula.

    DoP = max_pol * (1 - cos^2(gamma)) / (1 + cos^2(gamma))

    Maximum polarization occurs at gamma = 90 degrees (perpendicular to sun).
    Zero polarization at gamma = 0 or 180 degrees (looking at/away from sun).

    Args:
        scattering_angle: Scattering angles in radians
        max_polarization: Maximum degree of polarization

    Returns:
        Degree of polarization [0, max_pol]
    """
    cos_gamma_sq = np.cos(scattering_angle) ** 2
    dop = max_polarization * (1 - cos_gamma_sq) / (1 + cos_gamma_sq)
    return dop


def calculate_angle_of_polarization(
    view_dirs: np.ndarray, sun_dir: np.ndarray
) -> np.ndarray:
    """Calculate angle of polarization (E-vector orientation).

    The E-vector is perpendicular to the scattering plane (plane containing
    sun, observer, and sky point being viewed).

    Args:
        view_dirs: Per-pixel viewing directions, shape (..., 3)
        sun_dir: Unit vector pointing toward sun, shape (3,)

    Returns:
        Polarization angles in radians [0, 2*pi)
    """
    # Scattering plane normal: cross(sun_dir, view_dir)
    scattering_normal = np.cross(sun_dir, view_dirs)
    scattering_norm = np.linalg.norm(scattering_normal, axis=-1, keepdims=True)

    # Handle degenerate case (viewing directly at/away from sun)
    valid = scattering_norm[..., 0] > 1e-6
    scattering_normal = np.where(
        scattering_norm > 1e-6,
        scattering_normal / scattering_norm,
        np.zeros_like(scattering_normal),
    )

    # E-vector is perpendicular to both viewing direction and scattering normal
    e_vector = np.cross(view_dirs, scattering_normal)
    e_norm = np.linalg.norm(e_vector, axis=-1, keepdims=True)
    e_vector = np.where(
        e_norm > 1e-6,
        e_vector / e_norm,
        np.zeros_like(e_vector),
    )

    # Reference "up" vector for measuring angle (world Z axis)
    up = np.array([0.0, 0.0, 1.0])

    # Project up onto the plane perpendicular to viewing direction
    view_norm = view_dirs / (np.linalg.norm(view_dirs, axis=-1, keepdims=True) + 1e-8)
    up_proj = up - np.sum(up * view_norm, axis=-1, keepdims=True) * view_norm
    up_proj_norm = np.linalg.norm(up_proj, axis=-1, keepdims=True)
    up_proj = np.where(
        up_proj_norm > 1e-6,
        up_proj / up_proj_norm,
        np.array([1.0, 0.0, 0.0]),
    )

    # Angle between E-vector and projected up reference
    cos_angle = np.sum(e_vector * up_proj, axis=-1)
    cross_prod = np.cross(up_proj, e_vector)
    sin_angle = np.sum(cross_prod * view_norm, axis=-1)

    aop = np.arctan2(sin_angle, cos_angle)

    # Map to [0, 2*pi)
    aop = aop % (2 * np.pi)

    return aop


def render_heatmaps(
    dop: np.ndarray,
    aop: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    output_path: str,
    episode: Optional[int] = None,
) -> None:
    """Render DoP and AoP as side-by-side colormapped images.

    Args:
        dop: DoP heatmap array (resolution, resolution) with NaN for masked pixels
        aop: AoP heatmap array (resolution, resolution) with NaN for masked pixels
        sun_azimuth_deg: Sun azimuth in degrees
        sun_elevation_deg: Sun elevation in degrees
        output_path: Path to save the output image
        episode: Optional episode number for title
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    resolution = dop.shape[0]

    # DoP heatmap with viridis colormap
    ax1 = axes[0]
    im1 = ax1.imshow(
        dop,
        cmap="viridis",
        vmin=0,
        vmax=1,
        origin="lower",
        extent=[-1, 1, -1, 1],
    )
    ax1.set_title("Degree of Polarization (DoP)")
    plt.colorbar(im1, ax=ax1, label="DoP")

    # Add sun marker
    sun_r = 1 - sun_elevation_deg / 90  # r=0 at zenith, r=1 at horizon
    sun_x = sun_r * np.cos(np.radians(sun_azimuth_deg))
    sun_y = sun_r * np.sin(np.radians(sun_azimuth_deg))
    ax1.plot(sun_x, sun_y, "o", color="yellow", markersize=10, markeredgecolor="black")

    # Add cardinal directions
    ax1.text(0.95, 0, "E", ha="center", va="center", fontsize=12, fontweight="bold")
    ax1.text(-0.95, 0, "W", ha="center", va="center", fontsize=12, fontweight="bold")
    ax1.text(0, 0.95, "N", ha="center", va="center", fontsize=12, fontweight="bold")
    ax1.text(0, -0.95, "S", ha="center", va="center", fontsize=12, fontweight="bold")

    # Draw horizon circle
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta_circle), np.sin(theta_circle), "k--", alpha=0.5)

    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # AoP heatmap with cyclic HSV colormap
    ax2 = axes[1]

    # Create cyclic colormap: map AoP [0, 2*pi] to hue [0, 1]
    aop_normalized = aop / (2 * np.pi)
    hsv = np.zeros((*aop.shape, 3))
    hsv[..., 0] = aop_normalized  # Hue from AoP
    hsv[..., 1] = 1.0  # Full saturation
    hsv[..., 2] = np.where(np.isnan(aop), 0, 1)  # Value (masked = black)

    # Handle NaN values
    hsv[np.isnan(aop), :] = 0

    rgb = hsv_to_rgb(hsv)
    im2 = ax2.imshow(rgb, origin="lower", extent=[-1, 1, -1, 1])
    ax2.set_title("Angle of Polarization (AoP)")

    # Add sun marker
    ax2.plot(sun_x, sun_y, "o", color="yellow", markersize=10, markeredgecolor="black")

    # Add cardinal directions
    ax2.text(0.95, 0, "E", ha="center", va="center", fontsize=12, fontweight="bold")
    ax2.text(-0.95, 0, "W", ha="center", va="center", fontsize=12, fontweight="bold")
    ax2.text(0, 0.95, "N", ha="center", va="center", fontsize=12, fontweight="bold")
    ax2.text(0, -0.95, "S", ha="center", va="center", fontsize=12, fontweight="bold")

    # Draw horizon circle
    ax2.plot(np.cos(theta_circle), np.sin(theta_circle), "k--", alpha=0.5)

    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Add AoP colorbar legend
    # Create a small colorbar showing AoP mapping
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    hsv_bar = np.zeros((256, 1, 3))
    hsv_bar[:, 0, 0] = gradient[:, 0]
    hsv_bar[:, 0, 1] = 1.0
    hsv_bar[:, 0, 2] = 1.0
    rgb_bar = hsv_to_rgb(hsv_bar)
    cax.imshow(rgb_bar, origin="lower", aspect="auto")
    cax.set_yticks([0, 64, 128, 192, 255])
    cax.set_yticklabels(["0", "90", "180", "270", "360"])
    cax.set_ylabel("AoP (degrees)")
    cax.set_xticks([])

    # Title
    if episode is not None:
        fig.suptitle(
            f"Episode {episode} - Sun at {sun_azimuth_deg:.0f} az, {sun_elevation_deg:.0f} el",
            fontsize=14,
        )
    else:
        fig.suptitle(
            f"Sun at {sun_azimuth_deg:.0f} az, {sun_elevation_deg:.0f} el",
            fontsize=14,
        )

    plt.subplots_adjust(left=0.08, right=0.88, top=0.9, bottom=0.1, wspace=0.3)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_polarization_heatmap(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    output_path: str,
    episode: Optional[int] = None,
    resolution: int = 256,
    max_polarization: float = 0.8,
) -> None:
    """Generate and save polarization heatmaps for given sun position.

    Convenience function that computes and renders heatmaps in one call.

    Args:
        sun_azimuth_deg: Sun azimuth in degrees (0 = +X, 90 = +Y)
        sun_elevation_deg: Sun elevation in degrees (0 = horizon, 90 = zenith)
        output_path: Path to save the output image
        episode: Optional episode number for title
        resolution: Heatmap resolution (pixels)
        max_polarization: Maximum degree of polarization
    """
    dop, aop = compute_polarization_heatmap(
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        resolution=resolution,
        max_polarization=max_polarization,
    )

    render_heatmaps(
        dop=dop,
        aop=aop,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        output_path=output_path,
        episode=episode,
    )


def cubemap_face_directions(face_size: int, face_index: int) -> np.ndarray:
    """Compute viewing direction for each pixel on a cubemap face.

    Uses MuJoCo's internal cubemap face ordering (verified empirically — X/Y swapped):
        0: -X (back)
        1: +X (front)
        2: +Z (up/sky)
        3: -Z (down/ground)
        4: -Y (left)
        5: +Y (right)

    Args:
        face_size: Resolution of each face (width/height in pixels)
        face_index: 0-5 corresponding to face order above

    Returns:
        directions: (face_size, face_size, 3) unit vectors
    """
    # Create normalized pixel coordinates [-1, 1]
    u = np.linspace(-1, 1, face_size)
    v = np.linspace(-1, 1, face_size)
    uu, vv = np.meshgrid(u, v)

    directions = np.zeros((face_size, face_size, 3))

    if face_index == 0:  # -X (back)
        directions[..., 0] = -1
        directions[..., 1] = -uu
        directions[..., 2] = -vv
    elif face_index == 1:  # +X (front)
        directions[..., 0] = 1
        directions[..., 1] = uu
        directions[..., 2] = -vv
    elif face_index == 2:  # +Z (up/sky)
        directions[..., 0] = uu
        directions[..., 1] = -vv
        directions[..., 2] = 1
    elif face_index == 3:  # -Z (down/ground)
        directions[..., 0] = -uu
        directions[..., 1] = -vv
        directions[..., 2] = -1
    elif face_index == 4:  # -Y (left)
        directions[..., 0] = uu
        directions[..., 1] = -1
        directions[..., 2] = -vv
    elif face_index == 5:  # +Y (right)
        directions[..., 0] = -uu
        directions[..., 1] = 1
        directions[..., 2] = -vv

    # Normalize to unit vectors
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions / norms

    return directions


def compute_cubemap_polarization(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    face_size: int = 128,
    max_polarization: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DoP and AoP for all 6 cubemap faces.

    Args:
        sun_azimuth_deg: Sun azimuth in degrees (0 = +X, 90 = +Y)
        sun_elevation_deg: Sun elevation in degrees (0 = horizon, 90 = zenith)
        face_size: Resolution of each face (pixels)
        max_polarization: Maximum degree of polarization

    Returns:
        dop_cubemap: (6*face_size, face_size) stacked faces
        aop_cubemap: (6*face_size, face_size) stacked faces
    """
    # Compute sun direction
    sun_az_rad = np.radians(sun_azimuth_deg)
    sun_el_rad = np.radians(sun_elevation_deg)
    sun_dir = np.array([
        np.cos(sun_el_rad) * np.cos(sun_az_rad),
        np.cos(sun_el_rad) * np.sin(sun_az_rad),
        np.sin(sun_el_rad),
    ])

    # Allocate output arrays
    dop_cubemap = np.zeros((6 * face_size, face_size))
    aop_cubemap = np.zeros((6 * face_size, face_size))

    for face_idx in range(6):
        # Get viewing directions for this face
        view_dirs = cubemap_face_directions(face_size, face_idx)

        # Calculate scattering angle
        scattering_angle = calculate_scattering_angle(view_dirs, sun_dir)

        # Calculate DoP using Rayleigh formula
        dop = calculate_degree_of_polarization(scattering_angle, max_polarization)

        # Calculate AoP
        aop = calculate_angle_of_polarization(view_dirs, sun_dir)

        # Store in stacked format
        y_start = face_idx * face_size
        y_end = (face_idx + 1) * face_size
        dop_cubemap[y_start:y_end, :] = dop
        aop_cubemap[y_start:y_end, :] = aop

    return dop_cubemap, aop_cubemap


def render_cubemap_heatmaps(
    dop: np.ndarray,
    aop: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    output_path: str,
    episode: Optional[int] = None,
) -> None:
    """Render cubemap as cross-format visualization.

    Shows all 6 faces in standard cubemap cross layout:
          [+Z up]
    [-Y] [+X front] [+Y] [-X back]
          [-Z down]

    Args:
        dop: DoP cubemap array (6*face_size, face_size)
        aop: AoP cubemap array (6*face_size, face_size)
        sun_azimuth_deg: Sun azimuth in degrees
        sun_elevation_deg: Sun elevation in degrees
        output_path: Path to save the output image
        episode: Optional episode number for title
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    # Determine face size from input shape
    total_height, face_size = dop.shape
    assert total_height == 6 * face_size, "Expected 6 stacked square faces"

    # Extract individual faces
    dop_faces = []
    aop_faces = []
    for i in range(6):
        y_start = i * face_size
        y_end = (i + 1) * face_size
        dop_faces.append(dop[y_start:y_end, :])
        aop_faces.append(aop[y_start:y_end, :])

    # Create cross layout: 4 cols x 3 rows
    cross_width = face_size * 4
    cross_height = face_size * 3

    dop_cross = np.full((cross_height, cross_width), np.nan)
    aop_cross = np.full((cross_height, cross_width), np.nan)

    # Face placement in cross layout using MuJoCo's face ordering:
    # MuJoCo: 0=-Y, 1=+Y, 2=+Z, 3=-Z, 4=-X, 5=+X
    # Row 0, col 1: +Z (up) = face 2
    # Row 1, col 0: -X (left) = face 4
    # Row 1, col 1: +Y (front) = face 1
    # Row 1, col 2: +X (right) = face 5
    # Row 1, col 3: -Y (back) = face 0
    # Row 2, col 1: -Z (down) = face 3

    placements = [
        (2, 0, 1),   # +Z up: row 0, col 1
        (4, 1, 0),   # -X left: row 1, col 0
        (1, 1, 1),   # +Y front: row 1, col 1
        (5, 1, 2),   # +X right: row 1, col 2
        (0, 1, 3),   # -Y back: row 1, col 3
        (3, 2, 1),   # -Z down: row 2, col 1
    ]

    for face_idx, row, col in placements:
        y_start = row * face_size
        y_end = (row + 1) * face_size
        x_start = col * face_size
        x_end = (col + 1) * face_size
        dop_cross[y_start:y_end, x_start:x_end] = dop_faces[face_idx]
        aop_cross[y_start:y_end, x_start:x_end] = aop_faces[face_idx]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # DoP heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(
        dop_cross,
        cmap="viridis",
        vmin=0,
        vmax=1,
        origin="upper",
    )
    ax1.set_title("Degree of Polarization (DoP) - Cubemap")
    plt.colorbar(im1, ax=ax1, label="DoP", shrink=0.7)

    # Add face labels
    face_labels = [
        ("+Z (up)", 0, 1),
        ("-Y (left)", 1, 0),
        ("+X (front)", 1, 1),
        ("+Y (right)", 1, 2),
        ("-X (back)", 1, 3),
        ("-Z (down)", 2, 1),
    ]
    for label, row, col in face_labels:
        cx = (col + 0.5) * face_size
        cy = (row + 0.5) * face_size
        ax1.text(cx, cy, label, ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

    ax1.set_xlim(-0.5, cross_width - 0.5)
    ax1.set_ylim(cross_height - 0.5, -0.5)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # AoP heatmap with cyclic colormap
    ax2 = axes[1]

    # Create cyclic colormap
    aop_normalized = aop_cross / (2 * np.pi)
    hsv = np.zeros((*aop_cross.shape, 3))
    hsv[..., 0] = np.nan_to_num(aop_normalized, nan=0)
    hsv[..., 1] = np.where(np.isnan(aop_cross), 0, 1)
    hsv[..., 2] = np.where(np.isnan(aop_cross), 0.3, 1)  # Gray for empty areas

    rgb = hsv_to_rgb(hsv)
    ax2.imshow(rgb, origin="upper")
    ax2.set_title("Angle of Polarization (AoP) - Cubemap")

    # Add face labels
    for label, row, col in face_labels:
        cx = (col + 0.5) * face_size
        cy = (row + 0.5) * face_size
        ax2.text(cx, cy, label, ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

    ax2.set_xlim(-0.5, cross_width - 0.5)
    ax2.set_ylim(cross_height - 0.5, -0.5)
    ax2.set_aspect("equal")
    ax2.axis("off")

    # Add AoP colorbar
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    hsv_bar = np.zeros((256, 1, 3))
    hsv_bar[:, 0, 0] = gradient[:, 0]
    hsv_bar[:, 0, 1] = 1.0
    hsv_bar[:, 0, 2] = 1.0
    rgb_bar = hsv_to_rgb(hsv_bar)
    cax.imshow(rgb_bar, origin="lower", aspect="auto")
    cax.set_yticks([0, 64, 128, 192, 255])
    cax.set_yticklabels(["0", "90", "180", "270", "360"])
    cax.set_ylabel("AoP (degrees)")
    cax.set_xticks([])

    # Title
    if episode is not None:
        fig.suptitle(
            f"Episode {episode} - Cubemap Polarization - Sun at {sun_azimuth_deg:.0f} az, {sun_elevation_deg:.0f} el",
            fontsize=14,
        )
    else:
        fig.suptitle(
            f"Cubemap Polarization - Sun at {sun_azimuth_deg:.0f} az, {sun_elevation_deg:.0f} el",
            fontsize=14,
        )

    plt.subplots_adjust(left=0.02, right=0.88, top=0.9, bottom=0.05, wspace=0.1)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_cubemap_heatmap(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    output_path: str,
    episode: Optional[int] = None,
    face_size: int = 128,
    max_polarization: float = 0.8,
) -> None:
    """Generate and save cubemap polarization heatmaps for given sun position.

    Convenience function that computes and renders cubemap heatmaps in one call.

    Args:
        sun_azimuth_deg: Sun azimuth in degrees (0 = +X, 90 = +Y)
        sun_elevation_deg: Sun elevation in degrees (0 = horizon, 90 = zenith)
        output_path: Path to save the output image
        episode: Optional episode number for title
        face_size: Cubemap face resolution (pixels)
        max_polarization: Maximum degree of polarization
    """
    dop, aop = compute_cubemap_polarization(
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        face_size=face_size,
        max_polarization=max_polarization,
    )

    render_cubemap_heatmaps(
        dop=dop,
        aop=aop,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        output_path=output_path,
        episode=episode,
    )


def pixel_to_viewing_direction(
    pixel_row: int,
    pixel_col: int,
    resolution: Tuple[int, int],
    fov: Tuple[float, float],
    cam_xmat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert agent eye pixel to viewing direction in world coordinates.

    Args:
        pixel_row: Row index in agent's eye image (0 = top)
        pixel_col: Column index in agent's eye image (0 = left)
        resolution: (height, width) of the eye
        fov: (fov_y, fov_x) field of view in degrees
        cam_xmat: 3x3 camera rotation matrix. If None, assumes identity (camera
                  looking down -Z axis in world frame).

    Returns:
        Unit direction vector (3,) in world coordinates
    """
    H, W = resolution
    fov_y, fov_x = fov

    # Normalized pixel coordinates [-1, 1]
    u = (2 * pixel_col / (W - 1)) - 1 if W > 1 else 0
    v = (2 * pixel_row / (H - 1)) - 1 if H > 1 else 0

    # Convert FOV to tangent
    tan_fov_x = np.tan(np.radians(fov_x / 2))
    tan_fov_y = np.tan(np.radians(fov_y / 2))

    # Camera-space direction (MuJoCo camera looks down -Z axis)
    dir_cam = np.array([
        u * tan_fov_x,
        -v * tan_fov_y,  # Flip Y for image coordinates
        -1.0,
    ])
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    # Transform to world frame
    if cam_xmat is not None:
        dir_world = cam_xmat @ dir_cam
    else:
        dir_world = dir_cam

    return dir_world


def direction_to_cubemap_pixel(
    direction: np.ndarray,
    face_size: int,
) -> Tuple[int, int, int, Tuple[int, int]]:
    """Convert a viewing direction to cubemap face and pixel coordinates.

    Uses MuJoCo's internal cubemap face ordering (verified empirically — X/Y swapped):
        0: -X (back)
        1: +X (front)
        2: +Z (up/sky)
        3: -Z (down/ground)
        4: -Y (left)
        5: +Y (right)

    Args:
        direction: Unit direction vector (3,) in world coordinates
        face_size: Size of each cubemap face in pixels

    Returns:
        face_idx: Which cubemap face (0-5)
        pixel_row: Row in that face (0 = top)
        pixel_col: Column in that face (0 = left)
        stacked_coords: (row, col) in the 6-stacked skybox texture
    """
    x, y, z = direction

    # Find dominant axis to determine face
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)

    if abs_x >= abs_y and abs_x >= abs_z:
        # X-dominant
        if x > 0:
            face_idx = 1  # +X (front)
            u = y / x
            v = -z / x
        else:
            face_idx = 0  # -X (back)
            u = -y / (-x)
            v = -z / (-x)
    elif abs_z >= abs_x and abs_z >= abs_y:
        # Z-dominant
        if z > 0:
            face_idx = 2  # +Z (up)
            u = -x / z
            v = y / z
        else:
            face_idx = 3  # -Z (down)
            u = -x / (-z)
            v = -y / (-z)
    else:
        # Y-dominant
        if y > 0:
            face_idx = 5  # +Y (right)
            u = -x / y
            v = -z / y
        else:
            face_idx = 4  # -Y (left)
            u = x / (-y)
            v = -z / (-y)

    # Convert u, v from [-1, 1] to pixel coordinates
    pixel_col = int((u + 1) / 2 * (face_size - 1))
    pixel_row = int((v + 1) / 2 * (face_size - 1))

    # Clamp to valid range
    pixel_col = max(0, min(face_size - 1, pixel_col))
    pixel_row = max(0, min(face_size - 1, pixel_row))

    # Stacked format: faces are vertically stacked
    stacked_row = face_idx * face_size + pixel_row
    stacked_col = pixel_col

    return face_idx, pixel_row, pixel_col, (stacked_row, stacked_col)


def pixel_to_skybox_location(
    pixel_row: int,
    pixel_col: int,
    resolution: Tuple[int, int],
    fov: Tuple[float, float],
    face_size: int,
    cam_xmat: Optional[np.ndarray] = None,
) -> dict:
    """Map an agent eye pixel to its corresponding skybox location.

    This is the main convenience function that combines all steps.

    Args:
        pixel_row: Row index in agent's eye image (0 = top)
        pixel_col: Column index in agent's eye image (0 = left)
        resolution: (height, width) of the eye
        fov: (fov_y, fov_x) field of view in degrees
        face_size: Size of each cubemap face in pixels
        cam_xmat: 3x3 camera rotation matrix. If None, assumes identity.

    Returns:
        Dictionary with:
            - 'direction': viewing direction vector (3,)
            - 'face_idx': cubemap face index (0-5)
            - 'face_name': human-readable face name
            - 'face_pixel': (row, col) within the face
            - 'stacked_pixel': (row, col) in stacked skybox texture
            - 'cross_pixel': (row, col) in cross-format visualization

    Example:
        >>> result = pixel_to_skybox_location(
        ...     pixel_row=8, pixel_col=8,
        ...     resolution=(16, 16), fov=(140, 140),
        ...     face_size=128
        ... )
        >>> print(f"Looking at face {result['face_name']}")
        >>> print(f"Skybox pixel: {result['stacked_pixel']}")
    """
    FACE_NAMES = {
        0: "-Y (back)",
        1: "+Y (front)",
        2: "+Z (up)",
        3: "-Z (down)",
        4: "-X (left)",
        5: "+X (right)",
    }

    # Cross layout positions: (row, col) in units of face_size
    CROSS_POSITIONS = {
        0: (1, 3),  # -Y back
        1: (1, 1),  # +Y front
        2: (0, 1),  # +Z up
        3: (2, 1),  # -Z down
        4: (1, 0),  # -X left
        5: (1, 2),  # +X right
    }

    # Get viewing direction
    direction = pixel_to_viewing_direction(
        pixel_row, pixel_col, resolution, fov, cam_xmat
    )

    # Map to cubemap
    face_idx, face_row, face_col, stacked_coords = direction_to_cubemap_pixel(
        direction, face_size
    )

    # Calculate cross-format coordinates
    cross_row_offset, cross_col_offset = CROSS_POSITIONS[face_idx]
    cross_row = cross_row_offset * face_size + face_row
    cross_col = cross_col_offset * face_size + face_col

    return {
        'direction': direction,
        'face_idx': face_idx,
        'face_name': FACE_NAMES[face_idx],
        'face_pixel': (face_row, face_col),
        'stacked_pixel': stacked_coords,
        'cross_pixel': (cross_row, cross_col),
    }
