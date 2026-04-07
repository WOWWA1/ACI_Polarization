"""Polarization-sensing eye using the Rayleigh sky model and Stokes vectors.

This module implements a polarization-sensitive eye that calculates the polarization
pattern of skylight based on the Rayleigh scattering model. The output is Stokes
vectors [I, Q, U, V] for each pixel, where polarization is only calculated for
sky pixels (detected via depth buffer).
"""

from typing import Callable, Self, Tuple

import numpy as np
import torch
from gymnasium import spaces
from hydra_config import config_wrapper

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.utils import device, get_logger
from cambrian.utils.spec import MjCambrianSpec
from cambrian.utils.types import ObsType


@config_wrapper
class MjCambrianPolarizationEyeConfig(MjCambrianEyeConfig):
    """Configuration for polarization-sensing eye.

    Attributes:
        instance: Factory for MjCambrianPolarizationEye.
        sun_light_name: Name of the MuJoCo light to use as sun direction source.
        max_polarization: Maximum degree of polarization (Rayleigh theoretical max ~0.75).
        sky_depth_threshold: Normalized depth value above which a pixel is considered sky.
            Values close to 1.0 indicate pixels at the far clipping plane.
    """

    instance: Callable[[Self, str], "MjCambrianPolarizationEye"]

    sun_light_name: str
    max_polarization: float
    sky_depth_threshold: float


def get_sun_direction_from_light(spec: MjCambrianSpec, light_name: str) -> torch.Tensor:
    """Get the sun direction vector from a MuJoCo light source.

    Args:
        spec: The MuJoCo spec containing model and data.
        light_name: Name of the light in the MuJoCo model.

    Returns:
        torch.Tensor: Unit direction vector pointing toward the light (shape: (3,)).
    """
    light_id = spec.get_light_id(light_name)
    if light_id == -1:
        get_logger().warning(
            f"Light '{light_name}' not found. Using default sun direction (0, 0, 1).",
            extra={"once": True},
        )
        return torch.tensor([0.0, 0.0, 1.0], device=device)

    # MuJoCo light_xdir gives the direction the light is pointing
    # For sun, we want the direction FROM observer TO sun, which is opposite
    light_dir = spec.data.light_xdir[light_id].copy()
    sun_dir = torch.from_numpy(-light_dir).float().to(device)

    # Normalize
    norm = torch.norm(sun_dir)
    if norm > 1e-6:
        sun_dir = sun_dir / norm
    else:
        sun_dir = torch.tensor([0.0, 0.0, 1.0], device=device)

    return sun_dir


def get_viewing_directions(
    cam_xmat: np.ndarray,
    resolution: Tuple[int, int],
    fov: Tuple[float, float],
) -> torch.Tensor:
    """Generate viewing direction vectors for each pixel in world coordinates.

    Args:
        cam_xmat: 3x3 camera rotation matrix from MuJoCo (data.cam_xmat).
        resolution: (height, width) of the image.
        fov: (fov_y, fov_x) field of view in degrees.

    Returns:
        torch.Tensor: Viewing directions with shape (H, W, 3) in world frame.
    """
    H, W = resolution
    fov_y, fov_x = fov

    # Create normalized pixel coordinates [-1, 1]
    # Note: In image coordinates, (0,0) is top-left
    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")  # (H, W)

    # Convert FOV to tangent for perspective projection
    tan_fov_x = torch.tan(torch.deg2rad(torch.tensor(fov_x / 2, device=device)))
    tan_fov_y = torch.tan(torch.deg2rad(torch.tensor(fov_y / 2, device=device)))

    # Camera-space directions (MuJoCo camera looks down -Z axis)
    dir_x = uu * tan_fov_x
    dir_y = -vv * tan_fov_y  # Flip Y to match image coordinates
    dir_z = -torch.ones_like(uu)

    # Stack and normalize
    dirs_cam = torch.stack([dir_x, dir_y, dir_z], dim=-1)  # (H, W, 3)
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

    # Transform to world frame using camera rotation matrix
    cam_rot = torch.from_numpy(cam_xmat).float().to(device)
    dirs_world = torch.einsum("ij,hwj->hwi", cam_rot, dirs_cam)

    return dirs_world


def calculate_scattering_angle(
    viewing_directions: torch.Tensor,
    sun_direction: torch.Tensor,
) -> torch.Tensor:
    """Calculate the scattering angle between viewing directions and sun.

    The scattering angle gamma is the angle between the viewing direction
    and the sun direction: gamma = arccos(view_dir . sun_dir)

    Args:
        viewing_directions: Per-pixel viewing directions, shape (H, W, 3).
        sun_direction: Unit vector pointing toward sun, shape (3,).

    Returns:
        torch.Tensor: Scattering angles in radians, shape (H, W).
    """
    # Dot product between each viewing direction and sun direction
    cos_gamma = torch.sum(viewing_directions * sun_direction, dim=-1)
    cos_gamma = torch.clamp(cos_gamma, -1.0, 1.0)

    return torch.arccos(cos_gamma)


def calculate_degree_of_polarization(
    scattering_angle: torch.Tensor,
    max_polarization: float = 0.75,
) -> torch.Tensor:
    """Calculate degree of polarization using the Rayleigh formula.

    DoP = max_pol * (1 - cos^2(gamma)) / (1 + cos^2(gamma))

    Maximum polarization occurs at gamma = 90 degrees (perpendicular to sun).
    Zero polarization at gamma = 0 or 180 degrees (looking at/away from sun).

    Args:
        scattering_angle: Scattering angles in radians, shape (H, W).
        max_polarization: Maximum degree of polarization (typically ~0.75).

    Returns:
        torch.Tensor: Degree of polarization [0, max_pol], shape (H, W).
    """
    cos_gamma_sq = torch.cos(scattering_angle) ** 2
    dop = max_polarization * (1 - cos_gamma_sq) / (1 + cos_gamma_sq)
    return dop


def calculate_polarization_angle(
    viewing_directions: torch.Tensor,
    sun_direction: torch.Tensor,
) -> torch.Tensor:
    """Calculate the angle of polarization (E-vector orientation).

    The E-vector is perpendicular to the scattering plane, which is the plane
    containing the sun, observer, and sky point being viewed.

    Args:
        viewing_directions: Per-pixel viewing directions, shape (H, W, 3).
        sun_direction: Unit vector pointing toward sun, shape (3,).

    Returns:
        torch.Tensor: Polarization angle in radians [0, pi), shape (H, W).
    """
    # Scattering plane normal: cross(sun_dir, view_dir)
    sun_expanded = sun_direction.expand_as(viewing_directions)
    scattering_normal = torch.cross(sun_expanded, viewing_directions, dim=-1)

    # Handle degenerate case (viewing directly at/away from sun)
    scattering_norm = torch.norm(scattering_normal, dim=-1, keepdim=True)
    scattering_normal = torch.where(
        scattering_norm > 1e-6,
        scattering_normal / scattering_norm,
        torch.zeros_like(scattering_normal),
    )

    # E-vector is perpendicular to both viewing direction and scattering normal
    # This places it in the scattering plane, perpendicular to view direction
    e_vector = torch.cross(viewing_directions, scattering_normal, dim=-1)
    e_norm = torch.norm(e_vector, dim=-1, keepdim=True)
    e_vector = torch.where(
        e_norm > 1e-6,
        e_vector / e_norm,
        torch.zeros_like(e_vector),
    )

    # Reference "up" vector for measuring angle (world Z axis)
    up = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Project up onto the plane perpendicular to viewing direction
    view_norm = viewing_directions / (
        torch.norm(viewing_directions, dim=-1, keepdim=True) + 1e-8
    )
    up_proj = up - torch.sum(up * view_norm, dim=-1, keepdim=True) * view_norm
    up_proj_norm = torch.norm(up_proj, dim=-1, keepdim=True)
    up_proj = torch.where(
        up_proj_norm > 1e-6,
        up_proj / up_proj_norm,
        torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(up_proj),
    )

    # Angle between E-vector and projected up reference
    cos_angle = torch.sum(e_vector * up_proj, dim=-1)
    cross_prod = torch.cross(up_proj, e_vector, dim=-1)
    sin_angle = torch.sum(cross_prod * view_norm, dim=-1)

    aop = torch.atan2(sin_angle, cos_angle)

    # Map to [0, pi) since polarization angle has 180-degree symmetry
    return aop % torch.pi


def calculate_stokes_vector(
    intensity: torch.Tensor,
    degree_of_polarization: torch.Tensor,
    polarization_angle: torch.Tensor,
) -> torch.Tensor:
    """Calculate Stokes vector [I, Q, U] from polarization parameters.

    For linearly polarized light from Rayleigh scattering:
    - I = total intensity
    - Q = I * DoP * cos(2*AoP)
    - U = I * DoP * sin(2*AoP)

    Note: V (circular polarization) is omitted since Rayleigh scattering
    produces only linear polarization (V=0). This makes the output
    compatible with 3-channel RGB observations.

    Args:
        intensity: Light intensity per pixel, shape (H, W).
        degree_of_polarization: DoP values, shape (H, W).
        polarization_angle: AoP values in radians, shape (H, W).

    Returns:
        torch.Tensor: Stokes vector with shape (H, W, 3).
    """
    I = intensity
    Q = I * degree_of_polarization * torch.cos(2 * polarization_angle)
    U = I * degree_of_polarization * torch.sin(2 * polarization_angle)

    return torch.stack([I, Q, U], dim=-1)


class MjCambrianPolarizationEye(MjCambrianEye):
    """Eye that computes sky polarization using the Rayleigh model.

    This eye extends the base eye to provide Stokes vector observations
    calculated from the viewing direction and sun position. Polarization
    is only computed for sky pixels (detected via depth threshold).

    The output is a (H, W, 3) tensor containing [I, Q, U] Stokes parameters.
    V (circular polarization) is omitted since Rayleigh scattering produces
    only linear polarization. This makes the output compatible with RGB eyes.
    """

    def __init__(self, config: MjCambrianPolarizationEyeConfig, name: str):
        super().__init__(config, name)
        self._config: MjCambrianPolarizationEyeConfig

        # Ensure depth rendering is enabled for sky detection
        self._renders_depth = "depth_array" in self._config.renderer.render_modes
        if not self._renders_depth:
            get_logger().warning(
                "Polarization eye requires depth_array for sky detection. "
                "Adding depth_array to render modes.",
                extra={"once": True},
            )

        # Cache for viewing directions (recomputed each step as camera moves)
        self._viewing_directions: torch.Tensor = None
        self._sun_direction: torch.Tensor = None

        # Cache raw RGB for human viewer overlay (set during step)
        self._raw_rgb: torch.Tensor = None
        self._dop: torch.Tensor = None
        self._aop: torch.Tensor = None
        self._sky_mask: torch.Tensor = None

    def reset(self, spec: MjCambrianSpec) -> ObsType:
        """Reset the eye and initialize for polarization calculation."""
        # Call parent reset to set up renderer and camera
        # We don't use the parent's observation directly
        self._spec = spec

        if self._renderer is not None:
            resolution = [self._renderer.config.width, self._renderer.config.height]
            self._renderer.reset(spec, *resolution)

            self._fixedcamid = spec.get_camera_id(self._name)
            assert self._fixedcamid != -1, f"Camera '{self._name}' not found."

            import mujoco as mj

            self._renderer.viewer.camera.type = mj.mjtCamera.mjCAMERA_FIXED
            self._renderer.viewer.camera.fixedcamid = self._fixedcamid

        # Initialize observation buffer with correct shape for Stokes vectors
        self._prev_obs_shape = self.observation_space.shape
        self._prev_obs = torch.zeros(
            self._prev_obs_shape,
            dtype=torch.float32,
            device=device,
        )

        return self.step()

    def step(self, obs: ObsType = None) -> ObsType:
        """Render and compute polarization Stokes vectors.

        Args:
            obs: Optional pre-computed observation (not used for polarization).

        Returns:
            torch.Tensor: Stokes vectors with shape (H, W, 4).
        """
        if obs is not None:
            # If observation is provided, use it directly
            return self._update_obs(obs)

        assert self._renderer is not None, "Cannot step without a renderer."

        # Render RGB and depth
        render_output = self._renderer.render()
        if isinstance(render_output, (tuple, list)) and len(render_output) == 2:
            rgb, depth = render_output
            self._raw_rgb = rgb
        else:
            # Only RGB was rendered, can't do sky detection
            get_logger().warning(
                "Polarization eye requires both RGB and depth. "
                "Returning zeros.",
                extra={"once": True},
            )
            return self._update_obs(self._prev_obs)

        # Get camera orientation for viewing direction calculation
        cam_xmat = self._spec.data.cam_xmat[self._fixedcamid].reshape(3, 3)

        # Compute viewing directions for all pixels
        self._viewing_directions = get_viewing_directions(
            cam_xmat,
            self._config.resolution,
            self._config.fov,
        )

        # Get sun direction from MuJoCo light
        self._sun_direction = get_sun_direction_from_light(
            self._spec, self._config.sun_light_name
        )

        # Create sky mask using viewing direction (depth wasn't working reliably)
        # Pixels looking upward (positive Z component in world frame) are sky
        sky_mask = self._viewing_directions[..., 2] > 0

        # Calculate polarization parameters
        scattering_angle = calculate_scattering_angle(
            self._viewing_directions, self._sun_direction
        )

        dop = calculate_degree_of_polarization(
            scattering_angle, self._config.max_polarization
        )

        aop = calculate_polarization_angle(
            self._viewing_directions, self._sun_direction
        )

        # Cache for visualization
        self._dop = dop
        self._aop = aop
        self._sky_mask = sky_mask

        # Use RGB mean as intensity for Stokes I parameter
        intensity = rgb.mean(dim=-1)

        # Calculate Stokes vectors
        stokes = calculate_stokes_vector(intensity, dop, aop)

        # Apply sky mask - zero out non-sky pixels
        sky_mask_expanded = sky_mask.unsqueeze(-1).expand_as(stokes)
        stokes = torch.where(sky_mask_expanded, stokes, torch.zeros_like(stokes))

        # Normalize Stokes vector values to [0, 1] range for observation space
        # I is already in [0, 1] from RGB mean
        # Q, U are in [-1, 1], map to [0, 1]
        stokes[..., 1] = (stokes[..., 1] + 1) / 2  # Q
        stokes[..., 2] = (stokes[..., 2] + 1) / 2  # U

        return self._update_obs(stokes)

    def render(self):
        """Show RGB | DoP | AoP panels in the human viewer."""
        if self._raw_rgb is None:
            return super().render()
        from cambrian.renderer.overlays import MjCambrianCursor, MjCambrianViewerOverlay

        rgb_img = self._raw_rgb * 255.0  # (H, W, 3), float32

        if self._dop is not None and self._aop is not None and self._sky_mask is not None:
            dop_img = self._colorize_dop(self._dop, self._sky_mask)
            aop_img = self._colorize_aop(self._aop, self._sky_mask)
            composite = torch.cat(
                [rgb_img, dop_img.to(rgb_img.device), aop_img.to(rgb_img.device)],
                dim=1,
            )
        else:
            composite = rgb_img

        cursor = MjCambrianCursor(
            position=MjCambrianCursor.Position.BOTTOM_LEFT, x=0, y=0,
            layer=MjCambrianCursor.Layer.BACK,
        )
        return [MjCambrianViewerOverlay.create_image_overlay(composite, cursor=cursor)]

    def _colorize_dop(self, dop: torch.Tensor, sky_mask: torch.Tensor) -> torch.Tensor:
        """Convert DoP (H, W) to viridis RGB (H, W, 3) in [0, 255], matching heatmap style."""
        import matplotlib.cm as cm
        dop_np = (dop * sky_mask).cpu().numpy()
        normalized = np.clip(dop_np / self._config.max_polarization, 0.0, 1.0)
        rgb = cm.viridis(normalized)[..., :3]  # (H, W, 3), float [0, 1]
        rgb[~sky_mask.cpu().numpy()] = 0.0  # black for non-sky pixels
        return torch.from_numpy((rgb * 255).astype(np.uint8)).float()

    def _colorize_aop(self, aop: torch.Tensor, sky_mask: torch.Tensor) -> torch.Tensor:
        """Convert AoP (H, W) in [0, π) to HSV hue-wheel RGB (H, W, 3) in [0, 255].

        Matches the heatmap: full saturation, AoP mapped to hue.
        Non-sky pixels are black.
        """
        from matplotlib.colors import hsv_to_rgb
        aop_np = aop.cpu().numpy()
        mask_np = sky_mask.cpu().numpy()
        # AoP is in [0, π) — map to hue [0, 1] (full cycle over 180° matches polarization symmetry)
        hsv = np.zeros((*aop_np.shape, 3))
        hsv[..., 0] = aop_np / np.pi  # Hue: [0, 1)
        hsv[..., 1] = 1.0             # Full saturation
        hsv[..., 2] = mask_np.astype(float)  # Value: 0 for non-sky, 1 for sky
        rgb = hsv_to_rgb(hsv)  # (H, W, 3), float [0, 1]
        return torch.from_numpy((rgb * 255).astype(np.uint8)).float()

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space for Stokes vectors (H, W, 3) - [I, Q, U]."""
        H, W = self._config.resolution
        return spaces.Box(0.0, 1.0, shape=(H, W, 3), dtype=np.float32)
