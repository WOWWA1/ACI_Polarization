"""Polarization Compass Navigation Environment.

This environment randomizes the sun direction each episode and places the goal
at a position 90° clockwise from the sun direction (projected on ground plane).
This forces the agent to use polarization vision to determine the goal direction.
"""

import numpy as np
import mujoco
from typing import Any, Dict, Optional, Tuple

from cambrian.envs.maze_env import MjCambrianMazeEnv, MjCambrianMazeEnvConfig
from cambrian.utils import get_logger
from cambrian.utils.skybox import generate_sun_skybox


class MjCambrianPolarizationCompassEnv(MjCambrianMazeEnv):
    """Environment where goal direction is determined by sun position.

    On each reset:
    1. A random sun direction is chosen (0°, 90°, 180°, or 270° around Z axis)
    2. The goal is placed 90° clockwise from the sun direction
    3. The agent must use polarization to find the sun, then navigate to goal
    """

    def __init__(self, config: MjCambrianMazeEnvConfig, **kwargs):
        super().__init__(config, **kwargs)

        self._sun_light_id: int = -1
        self._goal_body_id: int = -1
        self._agent_body_id: int = -1
        self._goal_joint_x_id: int = -1
        self._goal_joint_y_id: int = -1
        self._sun_angles = [0, 90, 180, 270]  # Possible sun angles in degrees

        # Sun trajectory for testing - cardinal directions with goal 90° clockwise
        # Elevation: 30° so sun is visible on side faces
        self._sun_trajectory = [
            (0, 30),     # Episode 1: Sun at +X, goal at -Y
            (90, 30),    # Episode 2: Sun at +Y, goal at +X
            (180, 30),   # Episode 3: Sun at -X, goal at +Y
            (270, 30),   # Episode 4: Sun at -Y, goal at -X
        ]
        self._trajectory_index = -1  # Incremented to 0 on first reset
        self._current_sun_angle: float = 0
        self._sun_angle_index: int = 0  # For deterministic cycling during testing

        # Distance from center to place goal
        self._goal_distance = 6.0  # units from origin (visible on screen)

        # Procedural skybox generation
        self._episode_count = 0
        self._save_skybox_textures = False
        self._skybox_save_dir = "skybox_debug"

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment with randomized sun direction."""

        # Call parent reset first
        obs, info = super().reset(seed=seed, options=options)

        # Find the sun light and goal body (re-find each reset in case model changed)
        self._sun_light_id = self._find_light("sun_light")
        if self._sun_light_id == -1:
            get_logger().warning(
                "Sun light 'sun_light' not found. Trying 'custom_light_1'.",
                extra={"once": True},
            )
            self._sun_light_id = self._find_light("custom_light_1")

        self._goal_body_id = self._find_goal_body()
        self._agent_body_id = self._find_agent_body()
        self._goal_joint_x_id = -1  # Reset joint cache
        self._goal_joint_y_id = -1

        # Cycle through sun trajectory (cardinal directions)
        self._trajectory_index = (self._trajectory_index + 1) % len(self._sun_trajectory)
        azimuth_deg, elevation_deg = self._sun_trajectory[self._trajectory_index]
        self._current_sun_angle = azimuth_deg  # For compatibility

        # Only update sun direction if light was found
        if self._sun_light_id != -1:
            self._update_sun_direction(azimuth_deg, elevation_deg)

            # Generate procedural skybox matching sun position
            skybox_success = generate_sun_skybox(
                self.model,
                light_name="sun_light",
                texture_name="skybox",
            )
            print(f"[SKYBOX] Episode sun at {azimuth_deg}°, generation success={skybox_success}")

            # Upload the modified texture to GPU if renderer is available
            if self._renderer is not None and hasattr(self._renderer, '_viewer'):
                viewer = self._renderer._viewer
                if viewer._mjr_context is not None:
                    tex_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "skybox"
                    )
                    if tex_id != -1:
                        mujoco.mjr_uploadTexture(
                            self.model, viewer._mjr_context, tex_id
                        )
                        print(f"[SKYBOX] Uploaded texture to GPU (tex_id={tex_id})")

        # Move goal to 90° clockwise from sun
        goal_angle = self._current_sun_angle - 90  # 90° clockwise
        self._update_goal_position(goal_angle)

        # Step physics to update positions
        mujoco.mj_forward(self.model, self.data)

        return obs, info

    def _find_light(self, name: str) -> int:
        """Find light ID by name."""
        # Try mj_name2id first
        light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, name)
        if light_id != -1:
            return light_id

        # Fallback: iterate through lights
        for i in range(self.model.nlight):
            try:
                light_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_LIGHT, i)
                if light_name and name in light_name:
                    return i
            except:
                pass

        # Last resort: return first non-fill light (index 0 or 1)
        if self.model.nlight > 0:
            return 0
        return -1

    def _find_goal_body(self) -> int:
        """Find the goal body ID."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and "goal" in body_name.lower():
                return i
        return -1

    def _find_agent_body(self) -> int:
        """Find the agent body ID."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and "agent" in body_name.lower() and "body" in body_name.lower():
                return i
        return -1

    def _update_sun_direction(self, azimuth_deg: float, elevation_deg: float = 45.0):
        """Update the sun light direction based on azimuth and elevation.

        Args:
            azimuth_deg: Horizontal angle in degrees. 0° = +X, 90° = +Y, etc.
            elevation_deg: Vertical angle in degrees. 0° = horizon, 90° = overhead.
        """
        if self._sun_light_id == -1:
            return

        azimuth_rad = np.radians(azimuth_deg)
        elevation_rad = np.radians(elevation_deg)

        # Calculate light direction from azimuth and elevation
        # Sun direction (where sun appears): spherical to cartesian
        # Light dir in MuJoCo points FROM light TO scene (opposite of sun direction)
        cos_elev = np.cos(elevation_rad)
        light_dir_x = -np.cos(azimuth_rad) * cos_elev
        light_dir_y = -np.sin(azimuth_rad) * cos_elev
        light_dir_z = -np.sin(elevation_rad)

        # Normalize
        norm = np.sqrt(light_dir_x**2 + light_dir_y**2 + light_dir_z**2)

        # Update light direction in model
        self.model.light_dir[self._sun_light_id] = [
            light_dir_x / norm,
            light_dir_y / norm,
            light_dir_z / norm
        ]

        # Also update light position to be "behind" the sun direction
        # This helps with visualization
        self.model.light_pos[self._sun_light_id] = [
            20 * np.cos(azimuth_rad) * np.cos(elevation_rad),
            20 * np.sin(azimuth_rad) * np.cos(elevation_rad),
            20 * np.sin(elevation_rad)
        ]

    def _cache_goal_joints(self):
        """Find and cache the goal's slide joint IDs."""
        if self._goal_body_id == -1:
            return

        for i in range(self.model.njnt):
            if self.model.jnt_bodyid[i] == self._goal_body_id:
                jnt_type = self.model.jnt_type[i]
                if jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    axis = self.model.jnt_axis[i]
                    if axis[0] > 0.5:  # X axis
                        self._goal_joint_x_id = i
                    elif axis[1] > 0.5:  # Y axis
                        self._goal_joint_y_id = i

    def _update_goal_position(self, angle_deg: float):
        """Move goal to position at given angle from agent.

        Args:
            angle_deg: Angle in degrees. 0° = +X direction, 90° = +Y, etc.
        """
        if self._goal_body_id == -1 or self._agent_body_id == -1:
            return

        # Cache joint IDs if not done yet
        if self._goal_joint_x_id == -1 and self._goal_joint_y_id == -1:
            self._cache_goal_joints()

        angle_rad = np.radians(angle_deg)

        # Get agent's current position
        agent_pos = self.data.xpos[self._agent_body_id].copy()

        # Desired position for goal RELATIVE TO AGENT
        target_x = agent_pos[0] + self._goal_distance * np.cos(angle_rad)
        target_y = agent_pos[1] + self._goal_distance * np.sin(angle_rad)

        # First, reset goal qpos to 0 to find base position
        if self._goal_joint_x_id != -1:
            qpos_adr_x = self.model.jnt_qposadr[self._goal_joint_x_id]
            self.data.qpos[qpos_adr_x] = 0
        if self._goal_joint_y_id != -1:
            qpos_adr_y = self.model.jnt_qposadr[self._goal_joint_y_id]
            self.data.qpos[qpos_adr_y] = 0

        # Compute forward kinematics to get goal's base position
        mujoco.mj_forward(self.model, self.data)
        base_pos = self.data.xpos[self._goal_body_id].copy()

        # Now compute qpos needed to reach target from base
        qpos_x = target_x - base_pos[0]
        qpos_y = target_y - base_pos[1]

        # Update slide joint positions
        if self._goal_joint_x_id != -1:
            self.data.qpos[qpos_adr_x] = qpos_x
        if self._goal_joint_y_id != -1:
            self.data.qpos[qpos_adr_y] = qpos_y

        # Update physics
        mujoco.mj_forward(self.model, self.data)

