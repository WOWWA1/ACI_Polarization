"""Procedural skybox generation based on MuJoCo light sources.

This module provides utilities to generate simple skybox textures where
the sun position matches a MuJoCo directional light source.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import mujoco


def generate_sun_skybox(
    model,
    light_name: str,
    texture_name: str = "skybox",
    sun_radius: int = 15,
    sky_color: tuple = (128, 178, 255),
    ground_color: tuple = (100, 150, 100),
    sun_color: tuple = (255, 255, 255),
    save_path: Optional[Union[str, Path]] = None,
    renderer=None,
    debug_mode: bool = False,
):
    """Generate a skybox texture with sun position matching a MuJoCo light.

    Creates a simple procedural skybox with:
    - Blue sky (top half)
    - Green ground (bottom half)
    - White circular sun patch at the light's azimuth position

    Args:
        model: MuJoCo model (mjModel)
        light_name: Name of the directional light to use as sun
        texture_name: Name of the skybox texture in the model to update
        sun_radius: Radius of the sun circle in pixels
        sky_color: RGB tuple for sky color (0-255)
        ground_color: RGB tuple for ground color (0-255)
        sun_color: RGB tuple for sun color (0-255)
        save_path: Optional path to save the generated texture as PNG
        renderer: Optional renderer with viewer to upload texture to GPU
        debug_mode: If True, use distinct colors for each face to debug mapping

    Returns:
        bool: True if successful, False if light or texture not found
    """
    # Find the light
    light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
    if light_id == -1:
        return False

    # Find the texture
    tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name)
    if tex_id == -1:
        return False

    # Get light direction (points FROM light TO scene)
    # So sun appears in opposite direction
    light_dir = model.light_dir[light_id].copy()

    # Calculate sun direction (opposite of light direction)
    sun_dir = -light_dir

    # Calculate azimuth and elevation
    sun_azimuth = np.arctan2(sun_dir[1], sun_dir[0])
    sun_elevation = np.arctan2(sun_dir[2], np.sqrt(sun_dir[0]**2 + sun_dir[1]**2))

    # Get texture dimensions
    tex_width = model.tex_width[tex_id]
    # Skybox textures are always 6 stacked square faces internally.
    # model.tex_height can return unexpected values in some MuJoCo versions.
    tex_height = 6 * tex_width
    tex_adr = model.tex_adr[tex_id]

    # Get actual number of channels from texture size
    tex_size = tex_width * tex_height
    actual_tex_size = 0
    if tex_id < len(model.tex_adr) - 1:
        actual_tex_size = model.tex_adr[tex_id + 1] - model.tex_adr[tex_id]
    else:
        actual_tex_size = len(model.tex_data) - model.tex_adr[tex_id]

    tex_nchannel = actual_tex_size // tex_size if tex_size > 0 else 3

    # Generate the skybox image
    image = _create_skybox_image(
        width=tex_width,
        height=tex_height,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        sun_radius=sun_radius,
        sky_color=sky_color,
        ground_color=ground_color,
        sun_color=sun_color,
        debug_mode=debug_mode,
    )

    # Convert to RGBA if needed
    if tex_nchannel == 4:
        rgba_image = np.zeros((tex_height, tex_width, 4), dtype=np.uint8)
        rgba_image[..., :3] = image
        rgba_image[..., 3] = 255  # Full opacity
        image = rgba_image

    # Save texture if path provided (convert to cross format for readability)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        # Convert 6-stacked format to cross format for easier visualization
        cross_image = _convert_stacked_to_cross(image, tex_width)
        img = Image.fromarray(cross_image, mode="RGB")
        img.save(save_path)

    # Write to model texture data with bounds checking
    tex_data_size = tex_width * tex_height * tex_nchannel
    flat_image = image.flatten()

    # Safety check: ensure we don't write past the end of tex_data
    max_write = min(tex_data_size, len(model.tex_data) - tex_adr, len(flat_image))
    if max_write > 0 and tex_adr >= 0 and tex_adr < len(model.tex_data):
        model.tex_data[tex_adr : tex_adr + max_write] = flat_image[:max_write]

    return True


def _create_skybox_image(
    width: int,
    height: int,
    sun_azimuth: float,
    sun_elevation: float,
    sun_radius: int,
    sky_color: tuple,
    ground_color: tuple,
    sun_color: tuple,
    debug_mode: bool = False,
) -> np.ndarray:
    """Create a skybox as 6 stacked square faces (MuJoCo internal format).

    MuJoCo stores skybox textures as 6 vertically stacked square faces.
    Face order (indices 0-5, top to bottom):
        0: +X (right)
        1: -X (left)
        2: +Y (front)
        3: -Y (back)
        4: +Z (up/sky)
        5: -Z (down/ground)

    Args:
        width: Face width in pixels
        height: Total height (should be 6 * width)
        sun_azimuth: Sun azimuth angle in radians (0 = +X, pi/2 = +Y)
        sun_elevation: Sun elevation angle in radians (0 = horizon, pi/2 = overhead)
        sun_radius: Radius of sun circle in pixels
        sky_color: RGB tuple for sky
        ground_color: RGB tuple for ground
        sun_color: RGB tuple for sun
        debug_mode: If True, use distinct colors for each face

    Returns:
        np.ndarray: RGB image with shape (height, width, 3), dtype uint8
    """
    face_size = width
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Face indices - need to figure out correct mapping
    # Using debug colors to identify which index maps to which direction
    FACE_0 = 0
    FACE_1 = 1
    FACE_2 = 2
    FACE_3 = 3
    FACE_4 = 4
    FACE_5 = 5

    if debug_mode:
        # Debug: distinct colors to identify faces
        debug_colors = [
            (255, 0, 0),      # 0: Red
            (0, 255, 0),      # 1: Green
            (0, 0, 255),      # 2: Blue
            (255, 255, 0),    # 3: Yellow
            (0, 255, 255),    # 4: Cyan
            (255, 0, 255),    # 5: Magenta
        ]
        for i in range(6):
            y_start = i * face_size
            y_end = (i + 1) * face_size
            image[y_start:y_end, :] = debug_colors[i]
        return image

    # MuJoCo cubemap face ordering (verified: X and Y are swapped vs labels)
    FACE_BACK = 0    # -X
    FACE_FRONT = 1   # +X
    FACE_UP = 2      # +Z (sky)
    FACE_DOWN = 3    # -Z (ground)
    FACE_LEFT = 4    # -Y
    FACE_RIGHT = 5   # +Y

    # Fill each face
    for i in range(6):
        y_start = i * face_size
        y_end = (i + 1) * face_size

        if i == FACE_UP:
            # Up face - all sky
            image[y_start:y_end, :] = sky_color
        elif i == FACE_DOWN:
            # Down face - all ground
            image[y_start:y_end, :] = ground_color
        else:
            # Side faces - sky on top half, ground on bottom half
            mid_y = y_start + face_size // 2
            image[y_start:mid_y, :] = sky_color
            image[mid_y:y_end, :] = ground_color

    # Determine which face the sun appears on based on azimuth and elevation
    sun_azimuth_normalized = sun_azimuth % (2 * np.pi)

    # If sun is high in sky (elevation > 45 degrees), place on Up face
    if sun_elevation > np.pi / 4:
        sun_face = FACE_UP
        # Map azimuth to position on Up face
        # Center of face is directly overhead, edges are toward horizons
        # Use azimuth to determine direction from center
        distance_from_center = 1.0 - (sun_elevation - np.pi / 4) / (np.pi / 4)  # 0 at zenith, 1 at 45°
        sun_x = int(face_size / 2 + distance_from_center * (face_size / 2) * np.cos(sun_azimuth_normalized))
        sun_y = FACE_UP * face_size + int(face_size / 2 - distance_from_center * (face_size / 2) * np.sin(sun_azimuth_normalized))
    else:
        # Sun is on a side face
        # Map azimuth to face
        # Azimuth 0 = +X (FACE_FRONT), pi/2 = +Y (FACE_RIGHT), pi = -X (FACE_BACK), 3pi/2 = -Y (FACE_LEFT)
        if sun_azimuth_normalized < np.pi / 4 or sun_azimuth_normalized >= 7 * np.pi / 4:
            sun_face = FACE_FRONT
            if sun_azimuth_normalized >= 7 * np.pi / 4:
                local_x = (sun_azimuth_normalized - 7 * np.pi / 4) / (np.pi / 2)
            else:
                local_x = (sun_azimuth_normalized + np.pi / 4) / (np.pi / 2)
        elif sun_azimuth_normalized < 3 * np.pi / 4:
            sun_face = FACE_RIGHT
            local_x = (sun_azimuth_normalized - np.pi / 4) / (np.pi / 2)
        elif sun_azimuth_normalized < 5 * np.pi / 4:
            sun_face = FACE_BACK
            local_x = (sun_azimuth_normalized - 3 * np.pi / 4) / (np.pi / 2)
        else:
            sun_face = FACE_LEFT
            local_x = (sun_azimuth_normalized - 5 * np.pi / 4) / (np.pi / 2)

        # Vertical position based on elevation (0 = horizon, 45° = top of side face)
        # Map elevation [0, pi/4] to y position [face_size/2, 0] (horizon to top)
        local_y = 0.5 - (sun_elevation / (np.pi / 4)) * 0.5  # 0.5 at horizon, 0 at top

        y_start = sun_face * face_size
        sun_x = int(local_x * face_size)
        sun_y = y_start + int(local_y * face_size)

    # Clamp sun position to stay within face bounds (prevent clipping at edges)
    # For side faces, the sun should stay within the sky portion (top half)
    margin = sun_radius + 2  # Add small buffer beyond radius

    # Determine which face the sun is on and clamp accordingly
    face_idx = sun_y // face_size
    face_y_start = face_idx * face_size
    face_y_end = (face_idx + 1) * face_size

    # Clamp X to stay within face width
    sun_x = max(margin, min(face_size - margin, sun_x))

    # Clamp Y to stay within face (and for side faces, within sky portion)
    if face_idx == FACE_UP:
        # Up face - sun can be anywhere
        sun_y = max(face_y_start + margin, min(face_y_end - margin, sun_y))
    elif face_idx in [FACE_RIGHT, FACE_FRONT, FACE_LEFT, FACE_BACK]:
        # Side faces - sun should stay in sky portion (top half of face)
        sky_y_end = face_y_start + face_size // 2
        sun_y = max(face_y_start + margin, min(sky_y_end - margin, sun_y))
    else:
        # Ground face - shouldn't have sun, but clamp anyway
        sun_y = max(face_y_start + margin, min(face_y_end - margin, sun_y))

    _draw_circle(image, sun_x, sun_y, sun_radius, sun_color, wrap_width=None)

    return image


def _convert_stacked_to_cross(stacked_image: np.ndarray, face_size: int) -> np.ndarray:
    """Convert 6-stacked faces to cross format for visualization.

    MuJoCo's internal cubemap face ordering (determined by testing):
        [0] = -Y (back)
        [1] = +Y (front)
        [2] = +Z (up/sky)
        [3] = -Z (down/ground)
        [4] = -X (left)
        [5] = +X (right)

    Output cross format:
             [+Z up]
        [-X] [+Y front] [+X] [-Y back]
             [-Z down]
    """
    # Cross format: 4 cols x 3 rows
    cross_width = face_size * 4
    cross_height = face_size * 3
    cross_image = np.ones((cross_height, cross_width, 3), dtype=np.uint8) * 255  # White background

    # Extract faces from stacked format
    faces = []
    for i in range(6):
        y_start = i * face_size
        y_end = (i + 1) * face_size
        faces.append(stacked_image[y_start:y_end, :, :3])  # Take only RGB

    # Place faces in cross layout using MuJoCo's face ordering:
    # Row 0: . U . .  (U = +Z = face 2)
    # Row 1: L F R B  (L=-X=4, F=+Y=1, R=+X=5, B=-Y=0)
    # Row 2: . D . .  (D = -Z = face 3)

    # Up face (+Z = face 2)
    cross_image[0:face_size, face_size:2*face_size] = faces[2]

    # Left face (-X = face 4)
    cross_image[face_size:2*face_size, 0:face_size] = faces[4]

    # Front face (+Y = face 1)
    cross_image[face_size:2*face_size, face_size:2*face_size] = faces[1]

    # Right face (+X = face 5)
    cross_image[face_size:2*face_size, 2*face_size:3*face_size] = faces[5]

    # Back face (-Y = face 0)
    cross_image[face_size:2*face_size, 3*face_size:4*face_size] = faces[0]

    # Down face (-Z = face 3)
    cross_image[2*face_size:3*face_size, face_size:2*face_size] = faces[3]

    return cross_image


def generate_unique_color_skybox(
    model,
    texture_name: str = "skybox",
    save_path: Optional[Union[str, Path]] = None,
):
    """Generate a skybox where each cross-map pixel has a unique RGB color.

    Encodes the cross-map pixel index as RGB:
        R = (index >> 16) & 0xFF
        G = (index >> 8) & 0xFF
        B = index & 0xFF
    where index = cross_y * cross_width + cross_x.

    This lets you verify the ray→cubemap mapping: the rendered camera color
    should match the cross-map pixel at the coordinates predicted by
    ray_to_cross_map_pixel().

    MuJoCo face ordering in stacked format:
        0=-Y, 1=+Y, 2=+Z, 3=-Z, 4=-X, 5=+X

    Cross layout (y=0 at top):
             [+Z row0]
        [-X] [+Y] [+X] [-Y]  row1
             [-Z row2]

    Args:
        model: MuJoCo model (mjModel).
        texture_name: Name of the skybox texture to overwrite.
        save_path: If given, save the cross-map image as a PNG here.

    Returns:
        tuple: (success: bool, cross_image: np.ndarray or None)
    """
    tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name)
    if tex_id == -1:
        return False, None

    face_size = model.tex_width[tex_id]
    cross_width = 4 * face_size
    cross_height = 3 * face_size

    # face index → (col, row) in cross layout
    face_col_row = {0: (3, 1), 1: (1, 1), 2: (1, 0), 3: (1, 2), 4: (0, 1), 5: (2, 1)}

    # Build stacked image (6*face_size, face_size, 3)
    stacked = np.zeros((6 * face_size, face_size, 3), dtype=np.uint8)

    for face_idx in range(6):
        col, row = face_col_row[face_idx]
        vv, uu = np.mgrid[0:face_size, 0:face_size]  # (face_size, face_size) each

        cross_x = col * face_size + uu
        cross_y = row * face_size + vv
        pixel_index = cross_y.astype(np.int64) * cross_width + cross_x.astype(np.int64)

        y_start = face_idx * face_size
        stacked[y_start:y_start + face_size, :, 0] = ((pixel_index >> 16) & 0xFF).astype(np.uint8)
        stacked[y_start:y_start + face_size, :, 1] = ((pixel_index >> 8) & 0xFF).astype(np.uint8)
        stacked[y_start:y_start + face_size, :, 2] = (pixel_index & 0xFF).astype(np.uint8)

    # Compute number of channels from texture data size
    tex_adr = model.tex_adr[tex_id]
    tex_size = face_size * (6 * face_size)
    if tex_id < len(model.tex_adr) - 1:
        actual_tex_size = model.tex_adr[tex_id + 1] - model.tex_adr[tex_id]
    else:
        actual_tex_size = len(model.tex_data) - model.tex_adr[tex_id]
    tex_nchannel = actual_tex_size // tex_size if tex_size > 0 else 3

    image = stacked
    if tex_nchannel == 4:
        rgba = np.zeros((6 * face_size, face_size, 4), dtype=np.uint8)
        rgba[..., :3] = stacked
        rgba[..., 3] = 255
        image = rgba

    flat = image.flatten()
    max_write = min(len(flat), len(model.tex_data) - tex_adr)
    if max_write > 0:
        model.tex_data[tex_adr:tex_adr + max_write] = flat[:max_write]

    # Build cross-map image for saving / reference
    cross_image = _convert_stacked_to_cross(stacked, face_size)

    if save_path is not None:
        from PIL import Image
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cross_image, mode="RGB").save(save_path)

    return True, cross_image


def ray_to_cross_map_pixel(
    ray_dir: np.ndarray,
    face_size: int,
) -> tuple:
    """Convert a 3D ray direction to (x, y) pixel coordinates in the cross-map layout.

    MuJoCo cubemap face layout (verified empirically — X and Y are swapped
    relative to standard OpenGL due to MuJoCo's coordinate convention):

        Major axis  Face        sc          tc
        +x          1 (+X)    +y/|x|     -z/|x|
        -x          0 (-X)    -y/|x|     -z/|x|
        +y          5 (+Y)    -x/|y|     -z/|y|
        -y          4 (-Y)    +x/|y|     -z/|y|
        +z          2 (+Z)    +x/|z|     -y/|z|
        -z          3 (-Z)    -x/|z|     -y/|z|

    sc, tc in [-1, 1] map to u, v in [0, face_size-1].
    v=0 is top of face in image (numpy convention).

    Cross-map layout (face_size px per cell, y=0 at top):
             [+Z]
        [-Y] [+X] [+Y] [-X]
             [-Z]

    MuJoCo face indices: 0=-X, 1=+X, 2=+Z, 3=-Z, 4=-Y, 5=+Y

    Args:
        ray_dir: Direction vector (3,) in world coordinates (x, y, z). Need not be unit.
        face_size: Size of each cubemap face in pixels.

    Returns:
        (cross_x, cross_y): Pixel coordinates in the cross-map image.
    """
    rx, ry, rz = float(ray_dir[0]), float(ray_dir[1]), float(ray_dir[2])
    ax, ay, az = abs(rx), abs(ry), abs(rz)

    if ax >= ay and ax >= az:
        ma = ax
        if rx > 0:
            face, sc, tc = 1, +ry / ma, -rz / ma   # +X → face 1
        else:
            face, sc, tc = 0, -ry / ma, -rz / ma   # -X → face 0
    elif ay >= ax and ay >= az:
        ma = ay
        if ry > 0:
            face, sc, tc = 5, -rx / ma, -rz / ma   # +Y → face 5
        else:
            face, sc, tc = 4, +rx / ma, -rz / ma   # -Y → face 4
    else:
        ma = az
        if rz > 0:
            face, sc, tc = 2, +rx / ma, -ry / ma   # +Z → face 2
        else:
            face, sc, tc = 3, -rx / ma, -ry / ma   # -Z → face 3

    # Map sc, tc from [-1, 1] to [0, face_size-1]
    u = int(np.clip((sc + 1) / 2 * face_size, 0, face_size - 1))
    v = int(np.clip((tc + 1) / 2 * face_size, 0, face_size - 1))

    # Cross-map position: (col, row) for each face index
    face_to_col_row = {
        2: (1, 0),  # +Z: row 0, col 1
        4: (0, 1),  # -X: row 1, col 0
        1: (1, 1),  # +Y: row 1, col 1
        5: (2, 1),  # +X: row 1, col 2
        0: (3, 1),  # -Y: row 1, col 3
        3: (1, 2),  # -Z: row 2, col 1
    }
    col, row = face_to_col_row[face]

    cross_x = col * face_size + u
    cross_y = row * face_size + v

    return cross_x, cross_y


def _draw_circle(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    color: tuple,
    wrap_width: int = None,
):
    """Draw a filled circle on the image.

    Args:
        image: Image array to draw on (modified in place)
        cx, cy: Circle center coordinates
        radius: Circle radius in pixels
        color: RGB tuple
        wrap_width: If set, wrap horizontally (for panoramic skybox)
    """
    height, width = image.shape[:2]

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                y = cy + dy
                x = cx + dx

                # Wrap horizontally if needed (skybox wraps around)
                if wrap_width:
                    x = x % wrap_width

                # Check bounds
                if 0 <= y < height and 0 <= x < width:
                    image[y, x] = color
