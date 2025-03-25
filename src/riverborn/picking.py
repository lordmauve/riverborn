import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass

from pyrr import Vector4, Vector3

from .camera import Camera


@dataclass
class Ray:
    origin: Vector3
    direction: Vector3


def get_mouse_ray(
    camera: Camera,
    mouse_x: float,
    mouse_y: float,
    screen_width: int,
    screen_height: int
) -> Ray:
    """
    Compute a view ray from the mouse screen position.

    Returns:
        (ray_origin, ray_direction) in world space.
    """
    # Convert mouse position to Normalized Device Coordinates (NDC)
    ndc_x: float = (mouse_x / screen_width) * 2 - 1
    ndc_y: float = 1 - (mouse_y / screen_height) * 2  # Invert y if needed

    # Points in clip space
    near_point_clip: Vector4 = Vector4([ndc_x, ndc_y, -1, 1], dtype=np.float32)
    far_point_clip: Vector4 = Vector4([ndc_x, ndc_y, 1, 1], dtype=np.float32)

    # Get inverse matrices
    inv_proj: Matrix44 = camera.get_proj_matrix().inverse
    inv_view: Matrix44 = camera.get_view_matrix().inverse

    # Unproject to view space
    near_view: Vector4 = inv_proj * near_point_clip
    far_view: Vector4 = inv_proj * far_point_clip
    near_view /= near_view[3]
    far_view /= far_view[3]

    # Transform to world space
    near_world: Vector4 = inv_view * near_view
    far_world: Vector4 = inv_view * far_view
    near_world /= near_world[3]
    far_world /= far_world[3]

    # Create ray
    ray_origin: Vector4 = near_world.vector3[0]
    ray_direction: Vector4 = far_world.vector3[0] - ray_origin
    ray_direction.normalize()

    return Ray(origin=ray_origin, direction=ray_direction)


def intersect_ray_plane(
    ray: Ray,
    plane_y: float
) -> NDArray[np.float32] | None:
    """
    Intersect the ray with the plane y = plane_y.

    Returns:
        The intersection point as a 3D vector, or None if no intersection.
    """
    dy: float = ray.direction[1]
    if dy == 0:
        return None  # Parallel to the plane

    t: float = (plane_y - ray.origin[1]) / dy
    if t < 0:
        return None  # Intersection behind the ray origin

    intersection: NDArray[np.float32] = ray.origin + t * ray.direction
    return intersection
