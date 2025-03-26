from dataclasses import dataclass

import glm

from .camera import Camera


@dataclass
class Ray:
    origin: glm.vec3
    direction: glm.vec3


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
    near_point_clip = glm.vec4([ndc_x, ndc_y, -1, 1])
    far_point_clip = glm.vec4([ndc_x, ndc_y, 1, 1])

    # Get inverse matrices
    inv_proj: glm.mat4 = glm.inverse(camera.get_proj_matrix())
    inv_view: glm.mat4 = glm.inverse(camera.get_view_matrix())

    # Unproject to view space
    near_view = inv_proj * near_point_clip
    far_view = inv_proj * far_point_clip

    # Transform to world space
    near_world = inv_view * near_view
    far_world = inv_view * far_view

    # Create ray
    ray_origin = near_world.xyz / near_world.w
    ray_direction = glm.normalize(far_world.xyz / far_world.w - ray_origin)

    return Ray(origin=ray_origin, direction=ray_direction)


def intersect_ray_plane(
    ray: Ray,
    plane_y: float
) -> glm.vec3 | None:
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

    return ray.origin + t * ray.direction
