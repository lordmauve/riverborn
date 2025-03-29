import moderngl
from pyglm import glm


def cull_instances(ctx: moderngl.Context, instance_buffer: moderngl.Buffer, num_instances: int, view_projection: glm.mat4, sphere_center: glm.mat3, sphere_radius: float) -> tuple[moderngl.Buffer, int]:
    """
    Performs viewport culling on instance matrices using transform feedback.

    Parameters:
      ctx             - The moderngl context.
      instance_buffer - A moderngl.Buffer containing instance matrices (each 4x4, 16 floats).
      num_instances   - The number of instance matrices in the buffer.
      view_projection - A 4x4 matrix (e.g. a numpy array) for the view–projection transform.
      sphere_center   - A 3-element tuple (x, y, z) representing the bounding sphere centre in model space.
      sphere_radius   - The radius of the bounding sphere (in model space).

    Returns:
      - A moderngl.Buffer containing only the instance matrices that passed the culling test.
      -  The number of visible instances.
    """
    import numpy as np
    if num_instances == 0:
        return instance_buffer, 0

    # Create a buffer to capture transform feedback output.
    # Each instance is 16 floats (16 * 4 bytes = 64 bytes)
    tf_buffer = ctx.buffer(reserve=num_instances * 64)

    # Vertex shader: compute the clip-space centre and a single projected radius.
    vertex_shader_source = '''
    #version 330
    in mat4 m_model;

    uniform mat4 view_projection;
    uniform vec3 sphere_center;
    uniform float sphere_radius;

    // Pass data to the geometry shader.
    out mat4 out_model;
    out vec2 out_ndc_center;
    out float out_ndc_radius;

    void main() {
        // Compute world-space centre of the bounding sphere.
        vec4 world_center = m_model * vec4(sphere_center, 1.0);
        // Transform to clip space.
        vec4 clip_center = view_projection * world_center;
        // Convert to Normalised Device Coordinates (NDC).
        vec2 ndc_center = clip_center.xy / clip_center.w;
        // Approximate the projected radius in NDC space.
        float ndc_radius = sphere_radius / abs(clip_center.w);

        out_model = m_model;
        out_ndc_center = ndc_center;
        out_ndc_radius = ndc_radius;
    }
    '''

    # Geometry shader: test if the sphere is entirely offscreen using the projected radius.
    geometry_shader_source = '''
    #version 330
    layout(points) in;
    layout(points, max_vertices = 1) out;

    in mat4 out_model[];
    in vec2 out_ndc_center[];
    in float out_ndc_radius[];

    // This is the variable we capture via transform feedback.
    out mat4 tf_model;

    void main() {
        vec2 center = out_ndc_center[0];
        float radius = out_ndc_radius[0];

        // If the bounding circle is completely off-screen, cull the instance.
        if (center.x + radius < -1.0 || center.x - radius > 1.0 ||
            center.y + radius < -1.0 || center.y - radius > 1.0) {
            return;
        }

        tf_model = out_model[0];
        EmitVertex();
        EndPrimitive();
    }
    '''

    # Create the program with transform feedback varyings (we capture "tf_model").
    prog = ctx.program(
        vertex_shader=vertex_shader_source,
        geometry_shader=geometry_shader_source,
        varyings=['tf_model']
    )

    # Set the uniform values.
    prog['view_projection'].write(view_projection)
    prog['sphere_center'].value = tuple(sphere_center)
    prog['sphere_radius'].value = float(sphere_radius)

    # Create a Vertex Array Object binding our instance buffer.
    # The format '16f' corresponds to 16 floats per instance (a 4x4 matrix).
    vao = ctx.vertex_array(prog, [(instance_buffer, '16f', 'm_model')])

    with ctx.query(primitives=True) as query:
        # Disable rasterisation – we are only interested in transform feedback.
        ctx.enable(moderngl.RASTERIZER_DISCARD)
        # Execute transform feedback; drawing points (one per instance).
        vao.transform(tf_buffer, vertices=num_instances)
        ctx.disable(moderngl.RASTERIZER_DISCARD)
    visible_count = query.primitives

    # The returned buffer now contains the culled instance matrices.
    return tf_buffer, visible_count
