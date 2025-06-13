import jax
import ledgetter.utils.vector_tools as vector_tools


def lambertian_renderer(light_direction, light_intensity, normals, points, rho):
    """
    Computes the Lambertian model

    Args:
        light_direction (Array ..., l, 3): Directions of l light sources.
        light_intensity (Array ..., l, c): Intensities of l light sources in c color channels.
        normals (Array ..., 3): Surface normal vectors at each point.
        points (Array ..., 3): 3D points in space (unused in the computation).
        rho (Array ..., c): Surface reflectance (albedo) in c color channels.

    Returns:
        Array ..., c, l: Rendered color values per color channel and light source.
    """
    render =  jax.nn.relu(jax.numpy.einsum('...li, ...lc, ...i, ...c -> ...cl', light_direction, light_intensity, normals, rho))
    return render

def phong_renderer(light_direction, light_intensity, normals, points, rho_spec, tau_spec, epsilon = 1e-6):
    """
    Computes the specular reflectance using a Phong-like model given light direction, intensity, 
    surface normals, camera position, and specular properties.

    Args:
        light_direction (Array ..., l, 3): Directions of l light sources.
        light_intensity (Array ..., l, c): Intensities of l light sources in c color channels.
        normals (Array ..., 3): Surface normal vectors at each point.
        points (Array ..., 3): 3D points in space, used to compute the camera direction.
        rho_spec (Array ...): Specular reflectance.
        tau_spec (float): Shininess coefficient controlling the specular highlight size.
        epsilon (float, optional): Small value to prevent numerical instability. Defaults to 1e-6.

    Returns:
        Array ..., c, l: Rendered specular reflectance per color channel and light source.
    """
    camera_direction = - vector_tools.norm_vector(points, epsilon=epsilon)[1]
    light_dot = 2 * jax.nn.relu(jax.numpy.einsum('...i, ...li -> ...l',normals, light_direction))
    total_reflexion = jax.numpy.einsum('...l, ...i -> ...li', light_dot, normals) - light_direction
    specular_angular_factor = jax.nn.relu(jax.numpy.einsum('...li, ...i -> ...l', total_reflexion, camera_direction))
    reflected_energy = jax.numpy.power(specular_angular_factor + epsilon, tau_spec)
    render = jax.numpy.einsum('...l, ...lc, ...-> ...cl', reflected_energy, light_intensity, rho_spec)
    return render
