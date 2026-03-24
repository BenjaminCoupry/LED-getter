import jax
import ledgetter.utils.vector_tools as vector_tools

#TODO utiliser filters
def lambertian_brdf(light_direction, light_intensity, normals, points, rho):
    """
    Computes the Lambertian model

    Args:
        light_direction (Array ..., l, 3): Directions of l light sources.
        light_intensity (Array ..., l, f): Intensities of l light sources in c color filter.
        normals (Array ..., 3): Surface normal vectors at each point.
        points (Array ..., 3): 3D points in space (unused in the computation).
        rho (Array ..., f): Surface reflectance (albedo) in c color filter.
        filters (Array ..., c)

    Returns:
        Array ..., f, l: Rendered color values per color channel and light source.
    """
    result =  jax.nn.relu(jax.numpy.einsum('...li, ...lf, ...i, ...f -> ...fl', light_direction, light_intensity, normals, rho))
    return result

def phong_brdf(light_direction, light_intensity, normals, points, rho_spec, tau_spec, epsilon = 1e-6):
    """
    Computes the specular reflectance using a Phong-like model given light direction, intensity, 
    surface normals, camera position, and specular properties.

    Args:
        light_direction (Array ..., l, 3): Directions of l light sources.
        light_intensity (Array ..., l, f): Intensities of l light sources in f color filter.
        normals (Array ..., 3): Surface normal vectors at each point.
        points (Array ..., 3): 3D points in space, used to compute the camera direction.
        rho_spec (Array ...): Specular reflectance.
        tau_spec (float): Shininess coefficient controlling the specular highlight size.
        epsilon (float, optional): Small value to prevent numerical instability. Defaults to 1e-6.

    Returns:
        Array ..., f, l: Rendered specular reflectance per color channel and light source.
    """
    camera_direction = - vector_tools.norm_vector(points, epsilon=epsilon)[1]
    light_dot = 2 * jax.nn.relu(jax.numpy.einsum('...i, ...li -> ...l',normals, light_direction))
    total_reflexion = jax.numpy.einsum('...l, ...i -> ...li', light_dot, normals) - light_direction
    specular_angular_factor = jax.nn.relu(jax.numpy.einsum('...li, ...i -> ...l', total_reflexion, camera_direction))
    reflected_energy = jax.numpy.power(specular_angular_factor + epsilon, tau_spec)
    result = jax.numpy.einsum('...l, ...lf, ...-> ...fl', reflected_energy, light_intensity, rho_spec)
    return result
