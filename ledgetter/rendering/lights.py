import jax
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.image.lanczos as lanczos
import ledgetter.space.rotations as rotations
import ledgetter.space.coord_systems as coord_systems
import ledgetter.space.spherical_harmonics as spherical_harmonics


def get_directional_light(light_directions, dir_light_power, points):
    """
    Generates directional light sources with given directions and power.

    Args:
        light_directions (Array l, 3): Directions of l light sources.
        dir_light_power (Array l): Power of l light sources.
        points (Array ..., 3): The points in space where the light is to be evaluated.

    Returns:
        Tuple[Array ..., l, 3], Array ..., l, c]: Light directions and corresponding intensities.
    """
    nl, npix = dir_light_power.shape[-1], points.shape[:-1]
    light_local_intensity = jax.numpy.broadcast_to(jax.numpy.expand_dims(dir_light_power, axis=-1), npix+(nl,3))
    light_local_directions = jax.numpy.broadcast_to(light_directions, npix+(nl,3))
    return light_local_directions, light_local_intensity

def get_rail_light(center, light_distance, light_directions, dir_light_power, points):
    """
    Computes the local direction and intensity of light sources along a rail, 
    based on their distance from a given center and their direction.

    Args:
        center (Array ..., 3): The center position of the light source.
        light_distance (Array ..., l): The distance of light sources from the center.
        light_directions (Array ..., l, 3): The directions of l light sources.
        dir_light_power (Array ..., l): The power of l light sources.
        points (Array ..., 3): The points in space where the light is to be evaluated.

    Returns:
        Tuple[Array ..., l, 3], Array ..., l, c]: Local directions and intensities of light sources at given points.
    """
    light_locations = center + light_distance*light_directions
    light_local_distances, light_local_directions = vector_tools.norm_vector(light_locations - jax.numpy.expand_dims(points, axis=-2))
    light_local_power = jax.numpy.expand_dims(dir_light_power * jax.numpy.square(light_distance), axis=-2) / jax.numpy.square(light_local_distances)
    light_local_intensity =  jax.numpy.expand_dims(light_local_power, axis=-1)
    return light_local_directions, light_local_intensity

def get_isotropic_punctual_light(light_locations, light_power, points):
    """
    Computes the local direction and intensity of isotropic punctual light sources at given points.

    Args:
        light_locations (Array ..., l, 3): Locations of l light sources.
        light_power (Array ..., l): Power of l light sources.
        points (Array ..., 3): The points in space where the light is to be evaluated.

    Returns:
        Tuple[Array ..., l, 3], Array ..., l, c]: Local directions and intensities of light sources at given points.
    """
    light_local_distances, light_local_directions = vector_tools.norm_vector(light_locations - jax.numpy.expand_dims(points, axis=-2))
    light_local_power = jax.numpy.expand_dims(light_power, axis=-2) / jax.numpy.square(light_local_distances)
    light_local_intensity =  jax.numpy.expand_dims(light_local_power, axis=-1)
    return light_local_directions, light_local_intensity

def get_led_light(light_locations, light_power, light_principal_direction, mu, points):
    """
    Computes the local direction and intensity of LED light sources with directional and anisotropic properties.

    Args:
        light_locations (Array ..., l, 3): Locations of l light sources.
        light_power (Array ..., l): Power of l light sources.
        light_principal_direction (Array ..., l, 3): Principal direction of the LED light source.
        mu (Array ..., c): Anisotropy parameter controlling the intensity falloff based on the direction.
        points (Array ..., 3): The points in space where the light is to be evaluated.

    Returns:
        Tuple[Array ..., l, 3], Array ..., l, c]: Local directions and intensities of LED light sources at given points, accounting for anisotropic lighting.
    """
    light_local_distances, light_local_directions = vector_tools.norm_vector(light_locations - jax.numpy.expand_dims(points, axis=-2))
    angular_factor = jax.nn.relu(jax.numpy.einsum('...li, li -> ...l', -light_local_directions, light_principal_direction))
    anisotropy = jax.numpy.power(jax.numpy.expand_dims(angular_factor, axis=-1), mu)
    light_local_power = jax.numpy.expand_dims(light_power, axis=-2) / jax.numpy.square(light_local_distances)
    light_local_intensity = jax.numpy.einsum('...l, ...lc ->...lc', light_local_power, anisotropy)
    return light_local_directions, light_local_intensity

def get_grid_light(direction_grid, intensity_grid, pixels, min_range, max_range, span=3):
    x_transform = (jax.numpy.asarray(direction_grid.shape[:2])-1)*(pixels-min_range)/(max_range-min_range)
    grid_interpolator = lambda grid : lanczos.get_lanczos_reampler(lanczos.grid_from_array(grid), span)(x_transform)[0]
    light_local_directions_unnormed =  jax.numpy.moveaxis(jax.lax.map(grid_interpolator, jax.numpy.moveaxis(direction_grid, 2, 0)), 0, -2)
    light_local_intensity = jax.numpy.moveaxis(jax.lax.map(grid_interpolator, jax.numpy.moveaxis(intensity_grid, 2, 0)), 0, -2)
    light_local_directions =  vector_tools.norm_vector(light_local_directions_unnormed)[1]
    return light_local_directions, light_local_intensity

def get_constant_light(light_local_direction, light_local_intensity):
    """
    Returns the local direction and intensity of a constant light source.
    Parameters:
        light_local_direction (jax.numpy.ndarray): An array of shape (..., P, L, 3) 
              representing the normalized directions from the points to the light sources
        light_local_intensity (jax.numpy.ndarray): An array of shape (..., P, L, C) 
              representing the computed light intensity at the points for each light source.
    Returns:
        tuple: A tuple containing the light's local direction and intensity.
    """

    return light_local_direction, light_local_intensity
    

def get_harmonic_light(light_locations, light_power, light_principal_direction, free_rotation, coefficients, l_max, points):
    """
    Computes the harmonic light intensity and directions for a set of light sources.

    This function calculates the local light directions and intensities at given points
    based on the spherical harmonics representation of light anisotropy, light power, 
    and distances from the light sources.

    Args:
        light_locations (jax.numpy.ndarray): An array of shape (..., L, 3) representing 
            the 3D positions of the light sources.
        light_power (jax.numpy.ndarray): An array of shape (..., L) representing the power 
            of each light source.
        light_principal_direction (jax.numpy.ndarray): An array of shape (..., L, 3) representing 
            the principal direction of each light source.
        free_rotation (jax.numpy.ndarray): An array of shape (..., L) representing 
            the free rotation of the light sources.
        coefficients (jax.numpy.ndarray): An array of shape (M, C) representing the 
            spherical harmonics coefficients for light anisotropy.
        l_max (int): The maximum degree of spherical harmonics to consider.
        points (jax.numpy.ndarray): An array of shape (..., P, 3) representing the 3D positions 
            of the points where light intensity is computed.

    Returns:
        tuple: A tuple containing:
            - light_local_directions (jax.numpy.ndarray): An array of shape (..., P, L, 3) 
              representing the normalized directions from the points to the light sources.
            - light_local_intensity (jax.numpy.ndarray): An array of shape (..., P, L, C) 
              representing the computed light intensity at the points for each light source.
    """

    light_local_distances, light_local_directions = vector_tools.norm_vector(light_locations - jax.numpy.expand_dims(points, axis=-2))
    g_coefficients = vector_tools.partial_stop_gradients(coefficients, 0).T[:,None,None,:]
    swaped_anisotropy = spherical_harmonics.oriented_sh_function(-light_local_directions, light_principal_direction, free_rotation, g_coefficients, int(l_max))
    anisotropy = jax.nn.relu(jax.numpy.moveaxis(swaped_anisotropy, 0, -1))
    light_local_power = jax.numpy.expand_dims(light_power, axis=-2) / jax.numpy.square(light_local_distances)
    light_local_intensity = jax.numpy.einsum('...l, ...lc ->...lc', light_local_power, anisotropy)
    return light_local_directions, light_local_intensity