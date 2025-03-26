import jax
import ledgetter.utils.vector_tools as vector_tools


def get_directional_light(light_directions, light_power):
    """
    Generates directional light sources with given directions and power.

    Args:
        light_directions (Array ..., l, 3): Directions of l light sources.
        light_power (Array ..., l): Power of l light sources.

    Returns:
        Tuple[Array ..., l, 3], Array ..., l, c]: Light directions and corresponding intensities.
    """
    light_intensity = jax.numpy.expand_dims(light_power, axis=-1)
    return light_directions, light_intensity

def get_rail_light(center, light_distance, light_directions, light_power, points):
    """
    Computes the local direction and intensity of light sources along a rail, 
    based on their distance from a given center and their direction.

    Args:
        center (Array ..., 3): The center position of the light source.
        light_distance (Array ..., l): The distance of light sources from the center.
        light_directions (Array ..., l, 3): The directions of l light sources.
        light_power (Array ..., l): The power of l light sources.
        points (Array ..., 3): The points in space where the light is to be evaluated.

    Returns:
        Tuple[Array ..., l, 3], Array ..., l, c]: Local directions and intensities of light sources at given points.
    """
    light_locations = center + light_distance*light_directions
    light_local_distances, light_local_directions = vector_tools.norm_vector(light_locations - jax.numpy.expand_dims(points, axis=-2))
    light_local_power = jax.numpy.expand_dims(light_power * jax.numpy.square(light_distance), axis=-2) / jax.numpy.square(light_local_distances)
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
    angular_factor = jax.nn.relu(jax.numpy.einsum('pli, li -> pl', -light_local_directions, light_principal_direction))
    anisotropy = jax.numpy.power(jax.numpy.expand_dims(angular_factor, axis=-1), mu)
    light_local_power = jax.numpy.expand_dims(light_power, axis=-2) / jax.numpy.square(light_local_distances)
    light_local_intensity = jax.numpy.einsum('pl, plc ->plc', light_local_power, anisotropy)
    return light_local_directions, light_local_intensity