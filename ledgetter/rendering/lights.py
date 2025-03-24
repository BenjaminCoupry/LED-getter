import jax

def get_directional_light(light_directions, light_power):
    light_intensity = jax.numpy.expand_dims(light_power, axis=-1)
    return light_directions, light_intensity

def get_rail_light(center, light_distance, light_directions, light_power, points):
    light_locations = center + light_distance*light_directions
    light_local_vectors = light_locations - jax.numpy.expand_dims(points, axis=-2)
    light_local_distances = jax.numpy.linalg.norm(light_local_vectors, axis=-1)
    light_local_directions = light_local_vectors / jax.numpy.expand_dims(light_local_distances, axis=-1)
    light_local_power = jax.numpy.expand_dims(light_power * jax.numpy.square(light_distance), axis=-2) / jax.numpy.square(light_local_distances)
    light_local_intensity =  jax.numpy.expand_dims(light_local_power, axis=-1)
    return light_local_directions, light_local_intensity

def get_isotropic_punctual_light(light_locations, light_power, points):
    light_local_vectors = light_locations - jax.numpy.expand_dims(points, axis=-2)
    light_local_distances = jax.numpy.linalg.norm(light_local_vectors, axis=-1)
    light_local_directions = light_local_vectors / jax.numpy.expand_dims(light_local_distances, axis=-1)
    light_local_power = jax.numpy.expand_dims(light_power, axis=-2) / jax.numpy.square(light_local_distances)
    light_local_intensity =  jax.numpy.expand_dims(light_local_power, axis=-1)
    return light_local_directions, light_local_intensity

def get_led_light(light_locations, light_power, light_principal_direction, mu, points):
    light_local_vectors = light_locations - jax.numpy.expand_dims(points, axis=-2)
    light_local_distances = jax.numpy.linalg.norm(light_local_vectors, axis=-1)
    light_local_directions = light_local_vectors / jax.numpy.expand_dims(light_local_distances, axis=-1)
    angular_factor = jax.nn.relu(jax.numpy.einsum('pli, li -> pl', -light_local_directions, light_principal_direction))
    anisotropy = jax.numpy.power(jax.numpy.expand_dims(angular_factor, axis=-1), mu)
    light_local_power = jax.numpy.expand_dims(light_power, axis=-2) / jax.numpy.square(light_local_distances)
    light_local_intensity = jax.numpy.einsum('pl, plc ->plc', light_local_power, anisotropy)
    return light_local_directions, light_local_intensity