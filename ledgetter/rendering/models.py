import jax
import optax
import ledgetter.rendering.renderers as renderers
import ledgetter.rendering.lights as lights
import functools


def get_loss(model, delta=0.01):
    def loss(parameters, data):
        normals, points, images, validity_mask = data
        render =  model(parameters, normals, points)
        errors = optax.losses.huber_loss(render, targets = images, delta = delta)
        loss_value = jax.numpy.mean(errors, where = validity_mask)
        return loss_value
    return loss

def to_geometric_loss(loss, parameters_raw, data_raw):
    def geometric_loss(parameters, data):
        parameters_loss, data_loss = unravel_geometric_params(parameters, data)
        loss_value = loss(parameters_loss, data_loss)
        return loss_value
    projections = (functools.partial(optax.projections.projection_box, lower = 0, upper=1), jax.vmap(optax.projections.projection_l2_sphere, in_axes=0))
    parameters_g, data_g = ravel_geometric_params(parameters_raw, data_raw)
    return geometric_loss, (parameters_g, data_g), projections

def ravel_geometric_params(parameters, data):
    rho, normals = parameters[0], data[0]
    parameters_g = (rho, normals)
    data_g = (data, parameters)
    return parameters_g, data_g

def unravel_geometric_params(parameters_g, data_g):
    rho, normals = parameters_g
    data_raw, parameters_raw = data_g
    parameters = (rho,) + parameters_raw[1:]
    data = (normals,) + data_raw[1:]
    return parameters, data


def get_directional_model():
    def model(parameters, normals, points):
        rho, light_directions, light_power = parameters
        light_directions, light_intensity = lights.get_directional_light(light_directions, light_power)
        render =  renderers.lambertian_renderer(light_directions, light_intensity, normals, points, rho)
        return render
    projections = (functools.partial(optax.projections.projection_box, lower = 0, upper=1), jax.vmap(optax.projections.projection_l2_sphere, in_axes=0), optax.projections.projection_non_negative)
    return model, projections

def get_rail_punctual_model( light_directions, light_power, center):
    def model(parameters, normals, points):
        rho, light_distance = parameters
        light_local_directions, light_local_intensity = lights.get_rail_light(center, light_distance, light_directions, light_power, points)
        render =  renderers.lambertian_renderer(light_local_directions, light_local_intensity, normals, points, rho)
        return render
    projections = (functools.partial(optax.projections.projection_box, lower = 0, upper=1), optax.projections.projection_non_negative)
    return model, projections

def get_isotropic_punctual_model():
    def model(parameters, normals, points):
        rho, light_locations, light_power = parameters
        light_local_directions, light_local_intensity = lights.get_isotropic_punctual_light(light_locations, light_power, points)
        render =  renderers.lambertian_renderer(light_local_directions, light_local_intensity, normals, points, rho)
        return render
    projections = (functools.partial(optax.projections.projection_box, lower = 0, upper=1), lambda p : p , optax.projections.projection_non_negative)
    return model, projections

def get_led_model():
    def model(parameters, normals, points):
        rho, light_locations, light_power, light_principal_direction, mu = parameters
        light_local_directions, light_local_intensity = lights.get_led_light(light_locations, light_power, light_principal_direction, mu, points)
        render =  renderers.lambertian_renderer(light_local_directions, light_local_intensity, normals, points, rho)
        return render
    projections = (functools.partial(optax.projections.projection_box, lower = 0, upper=1), lambda p : p , optax.projections.projection_non_negative, jax.vmap(optax.projections.projection_l2_sphere, in_axes=0), optax.projections.projection_non_negative)
    return model, projections

def get_specular_led_model():
    def model(parameters, normals, points):
        rho, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec = parameters
        light_local_directions, light_local_intensity = lights.get_led_light(light_locations, light_power, light_principal_direction, mu, points)
        lambertian_render =  renderers.lambertian_renderer(light_local_directions, light_local_intensity, normals, points, rho)
        specular_render = renderers.specular_renderer(light_local_directions, light_local_intensity, normals, points, rho_spec, tau_spec)
        render = lambertian_render + specular_render
        return render
    projections = (functools.partial(optax.projections.projection_box, lower = 0, upper=1), lambda p : p , optax.projections.projection_non_negative, jax.vmap(optax.projections.projection_l2_sphere, in_axes=0), optax.projections.projection_non_negative, functools.partial(optax.projections.projection_box, lower = 0, upper=1), optax.projections.projection_non_negative)
    return model, projections
