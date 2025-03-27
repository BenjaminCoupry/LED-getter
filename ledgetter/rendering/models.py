import jax
import optax
import ledgetter.rendering.renderers as renderers
import ledgetter.rendering.lights as lights
import functools


def get_projections(parameters):
    projections_dict = {'rho':functools.partial(optax.projections.projection_box, lower = 0, upper=1),
                   'light_directions':jax.vmap(optax.projections.projection_l2_sphere, in_axes=0),
                   'light_power':optax.projections.projection_non_negative,
                   'light_distance':optax.projections.projection_non_negative,
                   'light_locations':lambda p : p,
                   'light_principal_direction': jax.vmap(optax.projections.projection_l2_sphere, in_axes=0),
                   'mu':optax.projections.projection_non_negative,
                   'rho_spec':functools.partial(optax.projections.projection_box, lower = 0, upper=1), 
                   'tau_spec':optax.projections.projection_non_negative}
    projections = {parameter : projections_dict[parameter] for parameter in parameters}
    return projections


def light_image(values, model):
    if model['light']=='directional':
        light_local_directions, light_local_intensity = lights.get_directional_light(values['light_directions'], values['light_power'])
    if model['light']=='rail':
        light_local_directions, light_local_intensity = lights.get_rail_light(values['center'], values['light_distance'], values['light_directions'], values['light_power'], values['points'])
    if model['light']=='punctual':
        light_local_directions, light_local_intensity = lights.get_isotropic_punctual_light(values['light_locations'], values['light_power'], values['points'])
    if model['light']=='LED':
        light_local_directions, light_local_intensity = lights.get_led_light(values['light_locations'], values['light_power'], values['light_principal_direction'], values['mu'], values['points'])
    return light_local_directions, light_local_intensity

def render_image(light_directions, light_intensity, values, model):
    render = jax.numpy.zeros(values['points'].shape[:-1]+(light_intensity.shape[-1], light_intensity.shape[-2]))
    if 'lambertian' in model['renderers']:
        render +=  renderers.lambertian_renderer(light_directions, light_intensity, values['normals'], values['points'], values['rho'])
    if 'specular' in model['renderers']:
        render += renderers.specular_renderer(light_directions, light_intensity, values['normals'], values['points'], values['rho_spec'], values['tau_spec'])
    return render

def get_loss(model, delta=0.01):
    def loss(parameters, data):
        values =  data | parameters
        light_directions, light_intensity = light_image(values, model)
        render = render_image(light_directions, light_intensity, values, model)
        errors = optax.losses.huber_loss(render, targets = values['images'], delta = delta)
        loss_value = jax.numpy.mean(errors, where = values['validity_mask'])
        return loss_value
    return loss

