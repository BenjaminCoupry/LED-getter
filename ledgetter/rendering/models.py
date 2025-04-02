import jax
import optax
import ledgetter.rendering.renderers as renderers
import ledgetter.rendering.lights as lights
import functools



def get_projection(parameter):
    match parameter:
        case 'rho' | 'rho_spec':
            return functools.partial(optax.projections.projection_box, lower=0, upper=1)
        case 'light_directions' | 'light_principal_direction' | 'normals' | 'direction_grid':
            return jax.numpy.vectorize(optax.projections.projection_l2_sphere, signature='(n)->(n)')
        case 'light_power' | 'light_distance' | 'mu' | 'tau_spec' | 'intensity_grid':
            return optax.projections.projection_non_negative
        case 'light_locations' | 'points' | 'pixels':
            return lambda p: p
        case _:
            raise ValueError(f"Unknown parameter: {parameter}")

def get_renderer(renderer):
    match renderer:
        case 'lambertian':
            return lambda light_directions, light_intensity, values : renderers.lambertian_renderer(light_directions, light_intensity, values['normals'], values['points'], values['rho'])
        case 'specular':
            return lambda light_directions, light_intensity, values : renderers.specular_renderer(light_directions, light_intensity, values['normals'], values['points'], values['rho_spec'], values['tau_spec'])
        case _:
            raise ValueError(f"Unknown renderer: {renderer}")

def get_light(light):
    match light:
        case 'directional':
            return lambda values: lights.get_directional_light(
                values['light_directions'], values['light_power'], values['points']
            )
        case 'rail':
            return lambda values: lights.get_rail_light(
                values['center'], values['light_distance'], values['light_directions'], 
                values['light_power'], values['points']
            )
        case 'punctual':
            return lambda values: lights.get_isotropic_punctual_light(
                values['light_locations'], values['light_power'], values['points']
            )
        case 'LED':
            return lambda values: lights.get_led_light(
                values['light_locations'], values['light_power'], 
                values['light_principal_direction'], values['mu'], values['points']
            )
        case 'grid':
            return lambda values: lights.get_grid_light(
                values['direction_grid'], values['intensity_grid'], values['pixels'],
                values['min_range'], values['max_range']
            )
        case 'constant':
            return lambda values: (values['light_local_direction'], values['light_local_intensity'])
        case _:
            raise ValueError(f"Unknown light: {light}")
        
def get_model(model):
    light = get_light(model['light'])
    renderer = lambda light_directions, light_intensity, values : {renderer:
        get_renderer(renderer)(light_directions, light_intensity, values) for renderer in model['renderers']}
    projections = {parameter : get_projection(parameter) for parameter in model['parameters']}
    return light, renderer, projections


def get_loss(model, delta=0.01):
    light, renderer, projections = get_model(model)
    def loss(parameters, data, images):
        values =  data | parameters
        light_directions, light_intensity = light(values)
        renders = renderer(light_directions, light_intensity, values)
        render = jax.tree_util.tree_reduce(lambda x, y : x + y, renders)
        errors = optax.losses.huber_loss(render, targets = images, delta = delta)
        loss_value = jax.numpy.mean(errors, where = values['validity_mask'])
        return loss_value
    return loss, projections


def model_from_parameters(parameters, data):
    values = parameters | data
    renderers = []
    if 'rho' in values:
        renderers.append('lambertian')
    if 'rho_spec' in values and 'tau_spec' in values:
        renderers.append('specular')
    light_type = None
    if all(key in values for key in ['direction_grid', 'intensity_grid', 'pixels']):
        light_type = 'grid'
    elif all(key in values for key in ['light_locations', 'light_power', 'light_principal_direction', 'mu', 'points']):
        light_type = 'LED'
    elif all(key in values for key in ['light_locations', 'light_power', 'points']):
        light_type = 'punctual'
    elif all(key in values for key in ['center', 'light_distance', 'light_directions', 'light_power', 'points']):
        light_type = 'rail'
    elif all(key in values for key in ['light_directions', 'light_power', 'points']):
        light_type = 'directional'
    elif all(key in values for key in ['light_local_direction', 'light_local_intensity']):
        light_type = 'constant'
    else:
        raise ValueError(f"Unknown light")
    model = {'light': light_type, 'renderers': renderers, 'parameters': list(parameters)}
    return model
