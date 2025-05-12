import jax
import optax
import ledgetter.rendering.renderers as renderers
import ledgetter.rendering.lights as lights
import ledgetter.rendering.validity as validity
import functools
import ledgetter.utils.functions as functions


def is_pixelwise(value):
    return value in {'rho', 'rho_spec', 'normals', 'points', 'pixels', 'validity_mask'}

def get_projection(parameter):
    match parameter: #TODO : project 'coefficients' to have the C0 equal between channels
        case 'rho' | 'rho_spec':
            return functools.partial(optax.projections.projection_box, lower=0, upper=1)
        case 'light_directions' | 'light_principal_direction' | 'normals' | 'direction_grid':
            return jax.numpy.vectorize(optax.projections.projection_l2_sphere, signature='(n)->(n)')
        case 'light_power' | 'light_distance' | 'mu' | 'tau_spec' | 'intensity_grid':
            return optax.projections.projection_non_negative
        case 'light_locations' | 'points' | 'pixels' | 'coefficients' | 'free_rotation':
            return lambda p: p
        case _:
            raise ValueError(f"Unknown parameter: {parameter}") 

@functions.filter_output_args
def get_validity(validity_masker):
    match validity_masker:
        case 'intensity':
            return validity.intensity_validity
        case 'cast_shadow':
            return validity.cast_shadow_validity
        case 'morphology':
            return validity.morphological_validity
        case _:
            raise ValueError(f"Unknown validity {validity_masker}") 
        
@functions.filter_output_args
def get_renderer(renderer):
    match renderer:
        case 'lambertian':
            return renderers.lambertian_renderer
        case 'specular':
            return renderers.specular_renderer
        case _:
            raise ValueError(f"Unknown renderer: {renderer}")

@functions.filter_output_args
def get_light(light):
    match light:
        case 'directional':
            return lights.get_directional_light
        case 'rail':
            return lights.get_rail_light
        case 'punctual':
            return lights.get_isotropic_punctual_light
        case 'LED':
            return lights.get_led_light
        case 'harmonic':
            return lights.get_harmonic_light
        case 'grid':
            return lights.get_grid_light
        case 'constant':
            return lights.get_constant_light
        case _:
            raise ValueError(f"Unknown light: {light}")


def get_valid(valid, shapes) :
    (n_pix, n_im, n_c) = shapes
    def valid_function(**kwargs):
        validity_mask = jax.numpy.ones((n_pix, n_im),dtype=bool)
        for validity_masker_i in valid['validity_maskers']:
            validity_masker = get_validity(validity_masker_i)
            light_local_directions = kwargs['light'](**kwargs)[0] if ('raycaster' in valid['options']) and (validity_masker_i == 'cast_shadow')  else None
            validity_mask_i = validity_masker(**(valid['options'] | kwargs | {'validity_mask' : validity_mask, 'light_local_directions' : light_local_directions}))
            validity_mask = jax.numpy.logical_and(validity_mask, validity_mask_i)
        return validity_mask
    return valid_function


def get_model(model):
    light = get_light(model['light'])
    renderer = lambda *args, **kwargs : {renderer_i:
        get_renderer(renderer_i)(*args, **kwargs) for renderer_i in model['renderers']}
    projections = {parameter : get_projection(parameter) for parameter in model['parameters']}
    return light, renderer, projections


def get_loss(light, renderer, delta=0.01):
    def loss(parameters, data, images):
        values =  data | parameters
        light_directions, light_intensity = light(**values)
        renders = renderer(light_directions, light_intensity, **values)
        render = jax.tree_util.tree_reduce(lambda x, y : x + y, renders)
        errors = optax.losses.huber_loss(render, targets = images, delta = delta)
        loss_value = jax.numpy.nanmean(errors, where = values['validity_mask'])
        return loss_value
    return loss


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
    elif all(key in values for key in ['light_locations', 'light_power', 'light_principal_direction', 'free_rotation', 'coefficients', 'points']):
        light_type = 'harmonic'
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
