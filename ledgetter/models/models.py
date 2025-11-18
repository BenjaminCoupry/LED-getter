import jax
import optax
import ledgetter.rendering.renderers as renderers
import ledgetter.rendering.lights as lights
import ledgetter.rendering.validity as validity
import functools
import ledgetter.utils.functions as functions

def is_pixelwise(value):
    return value in {'rho', 'rho_spec', 'normals', 'points', 'pixels', 'images', 'light_local_direction', 'light_local_intensity', 'validity_mask', 'objects_id'}


def get_intrinsic_shape(value, shapes):
    (n_pix, n_im, n_c) = shapes
    match value:
        case 'rho':
            return jax.ShapeDtypeStruct((n_pix, n_c), jax.numpy.float32)
        case 'normals' | 'points':
            return jax.ShapeDtypeStruct((n_pix, 3), jax.numpy.float32)
        case 'light_local_intensity':
            return jax.ShapeDtypeStruct((n_pix, n_im, n_c), jax.numpy.float32)
        case 'light_local_direction':
            return jax.ShapeDtypeStruct((n_pix, n_im, 3), jax.numpy.float32)
        case 'validity_mask':
            return jax.ShapeDtypeStruct((n_pix, n_im), jax.numpy.bool)
        case 'pixels':
            return jax.ShapeDtypeStruct((n_pix, 2), jax.numpy.int32)
        case _:
            #TODO
            raise ValueError(f"Unknown value: {value}") 

def get_projection(parameter):
    match parameter:
        case 'rho' | 'rho_spec':
            return functools.partial(optax.projections.projection_box, lower=0, upper=1)
        case 'light_directions' | 'light_principal_direction' | 'normals' | 'direction_grid':
            return jax.numpy.vectorize(optax.projections.projection_l2_sphere, signature='(n)->(n)')
        case 'dir_light_power' | 'light_power' | 'light_distance' | 'mu' | 'tau_spec' | 'intensity_grid':
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
        case 'phong':
            return renderers.phong_renderer
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


def get_valid(valid) :
    def valid_function(shapes, **kwargs):
        (n_pix, n_im, n_c) = shapes
        validity_mask = jax.numpy.ones((n_pix, n_im),dtype=bool)
        for validity_masker_i in valid['validity_maskers']:
            validity_masker = get_validity(validity_masker_i)
            skip_valid = (validity_masker_i == 'cast_shadow') and (('light' not in kwargs) or (kwargs['light'] is None) or ('raycaster' not in valid['options']) or (valid['options']['raycaster'] is None))
            if not skip_valid:
                light_local_directions = kwargs['light'](**kwargs)[0] if (validity_masker_i == 'cast_shadow')  else None
                validity_mask_i = validity_masker(**(valid['options'] | kwargs | {'validity_mask' : validity_mask, 'light_local_directions' : light_local_directions}))
                validity_mask = jax.numpy.logical_and(validity_mask, validity_mask_i)
        return validity_mask
    return valid_function

def get_grouped_renderer(unique_renderers):
    renderer = lambda *args, **kwargs : {renderer_i:
        get_renderer(renderer_i)(*args, **kwargs) for renderer_i in unique_renderers}
    return renderer

def get_model(model):
    light = get_light(model['light'])
    renderer = get_grouped_renderer(model['renderers'])
    projections = {parameter : get_projection(parameter) for parameter in model['parameters']}
    return light, renderer, projections


def get_loss(light, renderer, delta=0.01):
    def loss(parameters, data, images, validity_mask):
        values =  data | parameters
        light_directions, light_intensity = light(**values)
        renders = renderer(light_directions, light_intensity, **values)
        render = jax.tree_util.tree_reduce(lambda x, y : x + y, renders)
        errors = optax.losses.huber_loss(render, targets = images, delta = delta)
        loss_value = jax.numpy.nanmean(errors, where = jax.numpy.expand_dims(validity_mask, axis=-2))
        return loss_value
    return loss


def model_from_parameters(parameters, data):
    values = parameters | data
    renderers = renderers_from_values(values)
    light_type = light_from_values(values)
    model = {'light': light_type, 'renderers': renderers, 'parameters': list(parameters), 'data': list(data)}
    return model


def light_from_values(values):
    light = None
    if all(key in values for key in ['direction_grid', 'intensity_grid', 'pixels']):
        light = 'grid'
    elif all(key in values for key in ['light_locations', 'light_power', 'light_principal_direction', 'free_rotation', 'coefficients', 'points']):
        light = 'harmonic'
    elif all(key in values for key in ['light_locations', 'light_power', 'light_principal_direction', 'mu', 'points']):
        light = 'LED'
    elif all(key in values for key in ['light_locations', 'light_power', 'points']):
        light = 'punctual'
    elif all(key in values for key in ['center', 'light_distance', 'light_directions', 'dir_light_power', 'points']):
        light = 'rail'
    elif all(key in values for key in ['light_directions', 'dir_light_power', 'points']):
        light = 'directional'
    elif all(key in values for key in ['light_local_direction', 'light_local_intensity']):
        light = 'constant'
    else:
        raise ValueError(f"Unknown light")
    return light

def renderers_from_values(values):
    renderers = []
    if 'rho' in values:
        renderers.append('lambertian')
    if 'rho_spec' in values and 'tau_spec' in values:
        renderers.append('phong')
    return renderers
