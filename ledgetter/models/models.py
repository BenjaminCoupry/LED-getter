import jax
import optax
import ledgetter.rendering.brdfs as brdfs
import ledgetter.rendering.lights as lights
import ledgetter.rendering.validity as validity
import functools
import ledgetter.utils.functions as functions

def is_pixelwise(value):
    return value in {'rho', 'rho_spec', 'normals', 'points', 'pixels', 'filters', 'images', 'light_local_direction', 'light_local_intensity', 'validity_mask', 'objects_id'}


def get_intrinsic_shape(value, shapes):
    (n_pix, n_im, n_c, n_f) = shapes
    match value:
        case 'filters':
            return jax.ShapeDtypeStruct((n_pix, n_c), jax.numpy.float32)
        case 'rho':
            return jax.ShapeDtypeStruct((n_pix, n_f), jax.numpy.float32)
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
        case 'light_locations' | 'points' | 'pixels' | 'coefficients' | 'free_rotation' | 'filters':
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
def get_brdf(brdf):
    match brdf:
        case 'lambertian':
            return brdfs.lambertian_brdf
        case 'phong':
            return brdfs.phong_brdf
        case _:
            raise ValueError(f"Unknown brdf: {brdf}")

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
        (n_pix, n_im, n_c, n_f) = shapes
        validity_mask = jax.numpy.ones((n_pix, n_im),dtype=bool)
        for validity_masker_i in valid['validity_maskers']:
            validity_masker = get_validity(validity_masker_i)
            skip_valid = (validity_masker_i == 'cast_shadow') and (('light' not in kwargs) or (kwargs['light'] is None) or ('raycaster' not in valid['options']) or (valid['options']['raycaster'] is None))
            if not skip_valid:
                light_local_directions = kwargs['light'](**kwargs | {'shapes':shapes})[0] if (validity_masker_i == 'cast_shadow')  else None
                validity_mask_i = validity_masker(**(valid['options'] | kwargs | {'validity_mask' : validity_mask, 'light_local_directions' : light_local_directions}))
                validity_mask = jax.numpy.logical_and(validity_mask, validity_mask_i)
        return validity_mask
    return valid_function

def get_grouped_brdf(unique_brdfs):
    brdf = lambda *args, **kwargs : {brdf_i:
        get_brdf(brdf_i)(*args, **kwargs) for brdf_i in unique_brdfs}
    return brdf

def get_model(model):
    light = get_light(model['light'])
    brdf = get_grouped_brdf(model['brdfs'])
    projections = {parameter : get_projection(parameter) for parameter in model['parameters']}
    return light, brdf, projections

def filters_to_channel(filter_values, filters):
    channel_values = jax.numpy.take_along_axis(filter_values, jax.numpy.broadcast_to(filters[..., :, None], filters.shape + (filter_values.shape[-1],)), axis=-2)
    return channel_values

def channel_to_filters(channel_values, filters):
    n_f = jax.numpy.max(filters) + 1
    filter_values = jax.numpy.put_along_axis(jax.numpy.full(filters.shape[:-1] + (n_f,), jax.numpy.inf), filters, channel_values, axis=-1, inplace=False)
    return filter_values

def get_loss(light, brdf, delta=0.01):
    def loss(parameters, data, images, validity_mask, shapes):
        values =  data | parameters
        light_directions, light_intensity = light(**values | {'shapes':shapes})
        renders = brdf(light_directions, light_intensity, **values)
        render = jax.tree_util.tree_reduce(lambda x, y : x + y, renders)
        channel_render = filters_to_channel(render, values['filters'])
        errors = optax.losses.huber_loss(channel_render, targets = images, delta = delta)
        loss_value = jax.numpy.nanmean(errors, where = jax.numpy.expand_dims(validity_mask, axis=-2))
        return loss_value
    return loss


def model_from_parameters(parameters, data):
    values = parameters | data
    brdfs = brdfs_from_values(values)
    light_type = light_from_values(values)
    model = {'light': light_type, 'brdfs': brdfs, 'parameters': list(parameters), 'data': list(data)}
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

def brdfs_from_values(values):
    brdfs = []
    if 'rho' in values:
        brdfs.append('lambertian')
    if 'rho_spec' in values and 'tau_spec' in values:
        brdfs.append('phong')
    return brdfs
