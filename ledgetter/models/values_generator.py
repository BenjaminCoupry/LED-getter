import jax
import ledgetter.space.spherical_harmonics as spherical_harmonics
import ledgetter.models.models as models
import scipy.interpolate
import ledgetter.utils.functions as functions

registry = {}

def merge_light_values(values, light_values):
    global_light_values = {k: v for k, v in light_values.items() if not models.is_pixelwise(k)}
    local_light_values = {k: jax.numpy.asarray(scipy.interpolate.NearestNDInterpolator(light_values['pixels'], v)(values['pixels'])) for k, v in light_values.items() if models.is_pixelwise(k)}
    merged_values =  global_light_values | local_light_values | values
    return merged_values


def generate_missing_values(values, values_to_generate, shapes, images, light=None):
    generated_values =  functions.execute_generators(registry, values_to_generate, values, shapes=shapes, images=images, light=light)
    return generated_values


def split_parameters_data(values, wanted_params, wanted_data):
    parameters = {k: v for k, v in values.items() if k in wanted_params}
    data = {k: v for k, v in values.items() if k in wanted_data}
    return parameters, data

@functions.generator(inputs=[], outputs=['rho'], registry=registry)
@functions.filter_args
def init_rho(images):
    rho =  jax.numpy.mean(images, axis=-1)
    return {'rho': rho}

@functions.generator(inputs=[], outputs=['light_directions', 'dir_light_power'], registry=registry)
@functions.filter_args
def init_directional_light(shapes):
    (n_pix, n_im, n_c) = shapes
    light_directions = jax.numpy.zeros((n_im, 3)).at[:,2].set(-1)
    dir_light_power = jax.numpy.ones((n_im,))
    return {
        'light_directions': light_directions,
        'dir_light_power': dir_light_power
    }

@functions.generator(inputs=['points'], outputs=['light_distance', 'center'], registry=registry)
@functions.filter_args
def init_light_distance(points):
    center = jax.numpy.mean(points, axis=0)
    scale = jax.numpy.max(jax.numpy.linalg.norm(points-center,axis=-1))
    light_distance = scale*3
    return {
        'light_distance': light_distance,
        'center': center
    }

@functions.generator(inputs=['points', 'light_distance', 'light_directions', 'dir_light_power'], outputs=['light_locations', 'light_power'], registry=registry)
@functions.filter_args
def init_punctual_light(points, light_distance, light_directions, dir_light_power):
    center = jax.numpy.mean(points, axis=0)
    light_locations = center + light_distance*light_directions
    light_power = dir_light_power*jax.numpy.square(light_distance)
    return {
        'light_locations': light_locations,
        'light_power': light_power
    }

@functions.generator(inputs=['points', 'light_locations'], outputs=['light_principal_direction', 'mu'], registry=registry)
@functions.filter_args
def init_led_light(shapes, points, light_locations):
    (n_pix, n_im, n_c) = shapes
    center = jax.numpy.mean(points, axis=0)
    light_principal_direction = (center-light_locations)/jax.numpy.linalg.norm(center-light_locations,axis=-1, keepdims=True)
    mu = jax.numpy.ones((n_c,))*0.1
    return {
        'light_principal_direction': light_principal_direction,
        'mu': mu
    }

@functions.generator(inputs=[], outputs=['rho_spec', 'tau_spec'], registry=registry)
@functions.filter_args
def init_specular(shapes):
    (n_pix, n_im, n_c) = shapes
    rho_spec = jax.numpy.ones((n_pix,))*0.0
    tau_spec = jax.numpy.ones(tuple())*20.0
    return {
        'rho_spec': rho_spec,
        'tau_spec': tau_spec
    }

@functions.generator(inputs=['mu'], outputs=['free_rotation', 'coefficients', 'indices', 'l_max'], registry=registry)
@functions.filter_args
def init_sh_light(shapes, mu, l_max=5):
    (n_pix, n_im, n_c) = shapes
    goal_function = lambda teta : jax.numpy.power(jax.numpy.cos(teta[:,None]), mu)
    coefficients, indices = spherical_harmonics.coefficients_from_colatitude(goal_function, l_max)
    free_rotation = jax.numpy.zeros((n_im,))
    return {
        'free_rotation': free_rotation,
        'coefficients': coefficients,
        'indices': indices,
        'l_max': l_max
    }

@functions.generator(inputs=['pixels'], outputs=['direction_grid', 'intensity_grid', 'min_range', 'max_range'], registry=registry)
@functions.filter_args
def init_grid(shapes, pixels, pixel_step):
    (n_pix, n_im, n_c) = shapes
    min_range, max_range = jax.numpy.min(pixels, axis=0), jax.numpy.max(pixels, axis=0)
    nx, ny = int((max_range[0]-min_range[0])/pixel_step), int((max_range[1]-min_range[1])/pixel_step)
    direction_grid = jax.numpy.zeros((nx, ny ,n_im, 3)).at[:,:,:,2].set(-1)
    intensity_grid = jax.numpy.ones((nx, ny ,n_im, 1))
    return {
        'direction_grid': direction_grid,
        'intensity_grid': intensity_grid,
        'min_range': min_range,
        'max_range': max_range
    }

@functions.generator(inputs=[], outputs=['light_local_direction', 'light_local_intensity'], registry=registry)
def init_constant_light(light, **kwargs):
    light_local_direction, light_local_intensity = light(**kwargs)
    return {
        'light_local_direction': light_local_direction,
        'light_local_intensity': light_local_intensity,
    }
