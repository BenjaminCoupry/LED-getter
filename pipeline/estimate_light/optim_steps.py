import jax
import ledgetter.optim.gradient_descent as gradient_descent
import ledgetter.rendering.models as models

import ledgetter.rendering.lights as lights
import ledgetter.utils.vector_tools as vector_tools




def estimate_grid_light(points, normals, images, pixels, shapes, output, optimizer, mask, validity_mask, iterations, pixel_step):
    losses, steps = [], []

    rho = init_rho(images)
    direction_grid, intensity_grid, min_range, max_range = init_grid(shapes, pixels, pixel_step)
    if 'lambertian' in iterations:
        it = iterations['lambertian']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels, 'min_range':min_range, 'max_range':max_range}
        parameters = {'rho' : rho, 'direction_grid' : direction_grid, 'intensity_grid':intensity_grid}
        loss, projections = models.get_loss({'light':'grid', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        rho, direction_grid, intensity_grid = parameters['rho'], parameters['direction_grid'], parameters['intensity_grid']
        losses.append(losses_values)
        steps.append('lambertian')
    if True:
        return parameters, data, losses, steps

def estimate_physical_light(points, normals, images, pixels, shapes, output, optimizer, mask, validity_mask, iterations):
    losses, steps = [], []

    rho = init_rho(images)
    light_directions, light_power = init_directional_light(shapes)
    if 'directional' in iterations:
        it = iterations['directional']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels}
        parameters = {'rho' : rho, 'light_directions' : light_directions, 'light_power':light_power}
        loss, projections = models.get_loss({'light':'directional', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        rho, light_directions, light_power = parameters['rho'], parameters['light_directions'], parameters['light_power']
        losses.append(losses_values)
        steps.append('directional')
    if 'rail' not in iterations and 'punctual' not in iterations and 'LED' not in iterations and 'specular' not in iterations:
        return parameters, data, losses, steps
    
    light_distance, center = init_light_distance(points)
    if 'rail' in iterations:
        it = iterations['rail']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'center':center, 'light_directions' : light_directions, 'light_power': light_power, 'pixels':pixels}
        parameters = {'rho': rho, 'light_distance': light_distance}
        loss, projections = models.get_loss({'light':'rail', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        rho, light_distance = parameters['rho'], parameters['light_distance']
        losses.append(losses_values)
        steps.append('rail')
    if 'punctual' not in iterations and 'LED' not in iterations and 'specular' not in iterations:
        return parameters, data, losses, steps
    
    light_locations, light_power = init_punctual_light(points, light_distance, light_directions, light_power)
    if 'punctual' in iterations:
        it = iterations['punctual']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels}
        parameters = {'rho': rho, 'light_locations': light_locations, 'light_power': light_power}
        loss, projections = models.get_loss({'light':'punctual', 'renderers':['lambertian'], 'parameters' : list(parameters.keys())}, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        rho, light_locations, light_power = parameters['rho'], parameters['light_locations'], parameters['light_power']
        losses.append(losses_values)
        steps.append('punctual')
    if 'LED' not in iterations and 'specular' not in iterations:
        return parameters, data, losses, steps
    
    light_principal_direction, mu = init_led_light(shapes, points, light_locations)
    if 'LED' in iterations:
        it = iterations['LED']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels}
        parameters = {'rho': rho, 'light_locations': light_locations, 'light_power': light_power, 'light_principal_direction': light_principal_direction, 'mu': mu}
        loss, projections = models.get_loss({'light':'LED', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        rho, light_locations, light_power, light_principal_direction, mu = parameters['rho'], parameters['light_locations'], parameters['light_power'], parameters['light_principal_direction'], parameters['mu']
        losses.append(losses_values)
        steps.append('LED')
    if 'specular' not in iterations:
        return parameters, data, losses, steps
    
    rho_spec, tau_spec = init_specular(shapes)
    if 'specular' in iterations:
        it = iterations['specular']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels}
        parameters = {'rho': rho, 'light_locations': light_locations, 'light_power': light_power, 'light_principal_direction': light_principal_direction, 'mu': mu, 'rho_spec': rho_spec, 'tau_spec': tau_spec}
        loss, projections = models.get_loss({'light':'LED', 'renderers':['lambertian','specular'], 'parameters'  : list(parameters.keys())}, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        rho, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec = parameters['rho'], parameters['light_locations'], parameters['light_power'], parameters['light_principal_direction'], parameters['mu'], parameters['rho_spec'], parameters['tau_spec']
        losses.append(losses_values)
        steps.append('specular')
    if True:
        return parameters, data, losses, steps


def init_rho(images):
    rho =  jax.numpy.mean(images, axis=-1)
    return rho

def init_directional_light(shapes):
    (n_pix, n_im, n_c) = shapes
    light_directions = jax.numpy.zeros((n_im, 3)).at[:,2].set(-1)
    light_power = jax.numpy.ones((n_im,))
    return light_directions, light_power

def init_light_distance(points):
    center = jax.numpy.mean(points, axis=0)
    light_distance = jax.numpy.max(jax.numpy.linalg.norm(points-center,axis=-1))*3
    return light_distance, center

def init_punctual_light(points, light_distance, light_directions, light_power):
    center = jax.numpy.mean(points, axis=0)
    light_locations = center + light_distance*light_directions
    light_power = light_power*jax.numpy.square(light_distance)
    return light_locations, light_power

def init_led_light(shapes, points, light_locations):
    (n_pix, n_im, n_c) = shapes
    center = jax.numpy.mean(points, axis=0)
    light_principal_direction = (center-light_locations)/jax.numpy.linalg.norm(center-light_locations,axis=-1, keepdims=True)
    mu = jax.numpy.ones((n_c,))*0.1
    return light_principal_direction, mu

def init_specular(shapes):
    (n_pix, n_im, n_c) = shapes
    rho_spec = jax.numpy.ones((n_pix,))*0.0
    tau_spec = jax.numpy.ones(tuple())*20.0
    return rho_spec, tau_spec


def init_grid(shapes, pixels, pixel_step):
    (n_pix, n_im, n_c) = shapes
    min_range, max_range = jax.numpy.min(pixels, axis=0), jax.numpy.max(pixels, axis=0)
    nx, ny = int((max_range[0]-min_range[0])/pixel_step), int((max_range[1]-min_range[1])/pixel_step)
    direction_grid = jax.numpy.zeros((nx, ny ,n_im, 3)).at[:,:,:,2].set(-1)
    intensity_grid = jax.numpy.ones((nx, ny ,n_im, 1))
    return direction_grid, intensity_grid, min_range, max_range