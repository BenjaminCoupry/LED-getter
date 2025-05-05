import jax
import ledgetter.optim.gradient_descent as gradient_descent
import ledgetter.rendering.models as models
import ledgetter.space.spherical_harmonics as spherical_harmonics
import functools






def estimate_grid_light(points, normals, images, pixels, shapes, output, optimizer, mask, valid_options, iterations, pixel_step):
    losses, steps = [], []
    
    rho = init_rho(images)
    direction_grid, intensity_grid, min_range, max_range = init_grid(shapes, pixels, pixel_step)
    validity_mask = models.get_valid({'validity_maskers':['intensity', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask)
    if 'lambertian' in iterations:
        it = iterations['lambertian']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels, 'min_range':min_range, 'max_range':max_range}
        parameters = {'rho' : rho, 'direction_grid' : direction_grid, 'intensity_grid':intensity_grid}
        model = {'light':'grid', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(functools.partial(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1), data=data, images=images))(parameters)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
        rho, direction_grid, intensity_grid = parameters['rho'], parameters['direction_grid'], parameters['intensity_grid']
        losses.append(losses_values)
        steps.append('lambertian')
    if True:
        return parameters, data, losses, steps

def estimate_physical_light(points, normals, images, pixels, shapes, output, optimizer, mask, valid_options, iterations):
    losses, steps = [], []

    rho = init_rho(images)
    light_directions, light_power = init_directional_light(shapes)
    validity_mask = models.get_valid({'validity_maskers':['intensity', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask)
    if 'directional' in iterations:
        it = iterations['directional']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels}
        parameters = {'rho' : rho, 'light_directions' : light_directions, 'light_power':light_power}
        model = {'light':'directional', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(functools.partial(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1), data=data, images=images))(parameters)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
        rho, light_directions, light_power = parameters['rho'], parameters['light_directions'], parameters['light_power']
        losses.append(losses_values)
        steps.append('directional')
    if 'rail' not in iterations and 'punctual' not in iterations and 'LED' not in iterations and 'harmonic' not in iterations and 'specular' not in iterations:
        return parameters, data, losses, steps
    
    light_distance, center = init_light_distance(points)
    if 'rail' in iterations:
        it = iterations['rail']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'center':center, 'light_directions' : light_directions, 'light_power': light_power, 'pixels':pixels}
        parameters = {'rho': rho, 'light_distance': light_distance}
        model = {'light':'rail', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(functools.partial(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1), data=data, images=images))(parameters)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
        rho, light_distance = parameters['rho'], parameters['light_distance']
        losses.append(losses_values)
        steps.append('rail')
    if 'punctual' not in iterations and 'LED' not in iterations and 'harmonic' not in iterations and 'specular' not in iterations:
        return parameters, data, losses, steps
    
    light_locations, light_power = init_punctual_light(points, light_distance, light_directions, light_power)
    if 'punctual' in iterations:
        it = iterations['punctual']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels}
        parameters = {'rho': rho, 'light_locations': light_locations, 'light_power': light_power}
        model = {'light':'punctual', 'renderers':['lambertian'], 'parameters' : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(functools.partial(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1), data=data, images=images))(parameters)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
        rho, light_locations, light_power = parameters['rho'], parameters['light_locations'], parameters['light_power']
        losses.append(losses_values)
        steps.append('punctual')
    if 'LED' not in iterations and 'harmonic' not in iterations and 'specular' not in iterations:
        return parameters, data, losses, steps
    
    light_principal_direction, mu = init_led_light(shapes, points, light_locations)
    if 'LED' in iterations:
        it = iterations['LED']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels}
        parameters = {'rho': rho, 'light_locations': light_locations, 'light_power': light_power, 'light_principal_direction': light_principal_direction, 'mu': mu}
        model = {'light':'LED', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(functools.partial(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1), data=data, images=images))(parameters)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
        rho, light_locations, light_power, light_principal_direction, mu = parameters['rho'], parameters['light_locations'], parameters['light_power'], parameters['light_principal_direction'], parameters['mu']
        losses.append(losses_values)
        steps.append('LED')
    if  'harmonic' not in iterations and 'specular' not in iterations:
        return parameters, data, losses, steps

    rho_spec, tau_spec = init_specular(shapes)
    if 'specular' in iterations:
        it = iterations['specular']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels}
        parameters = {'rho': rho, 'light_locations': light_locations, 'light_power': light_power, 'light_principal_direction': light_principal_direction, 'mu': mu, 'rho_spec': rho_spec, 'tau_spec': tau_spec}
        model ={'light':'LED', 'renderers':['lambertian','specular'], 'parameters'  : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(functools.partial(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1), data=data, images=images))(parameters)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
        rho, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec = parameters['rho'], parameters['light_locations'], parameters['light_power'], parameters['light_principal_direction'], parameters['mu'], parameters['rho_spec'], parameters['tau_spec']
        losses.append(losses_values)
        steps.append('specular')
    if 'harmonic' not in iterations:
        return parameters, data, losses, steps    

    free_rotation, coefficients, indices, l_max = init_sh_light(shapes, mu, 5)
    if 'harmonic' in iterations:
        it = iterations['harmonic']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels, 'indices':indices, 'l_max':l_max}
        parameters = {'rho': rho, 'light_locations': light_locations, 'light_power': light_power, 'light_principal_direction': light_principal_direction, 'free_rotation': free_rotation, 'coefficients':coefficients, 'rho_spec': rho_spec, 'tau_spec': tau_spec}
        model = {'light':'harmonic', 'renderers':['lambertian', 'specular'], 'parameters'  : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(functools.partial(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1), data=data, images=images))(parameters) 
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
        rho, light_locations, light_power, light_principal_direction, free_rotation, coefficients, rho_spec, tau_spec = parameters['rho'], parameters['light_locations'], parameters['light_power'], parameters['light_principal_direction'], parameters['free_rotation'], parameters['coefficients'], parameters['rho_spec'], parameters['tau_spec']
        losses.append(losses_values)
        steps.append('harmonic')
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
    scale = jax.numpy.max(jax.numpy.linalg.norm(points-center,axis=-1))
    light_distance = scale*3
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


def init_sh_light(shapes, mu, l_max):
    (n_pix, n_im, n_c) = shapes
    goal_function = lambda teta : jax.numpy.power(jax.numpy.cos(teta[:,None]), mu)
    coefficients, indices = spherical_harmonics.coefficients_from_colatitude(goal_function, l_max)
    free_rotation = jax.numpy.zeros((n_im,))
    return free_rotation, coefficients, indices, l_max


def init_grid(shapes, pixels, pixel_step):
    (n_pix, n_im, n_c) = shapes
    min_range, max_range = jax.numpy.min(pixels, axis=0), jax.numpy.max(pixels, axis=0)
    nx, ny = int((max_range[0]-min_range[0])/pixel_step), int((max_range[1]-min_range[1])/pixel_step)
    direction_grid = jax.numpy.zeros((nx, ny ,n_im, 3)).at[:,:,:,2].set(-1)
    intensity_grid = jax.numpy.ones((nx, ny ,n_im, 1))
    return direction_grid, intensity_grid, min_range, max_range


