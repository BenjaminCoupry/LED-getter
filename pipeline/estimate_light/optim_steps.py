import jax
import ledgetter.optim.gradient_descent as gradient_descent
import ledgetter.rendering.models as models

import ledgetter.rendering.lights as lights
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.rendering.validity as validity
import scipy.interpolate
import functools
import ledgetter.utils.loading as loading


def solve_ps(light_values, points, normals, images, pixels, shapes, output, optimizer, mask, valid_options, iterations, chunck_number = 5):
    losses, steps = [], []
    (n_pix, n_im, n_c) = shapes
    light_local_direction, light_local_intensity, rho = init_ps_parameters(light_values, points, pixels)
    if 'PS' in iterations:
        it = iterations['PS']
        model = {'light':'constant', 'renderers':['lambertian'], 'parameters'  : ['normals','rho']}
        light, renderer, projections = models.get_model(model)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values={ 'points':points, 'light_local_direction':light_local_direction, 'light_local_intensity':light_local_intensity}, points=points)
        loss = models.get_loss(light, renderer, delta=0.01)
        @functools.partial(jax.numpy.vectorize, signature='(i),(c,l),(l),(l,i),(l,c),(i),(c)->(c),(i),(t)')
        def minimize(points, images, validity_mask, light_local_direction, light_local_intensity, normals, rho):
            data = { 'points':points, 'validity_mask': validity_mask[...,None,:], 'light_local_direction':light_local_direction, 'light_local_intensity':light_local_intensity}
            parameters = {'normals':normals, 'rho':rho}
            parameters, losses_values = gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, extra = True, unroll=1)(parameters, data = data, images=images)
            rho, normals = parameters['rho'], parameters['normals']
            return rho, normals, losses_values
        rho_all, normals_all, losses_sum, n = jax.numpy.zeros((n_pix, n_c)), jax.numpy.zeros((n_pix, 3)), jax.numpy.zeros((it,)), 0
        for i in range(chunck_number):
            chunck= loading.chunck_index((i, chunck_number), n_pix)
            args = jax.tree_map(lambda a : a[chunck], (points, images, validity_mask, light_local_direction, light_local_intensity, normals, rho))
            with jax.default_device(jax.devices("gpu")[0]):
                rho_chunck, normals_chunck, losses_values_chunck = minimize(*args)
            rho_all, normals_all, losses_sum, n = rho_all.at[chunck].set(rho_chunck), normals_all.at[chunck].set(normals_chunck), losses_sum + jax.numpy.sum(losses_values_chunck, axis=0), n+rho_chunck.shape[0]
            output(losses_sum[-1]/n, i, chunck_number-1)
        losses_values = losses_sum / n
        parameters = {'normals':normals_all, 'rho':rho_all}
        data = {'points' : points, 'pixels': pixels, 'validity_mask': validity_mask[...,None,:], 'light_local_direction':light_local_direction, 'light_local_intensity':light_local_intensity}
        losses.append(losses_values)
        steps.append('PS')
    if True:
        return parameters, data, losses, steps


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
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
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
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
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
        model = {'light':'rail', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
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
        model = {'light':'punctual', 'renderers':['lambertian'], 'parameters' : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
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
        model = {'light':'LED', 'renderers':['lambertian'], 'parameters'  : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        rho, light_locations, light_power, light_principal_direction, mu = parameters['rho'], parameters['light_locations'], parameters['light_power'], parameters['light_principal_direction'], parameters['mu']
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
        losses.append(losses_values)
        steps.append('LED')
    if 'specular' not in iterations:
        return parameters, data, losses, steps
    
    rho_spec, tau_spec = init_specular(shapes)
    if 'specular' in iterations:
        it = iterations['specular']
        data = {'normals':normals, 'points':points, 'validity_mask':validity_mask[...,None,:], 'pixels':pixels}
        parameters = {'rho': rho, 'light_locations': light_locations, 'light_power': light_power, 'light_principal_direction': light_principal_direction, 'mu': mu, 'rho_spec': rho_spec, 'tau_spec': tau_spec}
        model ={'light':'LED', 'renderers':['lambertian','specular'], 'parameters'  : list(parameters.keys())}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, output=output, extra = True, unroll=1))(parameters, data = data, images=images)
        validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, shapes)(images=images, mask=mask, light=light, values=(data | parameters), points=points)
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


def init_grid(shapes, pixels, pixel_step):
    (n_pix, n_im, n_c) = shapes
    min_range, max_range = jax.numpy.min(pixels, axis=0), jax.numpy.max(pixels, axis=0)
    nx, ny = int((max_range[0]-min_range[0])/pixel_step), int((max_range[1]-min_range[1])/pixel_step)
    direction_grid = jax.numpy.zeros((nx, ny ,n_im, 3)).at[:,:,:,2].set(-1)
    intensity_grid = jax.numpy.ones((nx, ny ,n_im, 1))
    return direction_grid, intensity_grid, min_range, max_range


def init_ps_parameters(light_values, points, pixels):
    light_local_values = {k: v for k, v in light_values.items() if k not in {'points', 'pixels','validity_mask'}} | {'points' : points, 'pixels': pixels}
    light_model = models.model_from_parameters(light_local_values, {})
    light, _, _ = models.get_model(light_model)
    light_local_direction, light_local_intensity = light(light_local_values)
    rho = scipy.interpolate.NearestNDInterpolator(light_values['pixels'], light_values['rho'])(pixels)
    return light_local_direction, light_local_intensity, rho

    