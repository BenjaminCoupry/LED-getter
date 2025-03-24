import jax
import ledgetter.optim.gradient_descent as gradient_descent
import ledgetter.rendering.models as models



def get_default_iterations():
    iterations = {'directional' : 400, 'rail':300, 'punctual':4000, 'LED' : 4000, 'specular':2000}
    return iterations

def estimate_light(points, normals, images, shapes, output, optimizer, mask, validity_mask, iterations):
    (n_pix, n_im, n_c) = shapes
    data = (normals, points, images, validity_mask[...,None,:])
    losses = []

    rho = init_rho(images)
    light_directions, light_power = init_directional_light(shapes)
    if 'directional' in iterations:
        parameters_0 = (rho, light_directions, light_power)
        model, projections = models.get_directional_model()
        parameters_0, losses_0 = jax.jit(gradient_descent.get_gradient_descent(optimizer, models.get_loss(model), iterations['directional'], projections=projections, output=output, extra = True, unroll=1))(parameters_0, data = data)
        rho, light_directions, light_power = parameters_0
        losses.append(losses_0)

    light_distance, center = init_light_distance(points)
    if 'rail' in iterations:
        parameters_1 = (rho, light_distance)
        model, projections = models.get_rail_punctual_model(light_directions, light_power, center)
        parameters_1, losses_1 = jax.jit(gradient_descent.get_gradient_descent(optimizer, models.get_loss(model), iterations['rail'], projections=projections, output=output, extra = True, unroll=1))(parameters_1, data = data)
        rho, light_distance = parameters_1
        losses.append(losses_1)

    light_locations, light_power = init_punctual_light(points, light_distance, light_directions, light_power)
    if 'punctual' in iterations:
        parameters_2 = (rho, light_locations, light_power)
        model, projections = models.get_isotropic_punctual_model()
        parameters_2, losses_2 = jax.jit(gradient_descent.get_gradient_descent(optimizer, models.get_loss(model), iterations['punctual'], projections=projections, output=output, extra = True, unroll=1))(parameters_2, data = data)
        rho, light_locations, light_power = parameters_2
        losses.append(losses_2)

    light_principal_direction, mu = init_led_light(shapes, points, light_locations)
    if 'LED' in iterations:
        parameters_3 = (rho, light_locations, light_power, light_principal_direction, mu)
        model, projections = models.get_led_model()
        parameters_3, losses_3 = jax.jit(gradient_descent.get_gradient_descent(optimizer, models.get_loss(model), iterations['LED'], projections=projections, output=output, extra = True, unroll=1))(parameters_3, data = data)
        rho, light_locations, light_power, light_principal_direction, mu = parameters_3
        losses.append(losses_3)

    rho_spec, tau_spec = init_specular(shapes)
    if 'specular' in iterations:
        parameters_4 = (rho, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec)
        model, projections = models.get_specular_led_model()
        parameters_4, losses_4 = jax.jit(gradient_descent.get_gradient_descent(optimizer, models.get_loss(model), iterations['specular'], projections=projections, output=output, extra = True, unroll=1))(parameters_4, data = data)
        rho, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec = parameters_4
        losses.append(losses_4)
    

    parameters = rho, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec
    return parameters, losses

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
