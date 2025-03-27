import jax
import ledgetter.optim.gradient_descent as gradient_descent
import ledgetter.rendering.models as models



def get_default_iterations():
    iterations = {'directional' : 400, 'rail':300, 'punctual':4000, 'LED' : 4000, 'specular':3000}
    return iterations
#updater le dictionnaire des params plutot que de tout extraire a chaque step
def estimate_light(points, normals, images, shapes, output, optimizer, mask, validity_mask, iterations):
    data = {'normals':normals, 'points':points, 'images':images, 'validity_mask':validity_mask[...,None,:]}
    losses = []

    rho = init_rho(images)
    light_directions, light_power = init_directional_light(shapes)
    if 'directional' in iterations:
        parameters_0 = {'rho' : rho, 'light_directions' : light_directions, 'light_power':light_power}
        projections = models.get_projections(parameters_0)
        loss = models.get_loss({'light':'directional', 'renderers':['lambertian']}, delta=0.01)
        parameters_0, losses_0 = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, iterations['directional'], projections=projections, output=output, extra = True, unroll=1))(parameters_0, data = data)
        rho, light_directions, light_power = parameters_0['rho'], parameters_0['light_directions'], parameters_0['light_power']
        losses.append(losses_0)

    light_distance, center = init_light_distance(points)
    if 'rail' in iterations:
        parameters_1 = {'rho': rho, 'light_distance': light_distance}
        projections = models.get_projections(parameters_1)
        loss = models.get_loss({'light':'rail', 'renderers':['lambertian']}, delta=0.01)
        extra_data = {'center':center, 'light_directions' : light_directions, 'light_power': light_power}
        parameters_1, losses_1 = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, iterations['rail'], projections=projections, output=output, extra = True, unroll=1))(parameters_1, data =  (data | extra_data))
        rho, light_distance = parameters_1['rho'], parameters_1['light_distance']
        losses.append(losses_1)

    light_locations, light_power = init_punctual_light(points, light_distance, light_directions, light_power)
    if 'punctual' in iterations:
        parameters_2 = {'rho': rho, 'light_locations': light_locations, 'light_power': light_power}
        projections = models.get_projections(parameters_2)
        loss = models.get_loss({'light':'punctual', 'renderers':['lambertian']}, delta=0.01)
        parameters_2, losses_2 = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, iterations['punctual'], projections=projections, output=output, extra = True, unroll=1))(parameters_2, data = data)
        rho, light_locations, light_power = parameters_2['rho'], parameters_2['light_locations'], parameters_2['light_power']
        losses.append(losses_2)

    light_principal_direction, mu = init_led_light(shapes, points, light_locations)
    if 'LED' in iterations:
        parameters_3 = {
            'rho': rho,
            'light_locations': light_locations,
            'light_power': light_power,
            'light_principal_direction': light_principal_direction,
            'mu': mu
        }
        projections = models.get_projections(parameters_3)
        loss = models.get_loss({'light':'LED', 'renderers':['lambertian']}, delta=0.01)
        parameters_3, losses_3 = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, iterations['LED'], projections=projections, output=output, extra = True, unroll=1))(parameters_3, data = data)
        rho, light_locations, light_power, light_principal_direction, mu = (
            parameters_3['rho'],
            parameters_3['light_locations'],
            parameters_3['light_power'],
            parameters_3['light_principal_direction'],
            parameters_3['mu']
        )
        losses.append(losses_3)

    rho_spec, tau_spec = init_specular(shapes)
    if 'specular' in iterations:
        parameters_4 = {
            'rho': rho,
            'light_locations': light_locations,
            'light_power': light_power,
            'light_principal_direction': light_principal_direction,
            'mu': mu,
            'rho_spec': rho_spec,
            'tau_spec': tau_spec
        }
        projections = models.get_projections(parameters_4)
        loss = models.get_loss({'light':'LED', 'renderers':['lambertian','specular']}, delta=0.01)
        parameters_4, losses_4 = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, iterations['specular'], projections=projections, output=output, extra = True, unroll=1))(parameters_4, data = data)
        rho, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec = (
            parameters_4['rho'],
            parameters_4['light_locations'],
            parameters_4['light_power'],
            parameters_4['light_principal_direction'],
            parameters_4['mu'],
            parameters_4['rho_spec'],
            parameters_4['tau_spec']
        )
        losses.append(losses_4)

        parameters = {
        'rho': rho,
        'light_locations': light_locations,
        'light_power': light_power,
        'light_principal_direction': light_principal_direction,
        'mu': mu,
        'rho_spec': rho_spec,
        'tau_spec': tau_spec
    }
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
