import jax
import ledgetter.optim.gradient_descent as gradient_descent
import ledgetter.rendering.models as models

import scipy.interpolate
import functools
import ledgetter.utils.loading as loading


def get_ps_minimizer(optimizer, it):
    def minimize(points, images, validity_mask, light_local_direction, light_local_intensity, normals, rho, optimizer, it):
        model = {'light':'constant', 'renderers':['lambertian'], 'parameters'  : ['normals','rho']}
        light, renderer, projections = models.get_model(model)
        loss = models.get_loss(light, renderer, delta=0.01)
        data = { 'points':points, 'validity_mask': validity_mask[...,None,:], 'light_local_direction':light_local_direction, 'light_local_intensity':light_local_intensity}
        parameters = {'normals':normals, 'rho':rho}
        parameters, losses_values = gradient_descent.get_gradient_descent(optimizer, loss, it, projections=projections, extra = True, unroll=1)(parameters, data = data, images=images)
        rho, normals = parameters['rho'], parameters['normals']
        return rho, normals, losses_values
    vectorized = jax.numpy.vectorize(functools.partial(minimize, optimizer = optimizer, it = it), signature='(i),(c,l),(l),(l,i),(l,c),(i),(c)->(c),(i),(t)')
    return vectorized


def solve_ps(light_dict, points, normals, images, pixels, shapes, output, optimizer, mask, valid_options, iterations, chunck_number = 100):
    losses, steps = [], []
    (n_pix, n_im, n_c) = shapes
    
    rho, light = init_ps_parameters(light_dict, pixels)
    if 'PS' in iterations:
        it = iterations['PS']
        minimize = get_ps_minimizer(optimizer, it)
        losses_sum, n = jax.numpy.zeros((it,)), 0
        validity_mask_ps, rho_ps, normals_ps = jax.numpy.zeros((n_pix, n_im)), jax.numpy.zeros((n_pix, n_c)), jax.numpy.zeros((n_pix, 3))
        for i in range(chunck_number):
            chunck, chunck_n_pix = loading.chunck_index((i, chunck_number), n_pix)
            c_normals, c_points, c_rho, c_images, c_pixels = jax.tree_util.tree_map(lambda u : u[chunck], (normals, points, rho, images, pixels))
            c_mask = jax.numpy.zeros(mask.shape, dtype=bool).at[mask].set(jax.numpy.zeros(n_pix, dtype=bool).at[chunck].set(True))
            c_light_local_direction, c_light_local_intensity = light(c_points, c_pixels)
            c_validity_mask_ps = models.get_valid({'validity_maskers':['cast_shadow', 'morphology'], 'options' : valid_options}, (chunck_n_pix, n_im, n_c))(images=c_images, mask=c_mask, light=light, points = c_points, pixels = c_pixels)
            with jax.default_device(jax.devices("gpu")[0]):
                c_rho_ps, c_normals_ps, c_losses_values_ps = minimize(c_points, c_images, c_validity_mask_ps, c_light_local_direction, c_light_local_intensity, c_normals, c_rho)
            losses_sum, n = losses_sum + jax.numpy.nansum(c_losses_values_ps, axis=0), n+chunck_n_pix
            validity_mask_ps, rho_ps, normals_ps = jax.tree_util.tree_map(lambda u, v : u.at[chunck].set(v), (validity_mask_ps, rho_ps, normals_ps), (c_validity_mask_ps, c_rho_ps, c_normals_ps) )
            output(losses_sum[-1]/n, i, chunck_number-1)
        losses_values = losses_sum / n
        parameters = {'normals':normals_ps, 'rho':rho_ps}
        data = {'points' : points, 'pixels': pixels, 'validity_mask': validity_mask_ps[...,None,:]}
        losses.append(losses_values)
        steps.append('PS')
    if True:
        return parameters, data, losses, steps
    
def init_ps_parameters(light_dict, pixels):
    light = light_dict['light']
    rho = scipy.interpolate.NearestNDInterpolator(light_dict['pixels'], light_dict['rho'])(pixels)
    return rho, light
