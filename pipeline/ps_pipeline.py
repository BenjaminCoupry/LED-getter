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


def solve_ps(light_values, points, normals, images, pixels, shapes, output, optimizer, mask, valid_options, iterations, chunck_number = 5):
    losses, steps = [], []
    (n_pix, n_im, n_c) = shapes
    local_values, light = init_ps_parameters(light_values, points, pixels, normals)
    if 'PS' in iterations:
        it = iterations['PS']
        minimize = get_ps_minimizer(optimizer, it)
        validity_mask_all, rho_all, normals_all, losses_sum, n = jax.numpy.zeros((n_pix, n_im)), jax.numpy.zeros((n_pix, n_c)), jax.numpy.zeros((n_pix, 3)), jax.numpy.zeros((it,)), 0
        for i in range(chunck_number):
            chunck, chunck_n_pix = loading.chunck_index((i, chunck_number), n_pix)
            chuncked_values = {k:(local_values[k][chunck] if models.is_pixelwise(k) else local_values[k]) for k in local_values}
            chuncked_light_local_direction, chuncked_light_local_intensity = light(chuncked_values)
            chuncked_images, chuncked_mask = images[chunck], jax.numpy.zeros(mask.shape, dtype=bool).at[mask].set(jax.numpy.zeros(n_pix, dtype=bool).at[chunck].set(True))
            chuncked_validity_mask = models.get_valid({'validity_maskers':['intensity', 'cast_shadow', 'morphology'], 'options' : valid_options}, (chunck_n_pix, n_im, n_c))(images=chuncked_images, mask=chuncked_mask, light=light, values=chuncked_values, points=chuncked_values['points'])
            with jax.default_device(jax.devices("gpu")[0]):
                rho_chunck, normals_chunck, losses_values_chunck = minimize(chuncked_values['points'], chuncked_images, chuncked_validity_mask, chuncked_light_local_direction, chuncked_light_local_intensity, chuncked_values['normals'], chuncked_values['rho'])
            validity_mask_all, rho_all, normals_all, losses_sum, n = validity_mask_all.at[chunck].set(chuncked_validity_mask), rho_all.at[chunck].set(rho_chunck), normals_all.at[chunck].set(normals_chunck), losses_sum + jax.numpy.nansum(losses_values_chunck, axis=0), n+chunck_n_pix
            output(losses_sum[-1]/n, i, chunck_number-1)
        losses_values = losses_sum / n
        parameters = {'normals':normals_all, 'rho':rho_all}
        data = {'points' : points, 'pixels': pixels, 'validity_mask': validity_mask_all[...,None,:]}
        losses.append(losses_values)
        steps.append('PS')
    if True:
        return parameters, data, losses, steps
    
def init_ps_parameters(light_values, points, pixels, normals):
    rho = scipy.interpolate.NearestNDInterpolator(light_values['pixels'], light_values['rho'])(pixels)
    local_values = {k: v for k, v in light_values.items() if k not in {'points', 'pixels', 'rho', 'normals' }} | {'points' : points, 'pixels': pixels, 'rho':rho, 'normals' : normals}
    light_model = models.model_from_parameters(local_values, {})
    light = models.get_light(light_model['light'])
    return local_values, light
