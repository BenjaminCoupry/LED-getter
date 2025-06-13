import ledgetter.utils.chuncks as chuncks
import ledgetter.models.defaults as defaults
import ledgetter.models.values_generator as values_generator
import ledgetter.utils.functions as functions
import ledgetter.optim.gradient_descent as gradient_descent
import ledgetter.models.models as models
import ledgetter.utils.files as files
import functools
import jax
import pipeline.common as common

def get_ps_minimizer(iterations, model, loss, optimizer, projections):
    @functions.force_positional
    @functions.structured_return(['rho', 'normals', None])
    @functools.partial(jax.numpy.vectorize, signature='(c,l),(l),(l,i),(l,c),(i),(i),(c)->(c),(i),(t)')
    def minimize(images, validity_mask, light_local_direction, light_local_intensity, points, normals, rho):
        parameters, data = values_generator.split_parameters_data(locals(), model['parameters'], model['data'])
        parameters, losses_values = functools.partial(gradient_descent.get_gradient_descent(optimizer, loss, iterations, projections=projections, output=None, extra = True, unroll=1), data=data, images=images, validity_mask=validity_mask)(parameters)
        return parameters['rho'], parameters['normals'], losses_values
    return minimize

def estimate_ps(iterations, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict, delta=0.01, chunck_number = 100):
    model, validity_masker = defaults.get_default('PS', raycaster, scale)
    light, renderer, projections = models.get_model(model)
    loss = models.get_loss(light, renderer, delta=delta)
    minimize = get_ps_minimizer(iterations, model, loss, optimizer, projections)
    
    def treatement(state, chunck, images, **values):
        chunck_elems = images.shape[0]
        chunck_shapes = (chunck_elems,)+shapes[1:] 
        values = values_generator.merge_and_generate(values, light_dict['light_values'], model['parameters'] | model['data'], chunck_shapes, images, light = light_dict['light'])
        validity_mask = validity_masker(shapes = chunck_shapes, images=images, light=light, **values)
        with jax.default_device(jax.devices("gpu")[0]):
            (losses_values_ps, ), updated_values = minimize(images, validity_mask, **values)
        chuncked_values, atomic_values = chuncks.split_dict(values | updated_values | {'validity_mask' : validity_mask}, models.is_pixelwise)
        losses_sum, n, _ = state
        state, metric = (losses_sum + jax.numpy.nansum(losses_values_ps, axis=0), n+chunck_elems, atomic_values), losses_sum[-1]/n
        return chuncked_values, state, metric
    
    state = (jax.numpy.zeros((iterations,)), 0, None)
    chunckable_args, atomic_args = chuncks.split_dict(values | {'images' : images}, models.is_pixelwise)
    chuncked_values, state = chuncks.chunckwise_treatement(treatement, state, chunckable_args, atomic_args, chunck_number, output=output)
    validity_mask = chuncked_values['validity_mask']
    losses_sum, n, atomic_values = state
    values = {k:v for k, v in chuncked_values.items() if k not in {'validity_mask'}} | atomic_values
    losses_values = losses_sum / n

    light_dict = common.update_light_dict(light_dict, name='PS', losses_values=losses_values, model=model, values=values, light=light)

    return light_dict, validity_mask