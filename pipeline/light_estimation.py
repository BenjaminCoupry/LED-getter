import glob
import pipeline.common as common
import pipeline.outputs as outputs
import optax
import numpy
import jax

import ledgetter.models.defaults as defaults
import ledgetter.models.values_generator as values_generator
import ledgetter.optim.gradient_descent as gradient_descent
import ledgetter.models.models as models
import ledgetter.utils.files as files
import functools



def estimate_light(iterations, pattern, values, images, mask, raycaster, shapes, output, optimizer, scale, light_dict, delta=0.01):
    model, validity_masker = defaults.get_default(pattern, raycaster, scale)
    values = values_generator.merge_and_generate(values, light_dict['light_values'], model['parameters'] | model['data'], shapes, images)
    parameters, data = values_generator.split_parameters_data(values, model['parameters'], model['data'])
    validity_mask = validity_masker(shapes = shapes, images=images, mask=mask, light=light_dict['light'], **light_dict['light_values'])
    light, renderer, projections = models.get_model(model)
    loss = models.get_loss(light, renderer, delta=delta)
    #parameters, losses_values = jax.jit(functools.partial(gradient_descent.get_gradient_descent(optimizer, loss, iterations, projections=projections, output=output, extra = True, unroll=1), images=images, validity_mask=validity_mask))(parameters, data=data, images=images, validity_mask=validity_mask)
    parameters, losses_values = jax.jit(gradient_descent.get_gradient_descent(optimizer, loss, iterations, projections=projections, output=output, extra = True, unroll=1))(parameters, data=data, images=images, validity_mask=validity_mask)
    light_dict = common.update_light_dict(light_dict, name=pattern, losses_values=losses_values, model=model, values=parameters | data, light=light)
    return light_dict, validity_mask