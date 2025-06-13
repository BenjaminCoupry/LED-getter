import jax
import numpy
import ledgetter.models.models as models
import ledgetter.rendering.renderers as renderers
import matplotlib.pyplot as plt
import imageio.v3 as iio
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.utils.plots as plots
import os
import pathlib
import json
import ledgetter.utils.functions as functions
import ledgetter.utils.light_serialization as light_serialization


def export_images(path, light_dict, images, validity_mask, mask, light_names):
    os.makedirs(path, exist_ok=True)
    light_direction, light_intensity = light_dict['light'](**light_dict['light_values'])
    rendered_images = models.get_grouped_renderer(light_dict['model']['renderers'])(light_direction, light_intensity, **light_dict['light_values'])
    rendered_image = jax.tree_util.tree_reduce(lambda x, y : x + y, rendered_images)
    delta = jax.numpy.abs(images-rendered_image)
    scale = numpy.quantile(images, 0.98)
    for im in range(images.shape[-1]):
        name = light_names[im]
        os.makedirs(os.path.join(path, name), exist_ok=True)
        ref_image = vector_tools.build_masked(mask, images[:,:,im]/scale)
        iio.imwrite(os.path.join(path, name,'reference.png'),numpy.uint8(255.0*numpy.clip(ref_image,0,1)))
        simulated_image = vector_tools.build_masked(mask, rendered_image[:,:,im]/scale)
        iio.imwrite(os.path.join(path, name, 'simulated.png'),numpy.uint8(255.0*numpy.clip(simulated_image,0,1)))
        delta_image = vector_tools.build_masked(mask, delta[:,:,im]/scale)
        iio.imwrite(os.path.join(path, name, 'delta.png'),numpy.uint8(255.0*numpy.clip(delta_image,0,1)))
        validity_image = vector_tools.build_masked(mask, validity_mask[:,im])
        iio.imwrite(os.path.join(path, name, 'validity.png'), validity_image)
        for renderer in rendered_images:
            os.makedirs(os.path.join(path, name, 'renderers'), exist_ok=True)
            simulated_image_renderer = vector_tools.build_masked(mask, rendered_images[renderer][:,:,im]/scale)
            iio.imwrite(os.path.join(path, name, 'renderers',f'{renderer}.png'),numpy.uint8(255.0*numpy.clip(simulated_image_renderer,0,1)))

def export_lightmaps(path, light_dict, mask, light_names):
    os.makedirs(path, exist_ok=True)
    light_direction, light_intensity = light_dict['light'](**light_dict['light_values'])
    max_intensity = numpy.quantile(light_intensity, 0.9)
    for im in range(light_direction.shape[-2]):
        name = light_names[im]
        os.makedirs(os.path.join(path, name), exist_ok=True)
        simulated_direction = vector_tools.build_masked(mask, light_direction[:,im,:])
        simulated_intensity = vector_tools.build_masked(mask, light_intensity[:, im, 0] if light_intensity.shape[-1]==1 else light_intensity[:, im, :])
        iio.imwrite(os.path.join(path, name, 'direction.png'),numpy.uint8(0.5*(simulated_direction*numpy.asarray([1,-1,-1])+1)*255))
        iio.imwrite(os.path.join(path, name,'intensity.png'),numpy.uint8(numpy.clip(simulated_intensity / max_intensity,0,1)*255))


def export_values(path, light_dict, mask, validity_mask):
    os.makedirs(path, exist_ok=True)
    light_values = light_dict['light_values']
    numpy.savez(os.path.join(path, 'values.npz'), mask = mask, validity_mask=validity_mask, **light_values)
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    if 'normals' in light_values :
        normalmap = vector_tools.build_masked(mask, light_values['normals'])
        iio.imwrite(os.path.join(path, 'images', 'geometry_normals.png'),numpy.uint8(0.5*(normalmap*numpy.asarray([1,-1,-1])+1)*255))
    if 'points' in light_values:
        zscale = 1- ((light_values['points'][:,-1] - numpy.min(light_values['points'][:,-1])) / (numpy.max(light_values['points'][:,-1]) - numpy.min(light_values['points'][:,-1])))
        zmap = vector_tools.build_masked(mask, zscale)
        iio.imwrite(os.path.join(path, 'images', 'geometry_zmap.png'),numpy.uint8((numpy.clip(zmap,0,1))*255))
    if 'rho' in light_values:
        albedomap = vector_tools.build_masked(mask, light_values['rho'])
        iio.imwrite(os.path.join(path, 'images', 'albedomap.png'),numpy.uint8(numpy.clip(albedomap/numpy.quantile(light_values['rho'], 0.99),0,1)*255))
    if 'rho_spec' in light_values:
        albedospecmap = vector_tools.build_masked(mask, light_values['rho_spec'])
        iio.imwrite(os.path.join(path, 'images', 'albedospecmap.png'),numpy.uint8(numpy.clip(albedospecmap/numpy.quantile(light_values['rho_spec'], 0.99),0,1)*255))

def export_losses(path, light_dict):
    losses = light_dict['losses']
    os.makedirs(path, exist_ok=True)
    names, losses_values = zip(*losses)
    numpy.savez(os.path.join(path, 'losses.npz'), steps_order=numpy.asarray(names, dtype=object), **{k: v for k,v in losses})
    loss_plot = plots.plot_losses(losses_values, names)
    loss_plot.write_html(os.path.join(path, 'loss_plot.html'))

def export_model(path, light_dict):
    model = light_dict['model']
    os.makedirs(path, exist_ok=True)
    def replace_sets_with_lists(obj):
        if isinstance(obj, dict):
            return {k: replace_sets_with_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, set):
            return [replace_sets_with_lists(elem) for elem in obj]
        elif isinstance(obj, tuple):
            return tuple(replace_sets_with_lists(elem) for elem in obj)
        else:
            return obj
    with open(os.path.join(path, 'model.json'), 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=2, default=lambda o: list(o) if isinstance(o, set) else o)

def export_light(path, light_dict, light_names):
    os.makedirs(os.path.join(path), exist_ok=True)
    light_direction, light_intensity = light_dict['light'](**light_dict['light_values'])
    L0, Phi = vector_tools.norm_vector(jax.numpy.mean(light_direction, axis=0))[1], numpy.mean(light_intensity, axis=0)
    str_L0, str_Phi = numpy.asarray(L0).astype(str), numpy.asarray(Phi).astype(str)
    names_array = numpy.asarray(light_names,dtype=str)[:,None]
    XL0 = numpy.concatenate([names_array,str_L0],axis=-1)
    XPhi = numpy.concatenate([names_array,str_Phi],axis=-1)
    numpy.savetxt(os.path.join(path,'light_direction.lp'), XL0, fmt = '%s', header = str(len(light_names)), delimiter = ' ', comments='')
    numpy.savetxt(os.path.join(path,'light_intensity.lp'), XPhi, fmt = '%s', header = str(len(light_names)), delimiter = ' ', comments='')
    try:
        serialized = light_serialization.serialize_light(light_dict['light'], light_dict['light_values'])
        with open(os.path.join(path,'light_function.jax'), "wb") as f:
            f.write(serialized)
    except Exception as e :
        print(f"Serialization exception : {e}, skipping light serialization")

def export_misc(path, light_dict, validity_mask, mask, images, light_names):
    os.makedirs(path, exist_ok=True)
    light_direction, light_intensity = light_dict['light'](**light_dict['light_values'])
    rendered_images = models.get_grouped_renderer(light_dict['model']['renderers'])(light_direction, light_intensity, **light_dict['light_values'])
    rendered_image = jax.tree_util.tree_reduce(lambda x, y : x + y, rendered_images)
    delta = jax.numpy.abs(images-rendered_image)
    iio.imwrite(os.path.join(path,'mask.png'), mask)
    validitymap = vector_tools.build_masked(mask, jax.numpy.mean(validity_mask, axis=-1))
    iio.imwrite(os.path.join(path, 'mean_validity.png'), numpy.uint8(numpy.clip(validitymap,0,1)*255.0))
    scale = numpy.quantile(images, 0.98)
    mean_delta_image = vector_tools.build_masked(mask, jax.numpy.mean(delta,axis=-1)/scale)
    iio.imwrite(os.path.join(path, 'mean_delta.png'),numpy.uint8(255.0*numpy.clip(mean_delta_image,0,1)))
    max_delta_image = vector_tools.build_masked(mask, jax.numpy.max(delta,axis=-1)/scale)
    iio.imwrite(os.path.join(path, 'max_delta.png'),numpy.uint8(255.0*numpy.clip(max_delta_image,0,1)))
    plot_function = plots.get_plot_light(light_dict['model']['light'])
    if plot_function is not None:
        light_plot = functions.filter_args(plot_function)(**light_dict['light_values'], mask=mask, names = light_names)
        light_plot.write_html(os.path.join(path, 'light_plot.html'))

def export_results(out_path, validity_mask,light_dict, mask, images, light_names):
    export_images(os.path.join(out_path,'images'), light_dict, images, validity_mask, mask, light_names)
    export_lightmaps(os.path.join(out_path,'lightmaps'), light_dict, mask, light_names)
    export_values(os.path.join(out_path,'values'), light_dict, mask, validity_mask)
    export_losses(os.path.join(out_path,'losses'), light_dict)
    export_model(os.path.join(out_path), light_dict)
    export_light(os.path.join(out_path,'light'), light_dict, light_names)
    export_misc(os.path.join(out_path,'misc'), light_dict, validity_mask, mask, images, light_names)