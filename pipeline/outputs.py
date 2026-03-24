import jax
import numpy
import ledgetter.models.models as models
import imageio.v3 as iio
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.utils.plots as plots
import os
import json
import ledgetter.utils.functions as functions
import ledgetter.utils.light_serialization as light_serialization
import scipy.interpolate
import scipy.spatial


def prepare_filter_interpolators(filters, mask):
    full_coordinates = jax.numpy.argwhere(mask)
    unique_filters = jax.numpy.max(filters)
    filter_masks = [filters == f for f in range(unique_filters + 1)]
    hit_masks = [jax.numpy.any(filter_mask, axis=-1) for filter_mask in filter_masks]
    hit_indexs = [jax.numpy.argmax(filter_mask, axis=-1) for filter_mask in filter_masks]
    triangulations = [scipy.spatial.Delaunay(full_coordinates[hit_mask]) for hit_mask in hit_masks]
    selectors = [lambda channel_values, hit_index=hit_index, hit_mask=hit_mask : jax.numpy.take_along_axis(channel_values, hit_index[..., None], axis=-1)[hit_mask, 0] for (hit_index, hit_mask) in zip(hit_indexs, hit_masks)]
    channel_interpolators = [lambda channel_values, triangulation=triangulation, selector=selector : scipy.interpolate.LinearNDInterpolator(triangulation, selector(channel_values))(full_coordinates) for (triangulation, selector) in zip(triangulations, selectors)]
    interpolator = lambda channel_values : jax.numpy.stack([i(channel_values) for i in channel_interpolators], axis=-1)
    return interpolator


def export_filters(path, label, value, mask, maximum):
    flat = vector_tools.build_masked(mask, value)
    if value.shape[-1] == 1:
        iio.imwrite(os.path.join(path, f'{label}.png'),numpy.uint8(numpy.clip(flat[...,0] / maximum, 0, 1) * 255))
    if value.shape[-1] == 3:
        iio.imwrite(os.path.join(path, f'{label}.png'),numpy.uint8(numpy.clip(flat / maximum, 0, 1) * 255))
    if value.shape[-1] > 1:
        os.makedirs(os.path.join(path, label), exist_ok=True)
        for c in range(value.shape[-1]):
            iio.imwrite(os.path.join(path, label, f'filter_{c:05d}.png'), numpy.uint8(numpy.clip(flat[..., c] / maximum, 0, 1) * 255))

def export_images(path, light_dict, images, validity_mask, mask, light_names, shapes):
    os.makedirs(path, exist_ok=True)
    light_direction, light_intensity = light_dict['light'](**light_dict['light_values'] | {'shapes':shapes})
    filters = light_dict['light_values']['filters']
    rendered_brdfs = models.get_grouped_brdf(light_dict['model']['brdfs'])(light_direction, light_intensity, **light_dict['light_values'])
    rendered_summed_brdfs = jax.tree_util.tree_reduce(lambda x, y : x + y, rendered_brdfs)
    rendered_image = models.filters_to_channel(rendered_summed_brdfs, filters)
    delta = jax.numpy.abs(images-rendered_image)
    scale = numpy.quantile(images, 0.98)
    for im in range(images.shape[-1]):
        name = light_names[im]
        os.makedirs(os.path.join(path, name), exist_ok=True)
        export_filters(os.path.join(path, name), 'reference', models.channel_to_filters(images[:,:,im], filters), mask, scale)
        export_filters(os.path.join(path, name), 'simulated', models.channel_to_filters(rendered_image[:,:,im], filters), mask, scale)
        export_filters(os.path.join(path, name), 'delta', models.channel_to_filters(delta[:,:,im], filters), mask, scale)
        validity_image = vector_tools.build_masked(mask, validity_mask[:,im])
        iio.imwrite(os.path.join(path, name, 'validity.png'), validity_image)
        os.makedirs(os.path.join(path, name, 'brdfs'), exist_ok=True)
        for brdf in rendered_brdfs:
            export_filters(os.path.join(path, name, 'brdfs'), brdf, rendered_brdfs[brdf][:,:,im], mask, scale)

def export_lightmaps(path, light_dict, mask, light_names, shapes):
    os.makedirs(path, exist_ok=True)
    light_direction, light_intensity = light_dict['light'](**light_dict['light_values']| {'shapes':shapes})
    max_intensity = numpy.quantile(light_intensity, 0.9)
    for im in range(light_direction.shape[-2]):
        name = light_names[im]
        os.makedirs(os.path.join(path, name), exist_ok=True)
        simulated_direction = vector_tools.build_masked(mask, light_direction[:,im,:])
        iio.imwrite(os.path.join(path, name, 'direction.png'),vector_tools.r3_to_rgb(simulated_direction))
        export_filters(os.path.join(path, name), 'intensity', light_intensity[:, im, :], mask, max_intensity)


def export_values(path, light_dict, mask, validity_mask):
    os.makedirs(path, exist_ok=True)
    light_values = light_dict['light_values']
    filters = light_values['filters']
    numpy.savez(os.path.join(path, 'values.npz'), mask = mask, validity_mask=validity_mask, **light_values)
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    if 'normals' in light_values :
        normalmap = vector_tools.build_masked(mask, light_values['normals'])
        iio.imwrite(os.path.join(path, 'images', 'geometry_normals.png'), vector_tools.r3_to_rgb(normalmap))
    if 'points' in light_values:
        zscale = 1- ((light_values['points'][:,-1] - numpy.min(light_values['points'][:,-1])) / (numpy.max(light_values['points'][:,-1]) - numpy.min(light_values['points'][:,-1])))
        zmap = vector_tools.build_masked(mask, zscale)
        iio.imwrite(os.path.join(path, 'images', 'geometry_zmap.png'),numpy.uint8((numpy.clip(zmap,0,1))*255))
    if 'rho' in light_values:
        max_rho = numpy.quantile(light_values['rho'][jax.numpy.isfinite(light_values['rho'])], 0.99)
        export_filters(os.path.join(path, 'images'), 'albedo', light_values['rho'], mask, max_rho)
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

def export_light(path, light_dict, validity_mask, light_names, shapes):
    os.makedirs(os.path.join(path), exist_ok=True)
    light_direction, light_intensity = light_dict['light'](**light_dict['light_values'] | {'shapes':shapes})
    L0, Phi = vector_tools.norm_vector(jax.numpy.mean(light_direction, where=validity_mask[:,:,None], axis=0))[1], numpy.mean(light_intensity, where=validity_mask[:,:,None], axis=0)
    str_L0, str_Phi = numpy.asarray(L0).astype(str), numpy.asarray(Phi).astype(str)
    names_array = numpy.asarray(light_names,dtype=str)[:,None]
    XL0 = numpy.concatenate([names_array,str_L0],axis=-1)
    XPhi = numpy.concatenate([names_array,str_Phi],axis=-1)
    numpy.savetxt(os.path.join(path,'light_direction.lp'), XL0, fmt = '%s', header = str(len(light_names)), delimiter = ' ', comments='')
    numpy.savetxt(os.path.join(path,'light_intensity.lp'), XPhi, fmt = '%s', header = str(len(light_names)), delimiter = ' ', comments='')
    try:
        serialized = light_serialization.serialize_light(light_dict['light'], light_dict['light_values'], shapes)
        with open(os.path.join(path,'light_function.jax'), "wb") as f:
            f.write(serialized)
    except Exception as e :
        print(f"Serialization exception : {e}, skipping light serialization")

def export_misc(path, light_dict, validity_mask, mask, images, light_names, pose, shapes):
    os.makedirs(path, exist_ok=True)
    filters = light_dict['light_values']['filters']
    light_direction, light_intensity = light_dict['light'](**light_dict['light_values'] | {'shapes':shapes})
    rendered_brdfs = models.get_grouped_brdf(light_dict['model']['brdfs'])(light_direction, light_intensity, **light_dict['light_values'])
    rendered_summed_brdfs = jax.tree_util.tree_reduce(lambda x, y : x + y, rendered_brdfs)
    rendered_image = models.filters_to_channel(rendered_summed_brdfs, filters)
    delta = jax.numpy.abs(images-rendered_image)
    scale = numpy.quantile(images, 0.98)
    iio.imwrite(os.path.join(path,'mask.png'), mask)
    validitymap = vector_tools.build_masked(mask, jax.numpy.mean(validity_mask, axis=-1))
    iio.imwrite(os.path.join(path, 'mean_validity.png'), numpy.uint8(numpy.clip(validitymap,0,1)*255.0))
    export_filters(path, 'min_delta', models.channel_to_filters(jax.numpy.min(delta, axis=-1), filters), mask, scale)
    export_filters(path, 'max_delta', models.channel_to_filters(jax.numpy.max(delta, axis=-1), filters), mask, scale)
    export_filters(path, 'mean_delta', models.channel_to_filters(jax.numpy.mean(delta, axis=-1), filters), mask, scale)
    export_filters(path, 'median_delta', models.channel_to_filters(jax.numpy.median(delta, axis=-1), filters), mask, scale)
    plot_function = plots.get_plot_light(light_dict['model']['light'])
    if plot_function is not None:
        light_plot = functions.filter_args(plot_function)(**light_dict['light_values'], mask=mask, names = light_names)
        light_plot.write_html(os.path.join(path, 'light_plot.html'))
    if pose is not None:
        func = lambda p : p.tolist() if isinstance(p, jax.numpy.ndarray) else p
        pose_list = jax.tree_util.tree_map(func, pose)
        with open(os.path.join(path, 'pose.json'), "w") as f:
            json.dump(pose_list, f, indent=2)

def export_results(out_path, shapes, validity_mask, light_dict, mask, images, light_names, skip=None, pose=None):
    if skip is None or 'images' not in skip:
        export_images(os.path.join(out_path,'images'), light_dict, images, validity_mask, mask, light_names, shapes)
    if skip is None or 'lightmaps' not in skip:
        export_lightmaps(os.path.join(out_path,'lightmaps'), light_dict, mask, light_names, shapes)
    if skip is None or 'values' not in skip:
        export_values(os.path.join(out_path,'values'), light_dict, mask, validity_mask)
    if skip is None or 'losses' not in skip:
        export_losses(os.path.join(out_path,'losses'), light_dict)
    if skip is None or 'model' not in skip:
        export_model(os.path.join(out_path), light_dict)
    if skip is None or 'light' not in skip:
        export_light(os.path.join(out_path,'light'), light_dict, validity_mask, light_names, shapes)
    if skip is None or 'misc' not in skip:
        export_misc(os.path.join(out_path,'misc'), light_dict, validity_mask, mask, images, light_names, pose, shapes)