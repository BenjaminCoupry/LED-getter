import jax
import numpy
import ledgetter.rendering.models as models
import ledgetter.rendering.renderers as renderers
import matplotlib.pyplot as plt
import imageio.v3 as iio
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.utils.plots as plots
import os
import pathlib



def export_images(out_path, rendered_images, ps_images_paths, images, mask, values):
    os.makedirs(os.path.join(out_path,'images'), exist_ok=True)
    rendered_image = jax.tree_util.tree_reduce(lambda x, y : x + y, rendered_images)
    scale = numpy.quantile(images, 0.98)
    for im in range(images.shape[-1]):
        name = pathlib.Path(ps_images_paths[im]).stem
        os.makedirs(os.path.join(out_path,'images', name), exist_ok=True)
        ref_image = vector_tools.build_masked(mask, images[:,:,im]/scale)
        iio.imwrite(os.path.join(out_path,'images',name,'reference.png'),numpy.uint8(255.0*numpy.clip(ref_image,0,1)))
        simulated_image = vector_tools.build_masked(mask, rendered_image[:,:,im]/scale)
        iio.imwrite(os.path.join(out_path,'images',name,'simulated.png'),numpy.uint8(255.0*numpy.clip(simulated_image,0,1)))
        if 'validity_mask' in values:
            validity_image = vector_tools.build_masked(mask, values['validity_mask'][:,0,im])
            iio.imwrite(os.path.join(out_path,'images', name, 'validity.png'), validity_image)
        for renderer in rendered_images:
            os.makedirs(os.path.join(out_path,'images', name, 'renderers'), exist_ok=True)
            simulated_image_renderer = vector_tools.build_masked(mask, rendered_images[renderer][:,:,im]/scale)
            iio.imwrite(os.path.join(out_path,'images', name, 'renderers',f'{renderer}.png'),numpy.uint8(255.0*numpy.clip(simulated_image_renderer,0,1)))


def export_misc(out_path, mask, values, losses, steps):
    os.makedirs(os.path.join(out_path,'misc'), exist_ok=True)
    iio.imwrite(os.path.join(out_path,'misc','mask.png'), mask)
    if 'normals' in values :
        normalmap = vector_tools.build_masked(mask, values['normals'])
        iio.imwrite(os.path.join(out_path,'misc','geometry_normals.png'),numpy.uint8(0.5*(normalmap*numpy.asarray([1,-1,-1])+1)*255))
    if 'points' in values:
        zmap = vector_tools.build_masked(mask, 1- ((values['points'][:,-1] - numpy.min(values['points'][:,-1])) / (numpy.max(values['points'][:,-1]) - numpy.min(values['points'][:,-1]))))
        iio.imwrite(os.path.join(out_path,'misc','geometry_zmap.png'),numpy.uint8((numpy.clip(zmap,0,1))*255))
    if 'rho' in values:
        albedomap = vector_tools.build_masked(mask, values['rho'])
        iio.imwrite(os.path.join(out_path, 'misc', 'albedomap.png'),numpy.uint8(numpy.clip(albedomap/numpy.quantile(values['rho'], 0.99),0,1)*255))
    if 'rho_spec' in values:
        albedospecmap = vector_tools.build_masked(mask, values['rho_spec'])
        iio.imwrite(os.path.join(out_path, 'misc', 'albedospecmap.png'),numpy.uint8(numpy.clip(albedospecmap/numpy.quantile(values['rho_spec'], 0.99),0,1)*255))
    loss_plot = plots.get_losses_plot(losses, steps)
    loss_plot.write_html(os.path.join(out_path, 'misc', 'loss_plot.html'))


def export_lightmaps(out_path, light_direction, light_intensity, mask, ps_images_paths):
    os.makedirs(os.path.join(out_path,'lightmaps'), exist_ok=True)
    max_intensity = numpy.quantile(light_intensity, 0.9)
    for im in range(light_direction.shape[-2]):
        name = pathlib.Path(ps_images_paths[im]).stem
        os.makedirs(os.path.join(out_path,'lightmaps', name), exist_ok=True)
        simulated_direction = vector_tools.build_masked(mask, light_direction[:,im,:])
        simulated_intensity = vector_tools.build_masked(mask, light_intensity[:, im, 0] if light_intensity.shape[-1]==1 else light_intensity[:, im, :])
        iio.imwrite(os.path.join(out_path, 'lightmaps', name, 'direction.png'),numpy.uint8(0.5*(simulated_direction*numpy.asarray([1,-1,-1])+1)*255))
        iio.imwrite(os.path.join(out_path, 'lightmaps', name,'intensity.png'),numpy.uint8(numpy.clip(simulated_intensity / max_intensity,0,1)*255))


def export_light(out_path, light_direction, light_intensity, ps_images_paths):
    os.makedirs(os.path.join(out_path,'light'), exist_ok=True)
    L0 = (lambda x : x/numpy.linalg.norm(x,axis=-1,keepdims=True))(numpy.mean(light_direction, axis=0))
    Phi = numpy.mean(light_intensity, axis=0)
    str_L0 = numpy.asarray(L0).astype(str)
    str_Phi = numpy.asarray(Phi).astype(str)
    names_array = numpy.asarray(list(map(lambda p : pathlib.Path(p).stem, ps_images_paths)),dtype=str)[:,None]
    XL0 = numpy.concatenate([names_array,str_L0],axis=-1)
    XPhi = numpy.concatenate([names_array,str_Phi],axis=-1)
    numpy.savetxt(os.path.join(out_path,'light','light_direction.lp'), XL0, fmt = '%s', header = str(len(ps_images_paths)), delimiter = ' ', comments='')
    numpy.savetxt(os.path.join(out_path,'light','light_intensity.lp'), XPhi, fmt = '%s', header = str(len(ps_images_paths)), delimiter = ' ', comments='')

def export_values(out_path, values, losses, mask):
    losses_all = numpy.concatenate(losses)
    numpy.savez(os.path.join(out_path, 'values.npz'), losses = losses_all, mask = mask, **values)

def export_results(out_path, mask, parameters, data, losses, steps, images, ps_images_paths):
    values = parameters | data
    model = models.model_from_parameters(parameters, data)
    light, renderer, _ = models.get_model(model)
    light_direction, light_intensity = light(values)
    rendered_images = renderer(light_direction, light_intensity, values)
    export_lightmaps(out_path, light_direction, light_intensity, mask, ps_images_paths)
    export_light(out_path, light_direction, light_intensity, ps_images_paths)
    export_values(out_path, values, losses, mask)
    export_misc(out_path, mask, values, losses, steps)
    export_images(out_path, rendered_images, ps_images_paths, images, mask, values)
