import jax
import numpy
import ledgetter.rendering.models as models
import ledgetter.rendering.renderers as renderers
import matplotlib.pyplot as plt
import imageio.v3 as iio
import ledgetter.utils.vector_tools as vector_tools
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
            simulated_image_renderer = vector_tools.build_masked(mask, rendered_images[renderer][:,:,im]/scale)
            iio.imwrite(os.path.join(out_path,'images', name, f'{renderer}_simulated.png'),numpy.uint8(255.0*numpy.clip(simulated_image_renderer,0,1)))



def export_misc(out_path, mask, values, losses, steps):
    os.makedirs(os.path.join(out_path,'misc'), exist_ok=True)
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
    loss_plot = get_losses_plot(losses, steps)
    loss_plot.savefig(os.path.join(out_path, 'misc', 'loss_plot.png'), dpi=300, bbox_inches='tight')


def export_lightmaps(out_path, light_direction, light_intensity, mask, ps_images_paths):
    os.makedirs(os.path.join(out_path,'lights'), exist_ok=True)
    max_intensity = numpy.quantile(light_intensity, 0.9)
    for im in range(light_direction.shape[-2]):
        name = pathlib.Path(ps_images_paths[im]).stem
        simulated_direction = vector_tools.build_masked(mask, light_direction[:,im,:])
        simulated_intensity = vector_tools.build_masked(mask, light_intensity[:, im, 0] if light_intensity.shape[-1]==1 else light_intensity[:, im, :])
        iio.imwrite(os.path.join(out_path, 'lights', f'{name}_direction.png'),numpy.uint8(0.5*(simulated_direction*numpy.asarray([1,-1,-1])+1)*255))
        iio.imwrite(os.path.join(out_path, 'lights',f'{name}_intensity.png'),numpy.uint8(numpy.clip(simulated_intensity / max_intensity,0,1)*255))


def export_LP(out_path, light_direction, ps_images_paths):
    os.makedirs(os.path.join(out_path,'LP'), exist_ok=True)
    L0 = (lambda x : x/numpy.linalg.norm(x,axis=-1,keepdims=True))(numpy.mean(light_direction, axis=0))
    str_L0 = numpy.asarray(L0).astype(str)
    names_array = numpy.asarray(list(map(lambda p : pathlib.Path(p).stem, ps_images_paths)),dtype=str)[:,None]
    X = numpy.concatenate([names_array,str_L0],axis=-1)
    numpy.savetxt(os.path.join(out_path,'LP','light_direction.lp'), X, fmt = '%s', header = str(len(ps_images_paths)), delimiter = ' ', comments='')

def export_light(out_path, values, losses):
    losses_all = numpy.concatenate(losses)
    numpy.savez(os.path.join(out_path, 'light.npz'), losses = losses_all, **values)

def get_losses_plot(losses, steps):
    iterations = list(map(len, losses))
    boundaries = numpy.cumsum([0] + iterations)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    line1, = ax1.plot(numpy.concatenate(losses), label="logarithmic", color = 'red')
    for i in range(len(iterations)):
        start = boundaries[i] if i!=0 else boundaries[0]-0.1*boundaries[-1]
        end = boundaries[i + 1]
        color = "wheat" if i % 2 == 0 else "lightblue"
        z = 0.9 if i % 2 == 0 else 0.85
        ax1.axvspan(start, end, color=color, alpha=0.5)
        ax2.text((start + end) / 2, z, steps[i],
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                transform=ax1.get_xaxis_transform())
    ax1.set_yscale('log')
    ax1.set_xlim(boundaries[0]-0.1*boundaries[-1], boundaries[-1])
    ax1.set_xlabel("Iteration")
    ax1.set_title("Loss over Iterations with Optimization Steps")
    
    ax2.set_ylabel("Loss")
    line2, = ax2.plot(numpy.concatenate(losses), label="linear", color = 'blue')
    ax2.grid(axis='y')
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="center right")
    ax1.tick_params(axis='y', which='both', colors='red') 
    ax2.tick_params(axis='y', which='both', colors='blue')
    fig.tight_layout()
    return fig



def export_results(out_path, mask, parameters, data, losses, steps, images, ps_images_paths):
    values = parameters | data
    model = models.model_from_parameters(parameters, data)
    light, renderer, _ = models.get_model(model)
    light_direction, light_intensity = light(values)
    rendered_images = renderer(light_direction, light_intensity, values)
    export_lightmaps(out_path, light_direction, light_intensity, mask, ps_images_paths)
    export_LP(out_path, light_direction, ps_images_paths)
    export_light(out_path, values, losses)
    export_misc(out_path, mask, values, losses, steps)
    export_images(out_path, rendered_images, ps_images_paths, images, mask, values)
