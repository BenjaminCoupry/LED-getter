import jax
import numpy
import ledgetter.rendering.lights as lights
import ledgetter.rendering.renderers as renderers
import matplotlib.pyplot as plt
import imageio.v3 as iio
import ledgetter.utils.vector_tools as vector_tools
import os
import pathlib



def export_misc(out_path, mask, normals, points, rho, rho_spec):
    os.makedirs(os.path.join(out_path,'misc'), exist_ok=True)
    normalmap = vector_tools.build_masked(mask, normals)
    iio.imwrite(os.path.join(out_path,'misc','geometry_normals.png'),numpy.uint8(0.5*(normalmap*numpy.asarray([1,-1,-1])+1)*255))
    zmap = vector_tools.build_masked(mask, 1- ((points[:,-1] - numpy.min(points[:,-1])) / (numpy.max(points[:,-1]) - numpy.min(points[:,-1]))))
    iio.imwrite(os.path.join(out_path,'misc','geometry_zmap.png'),numpy.uint8((numpy.clip(zmap,0,1))*255))
    albedomap = vector_tools.build_masked(mask, rho)
    iio.imwrite(os.path.join(out_path, 'misc', 'albedomap.png'),numpy.uint8(numpy.clip(albedomap/numpy.quantile(rho, 0.99),0,1)*255))
    albedospecmap = vector_tools.build_masked(mask, rho_spec)
    iio.imwrite(os.path.join(out_path, 'misc', 'albedospecmap.png'),numpy.uint8(numpy.clip(albedospecmap/numpy.quantile(rho_spec, 0.99),0,1)*255))


def export_images(out_path, light_direction, light_intensity, validity_mask, mask, ps_images_paths, images, normals, points, rho, rho_spec, tau_spec):
    os.makedirs(os.path.join(out_path,'images'), exist_ok=True)
    lamertian_images = renderers.lambertian_renderer(light_direction, light_intensity, normals, points, rho) #lambertian and specular
    specular_images = renderers.specular_renderer(light_direction, light_intensity, normals, points, rho_spec, tau_spec)
    model_images = lamertian_images + specular_images
    scale = numpy.quantile(images, 0.98)
    for im in range(images.shape[-1]):
        simulated_image = vector_tools.build_masked(mask, model_images[:,:,im]/scale)
        simulated_lambertian_image = vector_tools.build_masked(mask, lamertian_images[:,:,im]/scale)
        simulated_specular_image = vector_tools.build_masked(mask, specular_images[:,:,im]/scale)
        ref_image = vector_tools.build_masked(mask, images[:,:,im]/scale)
        validity_image = vector_tools.build_masked(mask, validity_mask[:,im])
        name = pathlib.Path(ps_images_paths[im]).stem
        iio.imwrite(os.path.join(out_path,'images',f'{name}_simulated_lambertian.png'),numpy.uint8(255.0*numpy.clip(simulated_lambertian_image,0,1)))
        iio.imwrite(os.path.join(out_path,'images',f'{name}_simulated_specular.png'),numpy.uint8(255.0*numpy.clip(simulated_specular_image,0,1)))
        iio.imwrite(os.path.join(out_path,'images',f'{name}_simulated.png'),numpy.uint8(255.0*numpy.clip(simulated_image,0,1)))
        iio.imwrite(os.path.join(out_path,'images',f'{name}_reference.png'),numpy.uint8(255.0*numpy.clip(ref_image,0,1)))
        iio.imwrite(os.path.join(out_path,'images',f'{name}_validity.png'), validity_image)


def export_lightmaps(out_path, light_direction, light_intensity, mask, ps_images_paths):
    os.makedirs(os.path.join(out_path,'lights'), exist_ok=True)
    max_intensity = numpy.quantile(light_intensity, 0.9)
    for im in range(light_direction.shape[-2]):
        name = pathlib.Path(ps_images_paths[im]).stem
        simulated_direction = vector_tools.build_masked(mask, light_direction[:,im,:])
        simulated_intensity = vector_tools.build_masked(mask, light_intensity[:, im, :])
        iio.imwrite(os.path.join(out_path, 'lights', f'{name}_direction.png'),numpy.uint8(0.5*(simulated_direction*numpy.asarray([1,-1,-1])+1)*255))
        iio.imwrite(os.path.join(out_path, 'lights',f'{name}_intensity.png'),numpy.uint8(numpy.clip(simulated_intensity / max_intensity,0,1)*255))

def export_LP(out_path, light_direction, ps_images_paths):
    os.makedirs(os.path.join(out_path,'LP'), exist_ok=True)
    L0 = (lambda x : x/numpy.linalg.norm(x,axis=-1,keepdims=True))(numpy.mean(light_direction, axis=0))
    str_L0 = numpy.asarray(L0).astype(str)
    names_array = numpy.asarray(list(map(lambda p : pathlib.Path(p).stem, ps_images_paths)),dtype=str)[:,None]
    X = numpy.concatenate([names_array,str_L0],axis=-1)
    numpy.savetxt(os.path.join(out_path,'LP','light_direction.lp'), X, fmt = '%s', header = str(len(ps_images_paths)), delimiter = ' ', comments='')

def export_light(out_path, parameters, pixels, losses):
    losses_all = jax.numpy.concatenate(losses)
    numpy.savez(os.path.join(out_path, 'light.npz'), pixels = pixels, losses = losses_all, **parameters)

def export_results(out_path, parameters, points, normals, pixels, images, validity_mask, mask, losses, ps_images_paths):
    light_direction, light_intensity = lights.get_led_light(parameters['light_locations'], parameters['light_power'], parameters['light_principal_direction'], parameters['mu'], points)
    export_misc(out_path, mask, normals, points, parameters['rho'], parameters['rho_spec'])
    export_images(out_path, light_direction, light_intensity, validity_mask, mask, ps_images_paths, images, normals, points, parameters['rho'], parameters['rho_spec'], parameters['tau_spec'])
    export_lightmaps(out_path, light_direction, light_intensity, mask, ps_images_paths)
    export_LP(out_path, light_direction, ps_images_paths)
    export_light(out_path, parameters, pixels, losses)



