import jax
import numpy
import ledgetter.rendering.lights as lights
import ledgetter.rendering.renderers as renderers
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import pathlib



def export_misc(out_path, mask, normals, points, rho, rho_spec):
    os.makedirs(os.path.join(out_path,'misc'), exist_ok=True)
    normalmap = numpy.zeros(mask.shape + (3,))
    normalmap[mask] = normals
    iio.imwrite(os.path.join(out_path,'misc','geometry_normals.png'),numpy.uint8(0.5*(normalmap*numpy.asarray([1,-1,-1])+1)*255))
    zmap = numpy.zeros(mask.shape)
    zmap[mask] = 1- ((points[:,-1] - numpy.min(points[:,-1])) / (numpy.max(points[:,-1]) - numpy.min(points[:,-1])))
    iio.imwrite(os.path.join(out_path,'misc','geometry_zmap.png'),numpy.uint8((numpy.clip(zmap,0,1))*255))
    albedomap = numpy.zeros(mask.shape + (3,))
    albedomap[mask] = rho
    iio.imwrite(os.path.join(out_path, 'misc', 'albedomap.png'),numpy.uint8(numpy.clip(albedomap/numpy.quantile(rho, 0.99),0,1)*255))
    albedospecmap = numpy.zeros(mask.shape)
    albedospecmap[mask] = rho_spec
    iio.imwrite(os.path.join(out_path, 'misc', 'albedospecmap.png'),numpy.uint8(numpy.clip(albedospecmap/numpy.quantile(rho_spec, 0.99),0,1)*255))


def export_images(out_path, light_direction, light_intensity, validity_mask, mask, ps_images_paths, images, normals, points, rho, rho_spec, tau_spec):
    os.makedirs(os.path.join(out_path,'images'), exist_ok=True)
    lamertian_images = renderers.lambertian_renderer(light_direction, light_intensity, normals, points, rho) #lambertian and specular
    specular_images = renderers.specular_renderer(light_direction, light_intensity, normals, points, rho_spec, tau_spec)
    model_images = lamertian_images + specular_images
    ref_image = numpy.zeros(mask.shape+(3,))
    simulated_image = numpy.zeros(mask.shape+(3,))
    simulated_lambertian_image = numpy.zeros(mask.shape+(3,))
    simulated_specular_image = numpy.zeros(mask.shape+(3,))
    validity_image = numpy.zeros(mask.shape, dtype=bool)
    scale = numpy.quantile(images, 0.98)
    for im in range(images.shape[-1]):
        simulated_image[mask] = model_images[:,:,im]/scale
        simulated_lambertian_image[mask] = lamertian_images[:,:,im]/scale
        simulated_specular_image[mask] = specular_images[:,:,im]/scale
        ref_image[mask] = images[:,:,im]/scale
        validity_image[mask] = validity_mask[:,im]
        name = pathlib.Path(ps_images_paths[im]).stem
        iio.imwrite(os.path.join(out_path,'images',f'{name}_simulated_lambertian.png'),numpy.uint8(255.0*numpy.clip(simulated_lambertian_image,0,1)))
        iio.imwrite(os.path.join(out_path,'images',f'{name}_simulated_specular.png'),numpy.uint8(255.0*numpy.clip(simulated_specular_image,0,1)))
        iio.imwrite(os.path.join(out_path,'images',f'{name}_simulated.png'),numpy.uint8(255.0*numpy.clip(simulated_image,0,1)))
        iio.imwrite(os.path.join(out_path,'images',f'{name}_reference.png'),numpy.uint8(255.0*numpy.clip(ref_image,0,1)))
        iio.imwrite(os.path.join(out_path,'images',f'{name}_validity.png'), validity_image)


def export_lightmaps(out_path, light_direction, light_intensity, mask, ps_images_paths):
    os.makedirs(os.path.join(out_path,'lights'), exist_ok=True)
    simulated_direction, simulated_intensity = numpy.zeros(mask.shape+(3,)), numpy.zeros(mask.shape+(3,))
    max_intensity = numpy.quantile(light_intensity, 0.9)
    for im in range(light_direction.shape[-2]):
        name = pathlib.Path(ps_images_paths[im]).stem
        simulated_direction[mask] = light_direction[:,im,:]
        simulated_intensity[mask] = light_intensity[:, im, :]
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
    rho, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec = parameters
    numpy.savez(os.path.join(out_path, 'light.npz'), pixels = pixels, rho=rho, light_locations=light_locations, light_power=light_power, light_principal_direction=light_principal_direction, mu=mu, rho_spec=rho_spec, tau_spec=tau_spec, losses = losses_all)

def export_results(out_path, parameters, points, normals, pixels, images, validity_mask, mask, losses, ps_images_paths):
    #TODO inclure la specularité dans le rendu final
    rho, light_locations, light_power, light_principal_direction, mu, rho_spec, tau_spec = parameters
    light_direction, light_intensity = lights.get_led_light(light_locations, light_power, light_principal_direction, mu, points)
    export_misc(out_path, mask, normals, points, rho, rho_spec)
    export_images(out_path, light_direction, light_intensity, validity_mask, mask, ps_images_paths, images, normals, points, rho, rho_spec, tau_spec)
    export_lightmaps(out_path, light_direction, light_intensity, mask, ps_images_paths)
    export_LP(out_path, light_direction, ps_images_paths)
    export_light(out_path, parameters, pixels, losses)
    #exporter les plots ?



