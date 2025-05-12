import jax
import ledgetter.utils.logs as logs
import optax
import ledgetter.utils.loading as loading
import ledgetter.utils.meshroom as meshroom
import ledgetter.rendering.validity as validity
import numpy
import imageio.v3 as iio
import os



def preprocess(ps_images_paths, sliced=slice(None), meshroom_project=None, aligned_image_path=None, geometry_path=None, pose_path=None, black_image_path=None):
    pose = loading.load_pose(pose_path if pose_path else meshroom_project, aligned_image_path if aligned_image_path else ps_images_paths) if pose_path or meshroom_project else None
    pixelmap = loading.get_pixelmap(pose if pose else ps_images_paths[0])[sliced]
    geometric_mask, normalmap, pointmap, raycaster  = loading.load_geometry(geometry_path if geometry_path else meshroom_project, pixelmap, pose)
    geom_images, undisto_mask, (_, n_im, n_c) = loading.load_images((ps_images_paths + [black_image_path]) if black_image_path else ps_images_paths, pixelmap[geometric_mask], pose, batch_size = 1000)
    geom_images = jax.numpy.maximum(0, geom_images[...,:-1] - geom_images[...,-1:]) if black_image_path else geom_images
    geom_points, geom_normals, geom_pixels = pointmap[geometric_mask], normalmap[geometric_mask], pixelmap[geometric_mask], 
    points, normals, pixels, images = geom_points[undisto_mask],geom_normals[undisto_mask], geom_pixels[undisto_mask], geom_images[undisto_mask]
    scale = jax.numpy.max(jax.numpy.linalg.norm(points - jax.numpy.mean(points, axis=0),axis=-1))
    mask = jax.numpy.zeros(pixelmap.shape[:2], dtype=bool).at[geometric_mask].set(undisto_mask)
    output = logs.get_tqdm_output(0)
    optimizer = optax.adam(0.001) #optax.lbfgs()
    return points, normals, pixels, images, raycaster, mask, (images.shape[0], n_im, n_c), output, optimizer, scale

def prepare_ps(light_path, ps_images_paths):
    shape = iio.improps(ps_images_paths[0]).shape
    with numpy.load(os.path.join(light_path, 'values.npz')) as light_values:
        light_pixels, light_rho = light_values['pixels'], light_values['rho']
    path = next(filter(lambda p : os.path.isfile(p), [os.path.join(light_path, 'light','light_function.jax'), os.path.join(light_path, 'values.npz')]))
    light = loading.load_light(path)
    light_dict = {'pixels':light_pixels, 'rho':light_rho, 'light':light}
    return light_dict, shape