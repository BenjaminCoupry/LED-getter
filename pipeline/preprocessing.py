import jax
import ledgetter.utils.logs as logs
import optax
import ledgetter.utils.loading as loading
import imageio.v3 as iio
import pathlib


def preprocess(ps_images_paths, sliced=slice(None), meshroom_project=None, aligned_image_path=None,
                geometry_path=None, pose_path=None, black_image_path=None, loaded_light_folder=None,
                load_light_function=False, learning_rate=0.001, tqdm_refresh=0, added_values = {}):
    light_names = list(map(lambda p : pathlib.Path(p).stem, ps_images_paths))
    pose = loading.load_pose(pose_path if pose_path else meshroom_project, aligned_image_path if aligned_image_path else ps_images_paths) if pose_path or meshroom_project else None
    pixelmap = loading.get_pixelmap(pose if pose else ps_images_paths[0])[sliced]
    geometric_mask, normalmap, pointmap, raycaster  = loading.load_geometry(geometry_path if geometry_path else meshroom_project, pixelmap, pose)
    geom_images, undisto_mask, (_, n_im, n_c) = loading.load_images((ps_images_paths + [black_image_path]) if black_image_path else ps_images_paths, pixelmap[geometric_mask], pose, batch_size = 1000)
    geom_images, n_im = (jax.numpy.maximum(0, geom_images[...,:-1] - geom_images[...,-1:]), n_im-1) if black_image_path else (geom_images, n_im)
    geom_points, geom_normals, geom_pixels = pointmap[geometric_mask], normalmap[geometric_mask], pixelmap[geometric_mask], 
    points, normals, pixels, images = geom_points[undisto_mask],geom_normals[undisto_mask], geom_pixels[undisto_mask], geom_images[undisto_mask]
    scale = jax.numpy.max(jax.numpy.linalg.norm(points - jax.numpy.mean(points, axis=0),axis=-1))
    mask = jax.numpy.zeros(pixelmap.shape[:2], dtype=bool).at[geometric_mask].set(undisto_mask)
    output, optimizer = logs.get_tqdm_output(tqdm_refresh), optax.adam(learning_rate)
    full_shape, shapes = iio.improps(ps_images_paths[0]).shape, (images.shape[0], n_im, n_c)
    values = {**added_values, 'points':points, 'normals':normals, 'pixels':pixels}
    light_dict = loading.load_light_dict(loaded_light_folder, do_load_light=load_light_function, light_names=light_names)
    return values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names
