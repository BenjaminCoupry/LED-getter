import jax
import ledgetter.utils.logs as logs
import optax
import ledgetter.utils.loading as loading
import imageio.v3 as iio
import pathlib
import itertools


def preprocess(ps_images_paths, sliced=slice(None), meshroom_project=None, aligned_image_path=None,
                geometry_path=None, pose_path=None, black_image_path=None, loaded_light_folder=None,
                load_light_function=False, learning_rate=0.001, tqdm_refresh=0, added_values = {}, 
                flip_lp=False, flip_mesh=True, apply_images_undisto=True, apply_geometry_images_undisto=True, spheres_to_load = None):
    ps_images_paths = list(map(lambda p : p if type(p) is tuple else (p,), ps_images_paths))
    light_names = list(map(lambda p : pathlib.Path(p[0]).stem, ps_images_paths))
    pose = loading.load_pose(pose_path if pose_path else (meshroom_project if meshroom_project else geometry_path), 
                             aligned_image_path=aligned_image_path if aligned_image_path else list(itertools.chain.from_iterable(ps_images_paths))) if pose_path or meshroom_project or geometry_path else None
    pixelmap = loading.get_pixelmap(pose if (pose and pose['width'] is not None and pose['height'] is not None) else ps_images_paths[0][0])[sliced]
    geometric_mask, normalmap, pointmap, raycaster  = loading.load_geometry(geometry_path if geometry_path else meshroom_project,
                                                                            pixelmap, pose, flip_mesh=flip_mesh, batch_size = 1000,
                                                                            apply_undisto=apply_geometry_images_undisto, spheres_to_load = spheres_to_load)
    geom_images, undisto_mask, (_, n_im, n_c) = loading.load_images((ps_images_paths + [(black_image_path,)]) if black_image_path else ps_images_paths,
                                                                     pixelmap[geometric_mask], pose, batch_size = 1000, apply_undisto=apply_images_undisto)
    geom_images, n_im = (jax.numpy.maximum(0, geom_images[...,:-1] - geom_images[...,-1:]), n_im-1) if black_image_path else (geom_images, n_im)
    geom_points, geom_normals, geom_pixels = pointmap[geometric_mask], normalmap[geometric_mask], pixelmap[geometric_mask], 
    points, normals, pixels, images = geom_points[undisto_mask],geom_normals[undisto_mask], geom_pixels[undisto_mask], geom_images[undisto_mask]
    scale = jax.numpy.quantile(jax.numpy.linalg.norm(points - jax.numpy.mean(points, axis=0),axis=-1), 0.95)
    mask = jax.numpy.zeros(pixelmap.shape[:2], dtype=bool).at[geometric_mask].set(undisto_mask)
    output, optimizer = logs.get_tqdm_output(tqdm_refresh), optax.adam(learning_rate)
    full_shape, shapes = iio.improps(ps_images_paths[0][0]).shape, (images.shape[0], n_im, n_c)
    values = {**added_values, 'points':points, 'normals':normals, 'pixels':pixels}
    light_dict = loading.load_light_dict(loaded_light_folder, do_load_light=load_light_function, light_names=light_names, flip_lp=flip_lp)
    return values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names
