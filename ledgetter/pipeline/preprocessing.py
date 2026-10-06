import jax
import glob
import ledgetter.utils.logs as logs
import optax
import ledgetter.utils.loading as loading
import imageio.v3 as iio
import pathlib
import itertools
import ledgetter.utils.files as files
import ledgetter.image.undistort as undistort
import ledgetter.utils.meshroom as meshroom

# def preprocess(ps_images_paths=None, sliced=slice(None), meshroom_project=None, aligned_image_path=None,
#         geometry_path=None, pose_path=None, black_image_path=None, loaded_light_folder=None,
#         load_light_function=False, learning_rate=0.001, tqdm_refresh=0, added_values = {}, 
#         flip_lp=False, flip_mesh=True, apply_geometry_undisto=True,
#         spheres_to_load = None, remove_image_gamma = False, parent_image_name=0,
#         meshroom_pose_index=None, meshroom_pose_id=None,
#         get_meshroom_developped_image=False, glob_ps_images_paths=False):
#     ps_images_paths = list(zip(*[glob.glob(path[0]) if (glob_ps_images_paths and len(path)==1) else path for path in ps_images_paths])) if ps_images_paths is not None else None
#     view_id, ps_images_paths = None, (list(map(lambda p : p if type(p) is tuple else (p,), ps_images_paths)) if ps_images_paths is not None else None) #ATTENTION NONE
#     view_id, ps_images_paths = meshroom.get_images_from_mvps(meshroom_project, pose_id=meshroom_pose_id, pose_index=meshroom_pose_index, get_developped=get_meshroom_developped_image) if (ps_images_paths is None and meshroom_project is not None) else (None, ps_images_paths)
#     light_names = list(map(lambda p : (pathlib.Path(p[0]).parent.parent.stem if parent_image_name==2 else (pathlib.Path(p[0]).parent.stem if parent_image_name==1 else pathlib.Path(p[0]).stem)), ps_images_paths))
#     pose_path = files.first_existing_file([pose_path, meshroom_project, geometry_path])
#     geometry_path = files.first_existing_file([geometry_path, meshroom_project])
#     aligned_image_path = files.first_existing_file([aligned_image_path, list(itertools.chain.from_iterable(ps_images_paths))], allow_any_obj=True) if view_id is not None else None
#     images_paths = (ps_images_paths + [(black_image_path,)]) if black_image_path is not None else ps_images_paths
#     full_shape = iio.improps(ps_images_paths[0][0]).shape
#     pose = loading.load_pose(pose_path, view_id = view_id, aligned_image_path=aligned_image_path) if pose_path is not None else None
#     pixelmap_path = pose if (pose is not None and pose['width'] is not None and pose['height'] is not None) else ps_images_paths[0][0]
#     coordinates_transform = undistort.get_coordinates_transform(jax.numpy.asarray(pose['K']),(full_shape[1], full_shape[0]) ,jax.numpy.asarray(pose['distorsion']), image_to_pinhole=True) if (pose is not None) and (pose['distorsion'] is not None) and apply_geometry_undisto else (lambda pixels : pixels)
#     pixelmap = loading.get_pixelmap(pixelmap_path)[sliced]
#     geometric_mask, normalmap, pointmap, raycaster, objects_id_mask  =\
#             loading.load_geometry(geometry_path, pixelmap, coordinates_transform, pose, flip_mesh=flip_mesh, spheres_to_load = spheres_to_load)
#     geom_points, geom_normals, geom_pixels, geom_objects_id = pointmap[geometric_mask], normalmap[geometric_mask], pixelmap[geometric_mask], objects_id_mask[geometric_mask]
#     geom_images, geom_filters, images_mask, (_, n_im, n_c, n_f) =\
#             loading.load_images(images_paths, geom_pixels, remove_image_gamma = remove_image_gamma) 
#     geom_images, n_im = (jax.numpy.maximum(0, geom_images[...,:-1] - geom_images[...,-1:]), n_im-1) if black_image_path else (geom_images, n_im)
#     points, normals, pixels, images, objects_id, filters = geom_points[images_mask],geom_normals[images_mask], geom_pixels[images_mask], geom_images[images_mask], geom_objects_id[images_mask], geom_filters[images_mask]
#     scale = jax.numpy.quantile(jax.numpy.linalg.norm(points - jax.numpy.mean(points, axis=0),axis=-1), 0.95)
#     mask = jax.numpy.zeros(pixelmap.shape[:2], dtype=bool).at[geometric_mask].set(images_mask)
#     output, optimizer = logs.get_tqdm_output(tqdm_refresh), optax.adam(learning_rate)
#     shapes = (images.shape[0], n_im, n_c, n_f)
#     values = {**added_values, 'points':points, 'normals':normals, 'pixels':pixels, 'objects_id':objects_id, 'filters':filters}
#     light_dict = loading.load_light_dict(loaded_light_folder, do_load_light=load_light_function, light_names=light_names, flip_lp=flip_lp)
#     return values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names, pose

def preprocess(ps_images_paths, sliced=slice(None), meshroom_project=None, aligned_image_path=None,
        geometry_path=None, pose_path=None, black_image_path=None, loaded_light_folder=None,
        load_light_function=False, learning_rate=0.001, tqdm_refresh=0, added_values = {}, 
        flip_lp=False, flip_mesh=True, apply_geometry_undisto=True,
        spheres_to_load = None, remove_image_gamma = False, parent_image_name=0,
        glob_ps_images_paths=False, view_id=None):
    
    ps_images_paths = list(zip(*[glob.glob(path[0]) if (glob_ps_images_paths and len(path)==1) else path for path in ps_images_paths]))
    
    light_names = list(map(lambda p : files.get_parent_stem(p[0], parent_image_name), ps_images_paths))
     
    if black_image_path is not None:
        ps_images_paths = ps_images_paths + [tuple(black_image_path)]
    else:
        ps_images_paths = ps_images_paths
 
    pose_path = files.first_existing_file([pose_path, meshroom_project, geometry_path])
    geometry_path = files.first_existing_file([geometry_path, meshroom_project])

    if view_id is None:
        all_images_flat = list(itertools.chain.from_iterable(ps_images_paths))
        aligned_image_path = files.first_existing_file(
            [aligned_image_path, all_images_flat], allow_any_obj=True
        )
    else:
        aligned_image_path = None
    
    if pose_path is not None:
        pose = loading.load_pose(pose_path, view_id = view_id, aligned_image_path=aligned_image_path)
    else:
        pose = None
        
    full_shape = iio.improps(ps_images_paths[0][0]).shape

    if pose is not None and pose['width'] is not None and pose['height'] is not None:
        pixelmap_path = pose
    else:
        pixelmap_path = ps_images_paths[0][0]

    if pose is not None and pose['distorsion'] is not None and apply_geometry_undisto:
        coordinates_transform = undistort.get_coordinates_transform(
            jax.numpy.asarray(pose['K']),
            (full_shape[1], full_shape[0]),
            jax.numpy.asarray(pose['distorsion']),
            image_to_pinhole=True,
        )
    else:
        coordinates_transform = lambda pixels: pixels

    pixelmap = loading.get_pixelmap(pixelmap_path)[sliced]
    
    geometric_mask, normalmap, pointmap, raycaster, objects_id_mask = loading.load_geometry(
        geometry_path, pixelmap, coordinates_transform, pose,
        flip_mesh=flip_mesh, spheres_to_load=spheres_to_load,
    )

    geom_points = pointmap[geometric_mask]
    geom_normals = normalmap[geometric_mask]
    geom_pixels = pixelmap[geometric_mask]
    geom_objects_id = objects_id_mask[geometric_mask]
    
    geom_images, geom_filters, images_mask, (_, n_im, n_c, n_f) = loading.load_images(
        ps_images_paths, geom_pixels, remove_image_gamma=remove_image_gamma
    )

    if black_image_path is not None:
        geom_images = jax.numpy.maximum(0, geom_images[..., :-1] - geom_images[..., -1:])
        n_im = n_im - 1

    points = geom_points[images_mask]
    normals = geom_normals[images_mask]
    pixels = geom_pixels[images_mask]
    images = geom_images[images_mask]
    objects_id = geom_objects_id[images_mask]
    filters = geom_filters[images_mask]
    
    scale = jax.numpy.quantile(
        jax.numpy.linalg.norm(points - jax.numpy.mean(points, axis=0), axis=-1), 0.95
    )

    mask = jax.numpy.zeros(pixelmap.shape[:2], dtype=bool).at[geometric_mask].set(images_mask)

    output = logs.get_tqdm_output(tqdm_refresh)
    optimizer = optax.adam(learning_rate)

    shapes = (images.shape[0], n_im, n_c, n_f)

    values = {
        **added_values,
        'points': points,
        'normals': normals,
        'pixels': pixels,
        'objects_id': objects_id,
        'filters': filters,
    }

    light_dict = loading.load_light_dict(
        loaded_light_folder, do_load_light=load_light_function,
        light_names=light_names, flip_lp=flip_lp,
    )

    return values, images, mask, raycaster, shapes, full_shape, output, optimizer, scale, light_dict, light_names, pose
    
    
        
    