import jax
import ledgetter.utils.logs as logs
import optax
import ledgetter.utils.loading as loading
import ledgetter.utils.meshroom as meshroom
import ledgetter.rendering.validity as validity




def preprocess(ps_images_paths, sliced=slice(None), meshroom_project=None, aligned_image_path=None, geometry_path=None, pose_path=None, black_image_path=None):
    pose = loading.load_pose(pose_path if pose_path else meshroom_project, aligned_image_path if aligned_image_path else ps_images_paths) if pose_path or meshroom_project else None
    pixelmap = loading.get_pixelmap(pose if pose else ps_images_paths[0])[sliced]
    geometric_mask, normalmap, pointmap, raycaster  = loading.load_geometry(geometry_path if geometry_path else meshroom_project, pixelmap, pose)
    geom_images, undisto_mask, (_, n_im, n_c) = loading.load_images((ps_images_paths + [black_image_path]) if black_image_path else ps_images_paths, pixelmap[geometric_mask], pose)
    geom_images = jax.numpy.maximum(0, geom_images[...,:-1] - geom_images[...,-1:]) if black_image_path else geom_images
    geom_points, geom_normals, geom_pixels = pointmap[geometric_mask], normalmap[geometric_mask], pixelmap[geometric_mask], 
    points, normals, pixels, images = geom_points[undisto_mask],geom_normals[undisto_mask], geom_pixels[undisto_mask], geom_images[undisto_mask]
    scale = jax.numpy.max(jax.numpy.linalg.norm(points - jax.numpy.mean(points, axis=0),axis=-1))
    mask = jax.numpy.zeros(pixelmap.shape[:2], dtype=bool).at[geometric_mask].set(undisto_mask)
    output = logs.get_tqdm_output(0)
    optimizer = optax.adam(0.001) #optax.lbfgs()
    return points, normals, pixels, images, raycaster, mask, (images.shape[0], n_im, n_c), output, optimizer, scale
