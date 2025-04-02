import jax
import ledgetter.utils.logs as logs
import optax
import ledgetter.utils.loading as loading
import ledgetter.utils.meshroom as meshroom
import ledgetter.rendering.validity as validity




def preprocess(ps_images_paths, sliced=slice(None), threshold=(0,0), meshroom_project=None, aligned_image_path=None, geometry_path=None, pose_path=None):
    pose = loading.load_pose(pose_path if pose_path else meshroom_project, aligned_image_path if aligned_image_path else ps_images_paths) if pose_path or meshroom_project else None
    pixelmap = loading.get_pixelmap(pose if pose else ps_images_paths[0])[sliced] #can also take a block index
    geometric_mask, normalmap, pointmap = loading.load_geometry(geometry_path if geometry_path else meshroom_project, pixelmap, pose)
    geom_images, undisto_mask, (_, n_im, n_c) = loading.load_images(ps_images_paths, pixelmap[geometric_mask], pose)
    geom_points, geom_normals, geom_pixels = pointmap[geometric_mask], normalmap[geometric_mask], pixelmap[geometric_mask]
    points, normals, pixels, images = geom_points[undisto_mask],geom_normals[undisto_mask], geom_pixels[undisto_mask], geom_images[undisto_mask]
    mask = jax.numpy.zeros(pixelmap.shape[:2], dtype=bool).at[geometric_mask].set(undisto_mask)
    validity_mask = validity.validity_mask(images, threshold[0], threshold[1])
    output = logs.get_tqdm_output(0)
    optimizer = optax.adam(0.001) #optax.lbfgs()
    return points, normals, pixels, images, validity_mask, mask, (images.shape[0], n_im, n_c), output, optimizer
