import jax
import ledgetter.utils.logs as logs
import optax
import ledgetter.utils.loading as loading
import ledgetter.utils.meshroom as meshroom
import ledgetter.rendering.validity as validity



def preprocess(ps_images_paths, step, threshold=(0,0), meshroom_project=None, aligned_image_path=None, mesh_path=None, pose_path=None):
    pose = loading.load_pose(pose_path if pose_path else meshroom_project, aligned_image_path if aligned_image_path else ps_images_paths) if pose_path or meshroom_project else None
    pixelmap = loading.get_pixelmap(pose)[::step, ::step]
    geometric_mask, normalmap, pointmap = loading.load_geometry(mesh_path if mesh_path else meshroom_project, pixelmap, pose)
    dev_images, undisto_mask, (_, n_im, n_c) = loading.load_images(ps_images_paths, pixelmap, pose)
    mask = jax.numpy.logical_and(geometric_mask, undisto_mask)
    points, normals, pixels, images = jax.numpy.asarray(pointmap[mask]), jax.numpy.asarray(normalmap[mask]), jax.numpy.asarray(pixelmap[mask]), jax.numpy.asarray(dev_images[mask])
    validity_mask = validity.validity_mask(images, threshold[0], threshold[1])
    output = logs.get_tqdm_output(0)
    optimizer = optax.adam(0.001) #optax.lbfgs()
    return points, normals, pixels, images, validity_mask, mask, (images.shape[0], n_im, n_c), output, optimizer
