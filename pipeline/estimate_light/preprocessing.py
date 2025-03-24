import jax
import ledgetter.utils.logs as logs
import optax
import ledgetter.utils.loading as loading
import ledgetter.utils.meshroom as meshroom
import ledgetter.rendering.validity as validity


def preprocess(project_path, ps_images_paths, step, threshold, aligned_image_path=None, mesh_path = None):
    pose = meshroom.get_pose(project_path, aligned_image_path if aligned_image_path is not None else ps_images_paths)
    mesh = meshroom.get_mesh(project_path if mesh_path is None else mesh_path)
    pixelmap = loading.get_pixelmap(pose)[::step, ::step]
    geometric_mask, normalmap, pointmap = loading.load_geometry(mesh, pose, pixelmap)
    dev_images, undisto_mask, (_, n_im, n_c) = loading.load_raw_images(ps_images_paths, pose, pixelmap)
    mask = jax.numpy.logical_and(geometric_mask, undisto_mask)
    points, normals, pixels, images = jax.numpy.asarray(pointmap[mask]), jax.numpy.asarray(normalmap[mask]), jax.numpy.asarray(pixelmap[mask]), jax.numpy.asarray(dev_images[mask])
    validity_mask = validity.validity_mask(images, threshold[0], threshold[1])
    output = logs.get_tqdm_output(0)
    optimizer = optax.adam(0.001) #optax.lbfgs()

    return points, normals, pixels, images, validity_mask, mask, (images.shape[0], n_im, n_c), output, optimizer