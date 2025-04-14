import jax
import open3d
import numpy
import ledgetter.utils.vector_tools as vector_tools

def get_camera_rays(coordinates, K):
    """Computes camera rays from pixel coordinates.

    Args:
        coordinates (Array ..., 2): Pixel coordinates.
        K (Array 3, 3): Camera intrinsic matrix.

    Returns:
        Array ..., 3: Camera rays.
    """
    homogeneous_coordinates = vector_tools.to_homogeneous(coordinates)
    inv_K = jax.numpy.linalg.inv(K)
    camera_rays = jax.numpy.einsum('uk, ...k -> ...u', inv_K, homogeneous_coordinates)
    return camera_rays

def get_rototranslation_matrix(R, t, to_camera=False):
    """Builds a 4x4 roto-translation matrix from rotation and translation.

    Args:
        R (Array 3, 3): Rotation matrix.
        t (Array 3,): Translation vector.

    Returns:
        Array 4, 4: Roto-translation matrix.
    """
    transform = jax.numpy.block([[R, jax.numpy.expand_dims(t,axis=-1)],[jax.numpy.zeros((1,3)),jax.numpy.ones((1,1))]])
    if to_camera:
        transform = jax.numpy.linalg.inv(transform)
    return transform

def get_geometry(raycaster, K):
    def geometry(coordinates):
        d = get_camera_rays(coordinates, K)
        mask, normals, points = raycaster(0, d)
        return mask, normals, points
    return geometry

