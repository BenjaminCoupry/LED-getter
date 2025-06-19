
import jax
import ledgetter.utils.vector_tools as vector_tools

def axis_angle_to_matrix(axis, angle):
    """
    Converts an axis-angle representation to a rotation matrix.
    Parameters:
        axis (array-like): A 3D vector representing the axis of rotation. 
                           It should be normalized to have a unit length.
        angle (float or array-like): The angle of rotation in radians. 
                                     Can be a scalar or an array for batch processing.
    Returns:
        numpy.ndarray: A 3x3 rotation matrix (or a batch of 3x3 matrices) 
                       corresponding to the given axis and angle.
    """
    K = vector_tools.cross_product_matrix(axis)
    R = jax.numpy.eye(3) + jax.numpy.expand_dims(jax.numpy.sin(angle), axis=(-1,-2)) * K + jax.numpy.expand_dims(1 - jax.numpy.cos(angle), axis=(-1,-2)) * jax.numpy.matmul(K, K)
    return R

def matrix_to_axis_angle(R):
    trace = jax.numpy.trace(R, axis1=-2, axis2=-1)
    angle = jax.numpy.arccos(jax.numpy.clip(0.5*(trace-1), -1, 1))
    direction = jax.numpy.stack([R[...,2,1]-R[...,1,2], R[...,0,2]-R[...,2,0], R[...,1,0]-R[...,0,1]], axis=-1)
    axis = vector_tools.norm_vector(direction)[1]
    return axis, angle

def rotation_between_vectors(v1, v2, free_rotation):
    """
    Computes the rotation matrix that aligns vector `v1` to vector `v2`, 
    with an additional free rotation applied around `v2`.
    Args:
        v1 (array-like): The initial vector to be rotated.
        v2 (array-like): The target vector to align `v1` with.
        free_rotation (float): The angle (in radians) of free rotation 
            to be applied around the axis defined by `v2`.
    Returns:
        jax.numpy.ndarray: A 3x3 rotation matrix that represents the 
        combined rotation aligning `v1` to `v2` and applying the free rotation.
    """
    n1, n2 = vector_tools.norm_vector(v1)[1], vector_tools.norm_vector(v2)[1]
    cross = jax.numpy.cross(n1, n2)
    angle_sin, axis = vector_tools.norm_vector(cross)
    angle = jax.numpy.arcsin(jax.numpy.clip(angle_sin, 1e-5, 1-1e-5))
    rotation = jax.numpy.matmul(axis_angle_to_matrix(v2, free_rotation), axis_angle_to_matrix(axis, angle))
    return rotation

def estimate_optimal_rotation(vectors_source, vectors_dest):
    H = jax.numpy.einsum('...ki,...kj->...ij',vectors_dest, vectors_source)
    U,_,Vh = jax.numpy.linalg.svd(H)
    det_product = jax.numpy.linalg.det(U)*jax.numpy.linalg.det(Vh)
    d = jax.numpy.ones(U.shape[:-2]+(H.shape[-1],)).at[...,-1].set(det_product)
    R = jax.numpy.einsum('...uk, ...k, ...kj->...uj', U, d, Vh)
    return R