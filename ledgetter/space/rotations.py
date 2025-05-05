
import jax
import ledgetter.utils.vector_tools as vector_tools

def axis_angle_to_matrix(axis, angle):
    K = vector_tools.cross_product_matrix(axis)
    R = jax.numpy.eye(3) + jax.numpy.expand_dims(jax.numpy.sin(angle), axis=(-1,-2)) * K + jax.numpy.expand_dims(1 - jax.numpy.cos(angle), axis=(-1,-2)) * jax.numpy.matmul(K, K)
    return R

def rotation_between_vectors(v1, v2, free_rotation):
    n1, n2 = vector_tools.norm_vector(v1)[1], vector_tools.norm_vector(v2)[1]
    cross = jax.numpy.cross(n1, n2)
    angle_sin, axis = vector_tools.norm_vector(cross)
    angle = jax.numpy.arcsin(jax.numpy.clip(angle_sin, 1e-5, 1-1e-5))
    rotation = jax.numpy.matmul(axis_angle_to_matrix(v2, free_rotation), axis_angle_to_matrix(axis, angle))
    return rotation