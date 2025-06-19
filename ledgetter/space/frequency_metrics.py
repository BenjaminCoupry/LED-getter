import jax
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.space.rotations as rotations
import ledgetter.space.kernels as kernels

def angular_error(test_normals, reference_normals):
    dot_product = jax.numpy.sum(vector_tools.norm_vector(test_normals)[1] * vector_tools.norm_vector(reference_normals)[1], axis=-1)
    error = jax.numpy.rad2deg(jax.numpy.arccos(jax.numpy.clip(dot_product,-1,1)))
    return error

def low_frequency_angular_error(mask, test_normals, reference_normals, test_points, reference_points, sigma):
    gaussian_test_normal = gaussian_normals(test_points[0,:], test_normals, test_points, mask, sigma)
    gaussian_reference_normal = gaussian_normals(reference_points[0,:], reference_normals, reference_points, mask, sigma)
    error = angular_error(gaussian_test_normal, gaussian_reference_normal)
    return error

def high_frequency_angular_error(mask, test_normals, reference_normals, test_points, reference_points, sigma, rotation_basis_size):
    gaussian_test_normal = gaussian_normals(test_points[0,:], test_normals, test_points, mask, sigma)
    gaussian_reference_normal = gaussian_normals(reference_points[0,:], reference_normals, reference_points, mask, sigma)
    distances = jax.numpy.linalg.norm(test_points[:,0] - test_points, axis=-1)
    _, nearest_indices = jax.lax.top_k(-jax.numpy.where(mask, distances, jax.numpy.inf), rotation_basis_size)
    R = rotations.estimate_optimal_rotation(gaussian_test_normal[nearest_indices], gaussian_reference_normal[nearest_indices])
    rotated_test_normals = jax.numpy.einsum('...ik,...k->...i', R, test_normals)
    error = angular_error(rotated_test_normals, reference_normals)
    return error, R

def gaussian_normals(point, normals, points, mask, sigma):
    kernel = kernels.get_gaussian_kernel(sigma)
    distances = jax.numpy.linalg.norm(point - points, axis=-1)
    weights = kernel(distances) * mask
    mean_normals = jax.numpy.average(normals, axis=-2, weights=weights)
    filtered_normals = vector_tools.norm_vector(mean_normals)[1]
    return filtered_normals