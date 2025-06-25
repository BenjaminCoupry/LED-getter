import jax
import ledgetter.utils.vector_tools as vector_tools
import ledgetter.space.rotations as rotations
import ledgetter.space.kernels as kernels
import ledgetter.image.grids as grids
import ledgetter.image.filters as filters
import functools


def angular_error(test_normals, reference_normals):
    dot_product = jax.numpy.sum(vector_tools.norm_vector(test_normals)[1] * vector_tools.norm_vector(reference_normals)[1], axis=-1)
    error = jax.numpy.rad2deg(jax.numpy.arccos(jax.numpy.clip(dot_product,-1,1)))
    return error

def estimate_rotation(test_normals, reference_normals, points, mask, sigma, span, batch_size=None):
    points_grid = grids.build_masked_grid(mask, points)
    test_normals_grid = grids.build_masked_grid(mask, test_normals)
    reference_normals_grid = grids.build_masked_grid(mask, reference_normals)
    normals_grid = grids.get_concatenated_grids((test_normals_grid, reference_normals_grid), axis=-1)
    proximal_grid = grids.get_proximal_grid(points_grid, kernels.get_gaussian_kernel(sigma))
    def coefficient_treatement(x, neighbouring_values, coefficients, masks):
        test_normals, reference_normals = jax.numpy.split(neighbouring_values, 2, axis=-1)
        value = rotations.estimate_optimal_rotation(test_normals, reference_normals, coefficients)
        return value
    rotation_grid = filters.get_proximal_treatement(normals_grid, proximal_grid, span, coefficient_treatement, normalise=True, filter_valid=True)
    R, _ = jax.lax.map(rotation_grid, jax.numpy.argwhere(mask), batch_size=batch_size)
    return R

def frequency_angular_errors(test_normals, reference_normals, points, mask, sigma, span, batch_size=None):
    filtered_test_normals = vector_tools.norm_vector(filters.apply_spatial_gaussian_filter(test_normals, points, mask, span, sigma, batch_size=batch_size))[1]
    filtered_reference_normals =  vector_tools.norm_vector(filters.apply_spatial_gaussian_filter(reference_normals, points, mask, span, sigma, batch_size=batch_size))[1]
    R = estimate_rotation(filtered_test_normals, filtered_reference_normals, points, mask, sigma, span, batch_size=batch_size)
    rotated_test_normal = jax.numpy.einsum('...ik, ...k -> ...i', R, test_normals)
    hf_error = angular_error(rotated_test_normal, reference_normals)
    lf_error = angular_error(filtered_test_normals, filtered_reference_normals)
    return hf_error, lf_error
