import jax
import itertools
import functools
import ledgetter.space.kernels as kernels
import ledgetter.image.grids as grids
import ledgetter.utils.vector_tools as vector_tools



def get_nbh_treatement(grid, span, treatement):
    def treated(x):
        displacements = vector_tools.get_displacements([span]*x.shape[-1])
        shifted_x = jax.numpy.expand_dims(x, axis=-2) + displacements
        neighbouring_values, value_masks = grid(shifted_x)
        value = treatement(x, neighbouring_values, displacements, value_masks)
        mask = grid(x)[1]
        return value, mask
    return treated

def get_proximal_treatement(grid, proximal_grid, span, coefficient_treatement, normalise=False, filter_valid=False):
    def treatement(x, neighbouring_values, displacements, value_masks):
        coefficients, proximal_masks = proximal_grid(jax.numpy.expand_dims(x, axis=-2), displacements)
        masks = jax.numpy.logical_and(proximal_masks, value_masks)
        coefficients = coefficients*masks if filter_valid else coefficients
        coefficients = coefficients/jax.numpy.sum(coefficients, axis=-1, keepdims=True) if normalise else coefficients
        value = coefficient_treatement(x, neighbouring_values, coefficients, masks)
        return value
    proximal_treatement = get_nbh_treatement(grid, span, treatement)
    return proximal_treatement

def get_local_filter(grid, proximal_grid, span, normalise=False, filter_valid=False):
    def coefficient_treatement(x, neighbouring_values, coefficients, masks):
        vec_dim = jax.numpy.ndim(neighbouring_values) - jax.numpy.ndim(masks)
        weighted_neighbouring_value = neighbouring_values * jax.numpy.reshape(coefficients, coefficients.shape + (1,)*vec_dim)
        value = jax.numpy.sum(weighted_neighbouring_value, axis=-vec_dim-1)
        return value
    local_filter = get_proximal_treatement(grid, proximal_grid, span, coefficient_treatement, normalise=normalise, filter_valid=filter_valid)
    return local_filter

def get_lanczos_reampler(grid, span):
    """Creates a Lanczos resampler using a given grid function.

    Args:
        grid_function : 
            Function that retrieves values and masks from a grid.
        span (int): Lanczos kernel span.

    Returns:
        Callable: 
            A vectorized resampling function that interpolates values from the grid.
    """
    kernel = kernels.get_lanczos_kernel(span)
    def proximal_grid(x, dx):
        kernel_values = kernel(x - jax.numpy.floor(x)- dx)
        value = jax.numpy.prod(kernel_values, axis=-1)
        return value, jax.numpy.ones(tuple(), dtype=bool)
    filtered = get_local_filter(grid, proximal_grid, span, normalise=False, filter_valid=False)
    return filtered

def get_spatial_gaussian_filter(grid, grid_points, span, sigma):
    kernel = kernels.get_gaussian_kernel(sigma)
    proximal_grid = grids.get_proximal_grid(grid_points, kernel)
    filtered = get_local_filter(grid, proximal_grid, span, normalise=True, filter_valid=True)
    return filtered

def apply_spatial_gaussian_filter(values, points, mask, span, sigma, batch_size = None):
    values_grid = grids.build_masked_grid(mask, values)
    points_grid = grids.build_masked_grid(mask, points)
    gaussian_grid = get_spatial_gaussian_filter(values_grid, points_grid, span, sigma)
    filtered_values, _ = jax.lax.map(gaussian_grid, jax.numpy.argwhere(mask), batch_size=batch_size)
    return filtered_values
