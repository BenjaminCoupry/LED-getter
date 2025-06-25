import jax
import itertools
import functools
import ledgetter.utils.vector_tools as vector_tools

def get_grid_from_array(array, valid_mask=None):
    """Creates a grid function that retrieves values from an array.

    Args:
        array (Array ..., v): Input array.

    Returns:
        Callable: 
            A function that retrieves values and masks from the array given coordinates.
    """
    def grid_function(x):
        x_shape = x.shape[-1]
        bounds = jax.numpy.asarray(jax.numpy.shape(array)[:x_shape])
        coordinates = jax.numpy.clip(jax.numpy.floor(x).astype(int), 0, bounds-1)
        bound_mask = jax.numpy.all(jax.numpy.logical_and(x>0, x<=bounds-1), axis=-1)
        coordinates_tuple = tuple(jax.numpy.unstack(coordinates, axis=-1))
        value = array[coordinates_tuple]
        mask = bound_mask if valid_mask is None else jax.numpy.logical_and(bound_mask, valid_mask[coordinates_tuple])
        return value, mask
    return grid_function

def build_masked_grid(mask, data):
    array = vector_tools.build_masked(mask, data)
    grid_function = get_grid_from_array(array, mask)
    return grid_function

def get_proximal_grid(grid, kernel):
    def proximal_grid(x, dx):
        (x_value, x_mask), (dx_value, dx_mask) = grid(x), grid(x + dx)
        mask = jax.numpy.logical_and(x_mask, dx_mask)
        kernel_values = jax.numpy.nan_to_num(kernel(dx_value - x_value))
        value = jax.numpy.prod(kernel_values, axis = tuple(range(-kernel_values.ndim+mask.ndim, 0)))
        return value, mask
    return proximal_grid

def compute_grid(grid, mask, batch_size = None):
    coordinates = jax.numpy.argwhere(mask)
    v_values, v_mask = jax.lax.map(grid, coordinates, batch_size=batch_size)
    g_values, g_mask = vector_tools.build_masked(mask, v_values), vector_tools.build_masked(mask, v_mask)
    grid = get_grid_from_array(g_values, g_mask)
    return grid

def get_concatenated_grids(grids, axis):
    def concatenated_grids(x):
        values, masks = zip(*map(lambda g : g(x), grids))
        value = jax.numpy.concatenate(values, axis=axis)
        mask = jax.numpy.all(jax.numpy.stack(masks, axis=-1),axis=-1)
        return value, mask
    return concatenated_grids

