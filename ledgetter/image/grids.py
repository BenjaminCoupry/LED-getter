import jax
import itertools
import functools
import ledgetter.utils.vector_tools as vector_tools

def grid_from_array(array, valid_mask=None):
    """Creates a grid function that retrieves values from an array.

    Args:
        array (Array ..., v): Input array.

    Returns:
        Callable: 
            A function that retrieves values and masks from the array given coordinates.
    """
    shape = jax.numpy.asarray(array.shape[:-1])
    def grid_function(x, array):
        coordinates = jax.numpy.clip(jax.numpy.round(x).astype(int), 0, shape-1)
        bound_mask = jax.numpy.all(jax.numpy.logical_and(x>0, x<=shape-1), axis=-1)
        coordinates_tuple = tuple(jax.numpy.unstack(coordinates, axis=-1))
        value = array[coordinates_tuple]
        mask = bound_mask if valid_mask is None else jax.numpy.logical_and(bound_mask, valid_mask[coordinates_tuple])
        return value, mask
    return functools.partial(grid_function, array=array)

def get_kernel_reampler(grid_function, shifts, kernel):
    def resampler(x):
        indices = jax.numpy.array(list(itertools.product(range(jax.numpy.shape(shifts)[0]), repeat=jax.numpy.shape(x)[-1])))
        x_floor = jax.numpy.floor(x).astype(int)
        kernel_values = kernel((x - x_floor) - jax.numpy.expand_dims(shifts,axis=-1))
        values, masks = grid_function(x_floor + shifts[indices])
        coefficients = jax.numpy.prod(jax.numpy.take_along_axis(kernel_values, indices, axis=0), axis=-1)
        result = jax.numpy.sum(jax.numpy.expand_dims(coefficients, axis=-1) * values, axis=0)
        padded = masks.shape[0] - jax.numpy.sum(masks, axis=0)
        return result, padded
    vectorized_resampler = jax.numpy.vectorize(resampler, signature='(k)->(v),()')
    return vectorized_resampler

def get_neighbour_reducer(displacements, reducer, *grid_functions, out_morphology = '(v)'):
    def resampler(x):
        values, masks = zip(*map(lambda grid_function: grid_function(x + displacements), grid_functions))
        mask = jax.numpy.all(jax.numpy.stack(masks, axis=-1), axis=-1)
        result = reducer(mask, *values)
        return result
    vectorized_resampler = jax.numpy.vectorize(resampler, signature=f'(k)->{out_morphology}')
    return vectorized_resampler


def get_displacements(size):
    get_range = lambda s : itertools.chain(itertools.repeat(0, 1), range(-s, 0), range(1, s+1))
    displacements = jax.numpy.asarray(list(itertools.product(*map(get_range, size))), dtype=jax.numpy.int32)
    return displacements