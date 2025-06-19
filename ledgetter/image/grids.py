import jax
import itertools
import functools
import ledgetter.utils.vector_tools as vector_tools

def grid_from_array(array):
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
        mask = jax.numpy.all(jax.numpy.logical_and(x>0, x<=shape-1), axis=-1)
        coordinates_tuple = tuple(jax.numpy.unstack(coordinates, axis=-1))
        value = array[coordinates_tuple]
        return value, mask
    return functools.partial(grid_function, array=array)

def get_coordinate_to_index(mask, coordinate):
    structured_indices = vector_tools.build_masked(mask, jax.numpy.arange(jax.numpy.sum(mask)), fill_value=-1)
    index = structured_indices[tuple(jax.numpy.unstack(coordinate, axis=-1))]
    return index

def get_index_to_coordinate(mask, index):
    flat_coordinates = jax.numpy.stack(jax.numpy.where(mask), axis=-1)
    coordinate = flat_coordinates[index]
    return coordinate

def get_displacements(size):
    shifts = jax.numpy.asarray(list(itertools.product(*map(lambda s : range(-s, s+1), size))), dtype=jax.numpy.int32)
    return shifts