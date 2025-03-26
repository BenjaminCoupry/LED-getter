import jax

def norm_vector(v, epsilon = 1e-6):
    """computes the norm and direction of vectors

    Args:
        v (Array ..., dim): vectors to compute the norm and direction for one vector

    Returns:
        Array ...: norms of the vectors
        Array ..., dim: unit direction vectors
    """
    norm = jax.numpy.linalg.norm(v, axis=-1)
    direction = v / jax.numpy.expand_dims(norm + epsilon, axis=-1)
    return norm, direction


def build_masked(mask, data, shape=None):
    """
    Builds a masked array by setting elements from the data array at positions specified by the mask.

    Args:
        mask: A boolean array or an array of indices where data is to be placed.
        data: The data array to be inserted at positions specified by the mask.
        shape: Optional shape for the output array. If None, the shape will be derived from the mask and data.

    Returns:
        filled_array: An array with the specified shape, or the shape derived from the mask and data,
                      where elements from the data array are placed at positions specified by the mask,
                      and zeros elsewhere.
    """
    if shape is None:
        shape = jax.numpy.shape(mask) + jax.numpy.shape(data)[1:]
    filled_array = jax.numpy.zeros(shape).at[mask].set(data)
    return filled_array