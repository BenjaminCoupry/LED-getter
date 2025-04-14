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

def to_homogeneous(v):
    """converts vectors to homogeneous coordinates

    Args:
        v (Array ..., dim): input vectors

    Returns:
        Array ..., dim+1: homogeneous coordinates of the input vectors
    """
    append_term = jax.numpy.ones(jax.numpy.shape(v)[:-1]+(1,))
    homogeneous = jax.numpy.append(v,append_term,axis=-1)
    return homogeneous


def build_masked(mask, data, shape=None, fill_value = 0):
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
    filled_array = jax.numpy.full(shape, dtype = jax.numpy.dtype(data), fill_value = fill_value).at[mask].set(data)
    return filled_array