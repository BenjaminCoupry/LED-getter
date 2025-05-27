import jax

def norm_vector(v, epsilon = 1e-8):
    """computes the norm and direction of vectors

    Args:
        v (Array ..., dim): vectors to compute the norm and direction for one vector

    Returns:
        Array ...: norms of the vectors
        Array ..., dim: unit direction vectors
    """
    s = jax.numpy.square(v)
    norm = jax.numpy.sqrt(jax.numpy.sum(jax.numpy.square(v), axis=-1) + epsilon)
    direction = v / jax.numpy.expand_dims(norm, axis=-1)
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

def cross_product_matrix(v):
    """
    Constructs the cross-product matrix (also known as the skew-symmetric matrix) 
    for a given 3D vector `v`. The cross-product matrix is a 3x3 matrix that can 
    be used to compute the cross product of `v` with another vector via matrix 
    multiplication.
    Parameters:
        v (...,3): Array of shape (3,) representing a 3D vector.
    Returns:
        Array ...,3,3: A 3x3 skew-symmetric matrix corresponding to the input vector `v`.
    """

    z, u1, u2, u3 = jax.numpy.broadcast_arrays(*((0,) + jax.numpy.unstack(v, axis=-1)))
    result = jax.numpy.stack([
        jax.numpy.stack([z, -u3,  u2], axis=-1),
        jax.numpy.stack([ u3, z, -u1], axis=-1),
        jax.numpy.stack([-u2,  u1, z], axis=-1)
    ], axis=-2)
    return result

def partial_stop_gradients(v, mask):
    """
    Applies partial stop-gradient operation to a vector `v` based on a boolean mask.
    This function prevents gradients from being computed for the elements of `v`
    where the corresponding value in `mask` is `True`. The remaining elements
    will retain their gradients during backpropagation.
    Args:
        v (jax.numpy.ndarray): The input vector for which gradients are to be partially stopped.
        mask (jax.numpy.ndarray): A boolean mask of the same shape as `v`. Elements with `True`
                                   in the mask will have their gradients stopped.
    Returns:
        jax.numpy.ndarray: A new vector with gradients stopped for the masked elements.
    """

    result = v.at[mask].set(jax.lax.stop_gradient(v[mask]))
    return result