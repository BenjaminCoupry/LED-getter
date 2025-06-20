import jax
import itertools
import functools
import ledgetter.image.grids as grids

def lanczos_kernel(x, span):
    """Computes the Lanczos kernel for resampling.

    Args:
        x (Array ...): Input coordinate offsets.
        span (int): Lanczos kernel span.

    Returns:
        Array ...: Computed Lanczos kernel values.
    """
    kernel = jax.numpy.where(jax.numpy.abs(x)<span, jax.numpy.sinc(x)*jax.numpy.sinc(x/span), 0)
    return kernel

def get_lanczos_reampler(grid_function, span):
    """Creates a Lanczos resampler using a given grid function.

    Args:
        grid_function : 
            Function that retrieves values and masks from a grid.
        span (int): Lanczos kernel span.

    Returns:
        Callable: 
            A vectorized resampling function that interpolates values from the grid.
    """
    shifts = jax.numpy.arange(-span + 1, span + 1)
    def resampler(x):
        indices = jax.numpy.array(list(itertools.product(range(2 * span), repeat=jax.numpy.shape(x)[-1])))
        x_floor = jax.numpy.floor(x).astype(int)
        lanczos_values = lanczos_kernel((x - x_floor) - jax.numpy.expand_dims(shifts,axis=-1), span)
        values, masks = grid_function(x_floor + shifts[indices])
        coefficients = jax.numpy.prod(jax.numpy.take_along_axis(lanczos_values, indices, axis=0), axis=-1)
        result = jax.numpy.sum(jax.numpy.expand_dims(coefficients, axis=-1) * values, axis=0)
        padded = masks.shape[0] - jax.numpy.sum(masks, axis=0)
        return result, padded
    vectorized_resampler = jax.numpy.vectorize(resampler, signature='(k)->(v),()')
    return vectorized_resampler