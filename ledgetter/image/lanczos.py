import jax
import itertools
import functools
import ledgetter.space.kernels as kernels
import ledgetter.image.grids as grids


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
    kernel = kernels.get_lanczos_kernel(span)
    shifts = jax.numpy.arange(-span + 1, span + 1)
    resampler = grids.get_kernel_reampler(grid_function, shifts, kernel)
    return resampler