import jax
import ledgetter.image.filters as filters
import ledgetter.space.coord_systems as coord_systems
import ledgetter.image.grids as grids
import ledgetter.optim.polynomials as polynomials
import functools


def get_scale(K, size):
    """Computes a scaling factor from the intrinsic matrix.

    Args:
        K (Array 3,3): Camera intrinsic matrix.
        size (Tuple[int, int]): Image size (width, height).

    Returns:
        float: Computed scale factor.
    """
    width, height = size
    f_x = K[0,0]
    scale = f_x/width*jax.numpy.maximum(width, height)
    return scale

def image_to_pinhole_radius(radius, distorsion, scale, steps=5000):
    radius_scale = radius / scale
    unique_radius_scale = jax.numpy.linspace(0, jax.numpy.max(radius_scale), steps)
    polynomial_coefficients = jax.numpy.asarray([0, 1, 0, distorsion[0], 0, distorsion[1], 0, distorsion[2]])
    unique_polynomial_coefficients = jax.numpy.zeros((steps, 8)).at[...].set(polynomial_coefficients).at[:, 0].set(-unique_radius_scale)
    unique_distorted_radius = scale*polynomials.first_positive_root(unique_polynomial_coefficients)
    distorted_radius = jax.numpy.interp(radius_scale, unique_radius_scale, unique_distorted_radius)
    return distorted_radius

def pinhole_to_image_radius(radius, distorsion, scale):
    radius_scale = radius / scale
    polynomial_coefficients = jax.numpy.asarray([0, 1, 0, distorsion[0], 0, distorsion[1], 0, distorsion[2]])
    distorted_radius = scale * polynomials.evaluate(polynomial_coefficients, radius_scale)
    return distorted_radius


def get_coordinates_transform(K, size, distorsion, image_to_pinhole=False):
    """Creates a function to apply radial distortion correction to coordinates.

    Args:
        K (Array 3,3): Camera intrinsic matrix.
        size (Tuple[int, int]): Image size (width, height)
        distorsion (Array 3,): Radial distortion coefficients.

    Returns:
        Callable: 
            Function that transforms distorted coordinates.
    """
    scale = get_scale(K, size)
    principal_point = K[:2,2]
    def coordinates_transform(coordinates):
        radius, phi = jax.numpy.unstack(coord_systems.cartesian_to_polar(coordinates, principal_point), axis=-1)
        if image_to_pinhole:
            distorted_radius = image_to_pinhole_radius(radius, distorsion, scale)
        else:
            distorted_radius = pinhole_to_image_radius(radius, distorsion, scale)
        transformed_coordinates = coord_systems.polar_to_cartesian(jax.numpy.stack([distorted_radius, phi],axis=-1), principal_point)
        return transformed_coordinates
    return coordinates_transform

@functools.partial(jax.jit, backend='gpu', static_argnames=('kernel_span',))
def undistorted_image(K, distorsion, image, coordinates, kernel_span, mask=None):
    """Generates an undistorted version of an image.

    Args:
        K (Array 3,3): Camera intrinsic matrix.
        distorsion (Array 3,): Radial distortion coefficients.
        image (Array H,W,C): Input distorted image.
        kernel_span (int): Lanczos kernel span.

    Returns:
        Callable: 
            Function that produces undistorted image values.
    """
    scale = get_scale(K, (image.shape[1],image.shape[0]))
    grid_function = grids.get_grid_from_array(jax.numpy.swapaxes(image, 0, 1), valid_mask = (jax.numpy.swapaxes(mask, 0, 1) if mask is not None else None))
    coordinates_transform = get_coordinates_transform(K, scale, distorsion, image_to_pinhole=False) #pinhole_to_image
    resampler = filters.get_lanczos_reampler(grid_function, kernel_span)
    transformed_coordinates = coordinates_transform(coordinates)
    resampled, mask = resampler(transformed_coordinates)
    return resampled, mask
