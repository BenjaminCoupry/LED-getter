import jax
import ledgetter.image.filters as filters
import ledgetter.space.coord_systems as coord_systems
import ledgetter.image.grids as grids


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

def get_coordinates_transform(K, scale, distorsion):
    """Creates a function to apply radial distortion correction to coordinates.

    Args:
        K (Array 3,3): Camera intrinsic matrix.
        scale (float): Scaling factor.
        distorsion (Array 3,): Radial distortion coefficients.

    Returns:
        Callable: 
            Function that transforms distorted coordinates.
    """
    principal_point = K[:2,2]
    def coordinates_transform(coordinates):
        radius, phi = jax.numpy.unstack(coord_systems.cartesian_to_polar(coordinates, principal_point), axis=-1)
        radius_scale = radius / scale
        distorded_radius = scale*(radius_scale*(1 
                                                + distorsion[0]*jax.numpy.power(radius_scale, 2) 
                                                + distorsion[1]*jax.numpy.power(radius_scale, 4) 
                                                + distorsion[2]*jax.numpy.power(radius_scale, 6)))
        transformed_coordinates = coord_systems.polar_to_cartesian(jax.numpy.stack([distorded_radius, phi],axis=-1), principal_point)
        return transformed_coordinates
    return coordinates_transform

def get_undistorted_image(K, distorsion, image, kernel_span, mask=None):
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
    coordinates_transform = get_coordinates_transform(K, scale, distorsion)
    resampler = filters.get_lanczos_reampler(grid_function, kernel_span)
    def undistorted_grid(coordinates):
        transformed_coordinates = coordinates_transform(coordinates)
        resampled, mask = resampler(transformed_coordinates)
        return resampled, mask
    return undistorted_grid
