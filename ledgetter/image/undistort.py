import jax
import ledgetter.image.lanczos as lanczos

def cart2pol(cartesian, center):
    """Converts Cartesian coordinates to polar coordinates.

    Args:
        cartesian (Array ..., 2): Input Cartesian coordinates.
        center (Array 2,): Origin of the polar coordinate system.

    Returns:
        Array ..., 2: Polar coordinates (rho, phi).
    """
    x, y = jax.numpy.unstack(cartesian - center, axis=-1)
    rho = jax.numpy.sqrt(jax.numpy.square(x) + jax.numpy.square(y))
    phi = jax.numpy.arctan2(y, x)
    polar = jax.numpy.stack([rho, phi], axis=-1)
    return polar

def pol2cart(polar, center):
    """Converts polar coordinates to Cartesian coordinates.

    Args:
        polar (Array ..., 2): Input polar coordinates (rho, phi).
        center (Array 2,): Origin of the Cartesian coordinate system.

    Returns:
        Array ..., 2: Cartesian coordinates.
    """
    rho, phi = jax.numpy.unstack(polar, axis=-1)
    x = rho * jax.numpy.cos(phi)
    y = rho * jax.numpy.sin(phi)
    cartesian = jax.numpy.stack([x, y],axis=-1) + center
    return cartesian

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
        radius, phi = jax.numpy.unstack(cart2pol(coordinates, principal_point), axis=-1)
        radius_scale = radius / scale
        distorded_radius = scale*(radius_scale*(1 
                                                + distorsion[0]*jax.numpy.power(radius_scale, 2) 
                                                + distorsion[1]*jax.numpy.power(radius_scale, 4) 
                                                + distorsion[2]*jax.numpy.power(radius_scale, 6)))
        transformed_coordinates = pol2cart(jax.numpy.stack([distorded_radius, phi],axis=-1), principal_point)
        return transformed_coordinates
    return coordinates_transform

def get_undistorted_grid(K, scale, distorsion, grid_function, kernel_span):
    """Creates a function to undistort and resample a grid.

    Args:
        K (Array 3,3): Camera intrinsic matrix.
        scale (float): Scaling factor.
        distorsion (Array 3,): Radial distortion coefficients.
        grid_function (Callable): 
            Function retrieving grid values.
        kernel_span (int): Lanczos kernel span.

    Returns:
        Callable: 
            Function that undistorts and resamples the grid.
    """
    coordinates_transform = get_coordinates_transform(K, scale, distorsion)
    resampler = lanczos.get_lanczos_reampler(grid_function, kernel_span)
    def undistorted_grid(coordinates):
        transformed_coordinates = coordinates_transform(coordinates)
        resampled, padded = resampler(transformed_coordinates)
        return resampled, padded
    return undistorted_grid

def get_undistorted_image(K, distorsion, image, kernel_span):
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
    grid_function = lanczos.grid_from_array(jax.numpy.swapaxes(image, 0, 1))
    undistorted_grid = get_undistorted_grid(K, scale, distorsion, grid_function, kernel_span=kernel_span)
    return undistorted_grid
