import jax

def cartesian_to_polar(cartesian, center):
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

def polar_to_cartesian(polar, center):
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

def cartesian_to_spherical(cartesian, center):
    x, y, z = jax.numpy.unstack(cartesian - center, axis=-1)
    rho = jax.numpy.sqrt(jax.numpy.square(x) + jax.numpy.square(y) + jax.numpy.square(z))
    theta = jax.numpy.arccos(jax.numpy.clip(z / rho, -1.0, 1.0))
    phi = jax.numpy.mod(jax.numpy.arctan2(y, x), 2 * jax.numpy.pi)
    spherical = jax.numpy.stack([rho, theta, phi], axis=-1)
    return spherical