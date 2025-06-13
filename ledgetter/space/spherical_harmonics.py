import jax
import ledgetter.space.rotations as rotations
import sphericart.jax


def spherical_harmonic_indices(l_max):
    """
    Generate the indices (l, m) for spherical harmonics up to a given maximum degree l_max.
    The spherical harmonics are characterized by two indices:
    - l: the degree, which ranges from 0 to l_max.
    - m: the order, which ranges from -l to l for each degree l.
    This function returns a 2D array where each row corresponds to a pair (l, m).
    Args:
        l_max (int): The maximum degree of the spherical harmonics.
    Returns:
        jax.numpy.ndarray: A 2D array of shape (N, 2), where N is the total number of (l, m) pairs.
                           Each row contains a pair (l, m), with l in the first column and m in the second column.
    """
    l = jax.numpy.repeat(jax.numpy.arange(l_max + 1), jax.numpy.arange(1, 2*l_max + 2, 2))
    m = jax.numpy.concatenate([jax.numpy.arange(-l_i, l_i + 1) for l_i in jax.numpy.arange(l_max + 1)])
    return jax.numpy.stack((l, m), axis=-1)


def sh_function(cartesian, coefficients, l_max):
    base = sphericart.jax.spherical_harmonics(cartesian, int(l_max))
    elements = coefficients * base
    value = jax.numpy.sum(elements, axis=-1)
    return value

def oriented_sh_function(cartesian, principal_direction, free_rotation, coefficients, l_max):
    Ri = rotations.rotation_between_vectors(principal_direction, jax.numpy.asarray([0,0,1]), free_rotation)
    local_cartesian = jax.numpy.einsum('...ui, ...i -> ...u', Ri, cartesian)
    value = sh_function(local_cartesian, coefficients, l_max)
    return value

def coefficients_from_colatitude(f, l_max, steps=100, domain = 0.99):
    """
    Compute the spherical harmonics coefficients for a given function defined on the colatitude.
    Parameters:
    -----------
    f : callable
        A function that takes an array of colatitude angles (in radians) and returns the corresponding values.
    l_max : int
        The maximum degree of spherical harmonics to compute.
    steps : int, optional
        The number of discretization steps for the colatitude range. Default is 100.
    domain : float, optional
        A scaling factor for the colatitude range, where the range is [0, domain * (π/2)]. Default is 0.99.
    Returns:
    --------
    coefficients : jax.numpy.ndarray
        An array of spherical harmonics coefficients with shape `(n_coefficients, output_dim)`, where `n_coefficients`
        is determined by `l_max` and `output_dim` is the dimensionality of the output of `f`.
    indices : jax.numpy.ndarray
        An array of indices corresponding to the spherical harmonics coefficients, with shape `(n_coefficients, c)`.
    """
    teta = jax.numpy.linspace(0, domain*(jax.numpy.pi/2), steps)
    points = jax.numpy.stack([jax.numpy.zeros(teta.shape[0]), jax.numpy.sin(teta), jax.numpy.cos(teta)], axis=-1)
    basis =  sphericart.jax.spherical_harmonics(points, l_max)
    indices = spherical_harmonic_indices(l_max)
    goal_function = f(teta)
    sub_coefficients = jax.numpy.linalg.lstsq(basis[:,indices[:,1]==0], goal_function)[0]
    coefficients = jax.numpy.zeros((indices.shape[0], goal_function.shape[-1])).at[indices[:,1]==0, :].set(sub_coefficients)
    return coefficients, indices