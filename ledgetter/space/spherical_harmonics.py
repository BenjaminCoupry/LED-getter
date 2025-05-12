import jax
import functools
import sphericart.jax


def spherical_harmonic_indices(l_max):
    l = jax.numpy.repeat(jax.numpy.arange(l_max + 1), jax.numpy.arange(1, 2*l_max + 2, 2))
    m = jax.numpy.concatenate([jax.numpy.arange(-l_i, l_i + 1) for l_i in jax.numpy.arange(l_max + 1)])
    return jax.numpy.stack((l, m), axis=-1)


def sh_function(cartesian, coefficients, l_max):
    base = sphericart.jax.spherical_harmonics(cartesian, int(l_max))
    elements = coefficients * base
    value = jax.numpy.sum(elements, axis=-1)
    return value

def coefficients_from_colatitude(f, l_max, steps=100, domain = 0.99):
    teta = jax.numpy.linspace(0, domain*(jax.numpy.pi/2), steps)
    points = jax.numpy.stack([jax.numpy.zeros(teta.shape[0]), jax.numpy.sin(teta), jax.numpy.cos(teta)], axis=-1)
    basis =  sphericart.jax.spherical_harmonics(points, l_max)
    indices = spherical_harmonic_indices(l_max)
    goal_function = f(teta)
    sub_coefficients = jax.numpy.linalg.lstsq(basis[:,indices[:,1]==0], goal_function)[0]
    coefficients = jax.numpy.zeros((indices.shape[0], goal_function.shape[-1])).at[indices[:,1]==0, :].set(sub_coefficients)
    return coefficients, indices