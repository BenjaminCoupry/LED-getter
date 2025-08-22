import jax

def quadratic_roots(a, b, c):
    delta = jax.numpy.square(b) - 4 * a * c
    x1, x2 = (-b + jax.numpy.sqrt(delta))/(2*a),  (-b - jax.numpy.sqrt(delta))/(2*a)
    return x1, x2
