import jax

def quadratic_roots(a, b, c):
    delta = jax.numpy.square(b) - 4 * a * c
    x1, x2 = (-b + jax.numpy.sqrt(delta))/(2*a),  (-b - jax.numpy.sqrt(delta))/(2*a)
    return x1, x2

def evaluate(coefficients, x):
    value = jax.numpy.sum(coefficients * jax.numpy.power(x[..., None], jax.numpy.arange(coefficients.shape[-1])), axis=-1)
    return value

def companion_matrix(coefficients):
    scaled_coefficients = coefficients / coefficients[..., -1, None]
    n = coefficients.shape[-1] - 1
    companion = jax.numpy.zeros(coefficients.shape[:-1] + (n, n))
    companion = companion.at[..., 1:, :-1].set(jax.numpy.eye(n - 1))
    companion = companion.at[..., 0, :].set(-scaled_coefficients[..., :-1][..., ::-1])
    return companion

def first_positive_root(coefficients, tol=1e-6):
    companion = companion_matrix(coefficients)
    complex_roots = jax.numpy.linalg.eigvals(companion)
    real_mask = jax.numpy.abs(jax.numpy.imag(complex_roots)) < tol
    positive_mask = jax.numpy.real(complex_roots) >= 0
    valid_roots = jax.numpy.where(jax.numpy.logical_and(positive_mask, real_mask), jax.numpy.real(complex_roots), jax.numpy.inf)
    root = jax.numpy.min(valid_roots, axis=-1)
    return root