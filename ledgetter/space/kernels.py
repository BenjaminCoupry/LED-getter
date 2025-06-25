import jax

def get_lanczos_kernel(span):
    kernel = lambda x: jax.numpy.where(jax.numpy.abs(x)<span, jax.numpy.sinc(x)*jax.numpy.sinc(x/span), 0)
    return kernel

def get_gaussian_kernel(sigma):
    kernel = lambda x : jax.numpy.exp(-jax.numpy.square(x)/(2.0*jax.numpy.square(sigma)))
    return kernel