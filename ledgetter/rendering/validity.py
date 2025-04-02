import jax

def validity_mask(images, local_threshold, global_threshold):
    local_cut = jax.numpy.quantile(images, local_threshold, axis=-1, keepdims=True)
    global_cut = jax.numpy.mean(images, axis=-3, keepdims=True) * 2 * global_threshold
    cut = jax.numpy.minimum(local_cut, global_cut)
    validity_mask = jax.numpy.any(images > cut, axis=-2)
    return validity_mask