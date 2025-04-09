import jax
import ledgetter.utils.vector_tools as vector_tools
import scipy.ndimage

def validity_mask(images, local_threshold, global_threshold, dilation, erosion, mask):
    local_cut = jax.numpy.quantile(images, local_threshold, axis=-1, keepdims=True) if local_threshold else jax.numpy.inf
    global_cut = (jax.numpy.mean(images, axis=-3, keepdims=True) * 2 * global_threshold) if global_threshold else jax.numpy.inf
    cut = jax.numpy.minimum(local_cut, global_cut)
    validity_mask = jax.numpy.any(images > cut, axis=-2)
    structured_mask = vector_tools.build_masked(mask, validity_mask, fill_value=True)
    dilated_mask = scipy.ndimage.binary_dilation(structured_mask, iterations = dilation, border_value=True, axes=(0,1)) if (dilation and dilation > 0) else structured_mask
    eroded_mask = scipy.ndimage.binary_erosion(dilated_mask, iterations = erosion, border_value=True, axes=(0,1)) if (erosion and erosion > 0) else dilated_mask
    validity_mask = eroded_mask[mask]
    return validity_mask