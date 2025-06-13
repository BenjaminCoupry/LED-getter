import jax
import itertools


def get_chuncker(n_chuncks):
    n_chuncks_t = n_chuncks if isinstance(n_chuncks, tuple) else (n_chuncks,)
    chuncker = itertools.product(*map(lambda n : map(lambda i : slice(int(i), None, int(n)), range(n)), tuple(n_chuncks_t)))
    total_chuncks = jax.numpy.int32(jax.numpy.prod(jax.numpy.asarray(n_chuncks)))
    return chuncker, total_chuncks

def chunckwise_treatement(treatement, state, chunckable_args, atomic_args, n_chuncks, output=None, elementary_shape = None):
    results = None
    chuncker, total_chuncks = get_chuncker(n_chuncks)
    for i, chunck in enumerate(chuncker):
        chuncked_args = jax.tree_util.tree_map(lambda u : u[chunck], chunckable_args)
        args = chuncked_args | atomic_args
        chuncked_results, state, metric = treatement(state, chunck, **args)
        if results is None:
            if elementary_shape is None:
                elementary_shape = jax.tree_util.tree_leaves(chunckable_args)[0].shape[:len(chunck)]
            results = jax.tree_util.tree_map(lambda r : jax.numpy.empty(elementary_shape + r.shape[len(chunck):], r.dtype), chuncked_results)
        results = jax.tree_util.tree_map(lambda u, v : u.at[chunck].set(v), results, chuncked_results)
        if output is not None:
            output(metric, i, total_chuncks-1)
    return results, state

def split_dict(dictionary, predicate):
    if isinstance(predicate, set):
        predicate_set = predicate
        predicate = lambda k : k in predicate_set
    if isinstance(predicate, tuple) and len(predicate)==2 and isinstance(predicate[0], set) and isinstance(predicate[1], set):
        in_set, out_set = predicate
        predicate = lambda k : True if k in in_set else (False if k in out_set else None)
    true_part = {k: v for k, v in dictionary.items() if predicate(k) is True}
    false_part = {k: v for k, v in dictionary.items() if predicate(k) is False}
    return true_part, false_part

# def compute_nd_sections(shape, n_sections):
#     shape_arr = jax.numpy.asarray(shape)
#     dimension = shape_arr.shape[0]
#     ratios = shape_arr/jax.numpy.min(shape_arr)
#     order = jax.numpy.argsort(ratios, descending=True)
#     sections_d = jax.numpy.empty(dimension, jax.numpy.int32)
#     for i in range(dimension):
#         allocated = jax.numpy.prod(sections_d[order[:i]])
#         remaining_sections, remaining_dims, remaining_ratios = n_sections/allocated, dimension-i, ratios[order[i:]]
#         scale = jax.numpy.power(remaining_sections/jax.numpy.prod(remaining_ratios), 1.0/(remaining_dims))
#         soft_section = jax.numpy.clip(scale*ratios[order[i]], 1, remaining_sections)
#         sections_d = sections_d.at[order[i]].set(jax.numpy.int32(jax.numpy.floor(soft_section)))
#     max_index = jax.numpy.prod(sections_d)
#     return sections_d, max_index