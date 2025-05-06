import os
import itertools


def common_end_length(path1, path2):
    parts1 = path1.strip("/").split("/")
    parts2 = path2.strip("/").split("/")
    parts = zip(reversed(parts1), reversed(parts2))
    count = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], parts))
    return count

def find_similar_path(source, options):
    source = [source] if type(source) is not list else source
    options = [options] if type(options) is not list else options
    key = lambda pair : common_end_length(*pair)
    sorted_pairs = sorted(itertools.product(source, options), key=key, reverse=True)
    similarity, pairs_it = next(itertools.groupby(sorted_pairs, key = key), (0, None))
    elements = None if (pairs_it is None or similarity == 0) else list(pairs_it)
    return elements, similarity