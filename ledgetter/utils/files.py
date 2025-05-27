import os
import itertools


def common_end_length(path1, path2):
    """
    Calculate the number of common trailing components in two file paths.

    This function compares the components of two file paths from the end
    (rightmost part) and counts how many components are identical.

    Args:
        path1 (str): The first file path.
        path2 (str): The second file path.

    Returns:
        int: The number of common trailing components in the two paths.
    """
    parts1 = path1.strip("/").split("/")
    parts2 = path2.strip("/").split("/")
    parts = zip(reversed(parts1), reversed(parts2))
    count = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], parts))
    return count

def find_similar_path(source, options):
    """
    Finds the most similar paths between a source path (or list of paths) and a list of option paths
    based on their common ending segments.
    Args:
        source (str or list of str): The source path or list of paths to compare.
        options (str or list of str): The option path or list of paths to compare against.
    Returns:
        tuple:
            - elements (list of tuple or None): A list of tuples containing the source and option 
              paths with the highest similarity, or None if no similarity is found.
            - similarity (int): The length of the common ending segments between the most similar paths.
    """
    source = [source] if type(source) is not list else source
    options = [options] if type(options) is not list else options
    key = lambda pair : common_end_length(*pair)
    sorted_pairs = sorted(itertools.product(source, options), key=key, reverse=True)
    similarity, pairs_it = next(itertools.groupby(sorted_pairs, key = key), (0, None))
    elements = None if (pairs_it is None or similarity == 0) else list(pairs_it)
    return elements, similarity