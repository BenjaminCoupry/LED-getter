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

def first_existing_file(paths):
    if any(map(os.path.isfile, paths)):
        path = next(filter(lambda p : os.path.isfile(p), paths))
    else:
        path = None
    return path

def int_to_roman(n):
    if not (0 < n < 4000):
        raise ValueError("Le nombre doit être entre 1 et 3999")
    val = [
        1000, 900, 500, 400,
        100, 90,  50, 40,
        10,  9,   5,  4, 1
    ]
    syms = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV", "I"
    ]
    roman = ""
    for i in range(len(val)):
        count = n // val[i]
        roman += syms[i] * count
        n -= val[i] * count
    return roman

def get_new_unique_name(names, name_to_add, roman=True):
    if name_to_add not in names:
        new_name = name_to_add
    else:
        name, ext = os.path.splitext(name_to_add)
        counter = 1
        counting = lambda i : int_to_roman(i) if roman else str(i)
        while f"{name}_{counting(counter)}{ext}" in names:
            counter +=1
        new_name = f"{name}_{counting(counter)}{ext}"
    return new_name