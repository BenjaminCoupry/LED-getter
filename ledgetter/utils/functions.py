import inspect
import functools


def filter_args(func):
    """
    A decorator that filters keyword arguments passed to a function, allowing only those
    that are explicitly defined in the function's signature as either keyword-only or
    positional-or-keyword arguments.
    Args:
        func (Callable): The function to be wrapped by the decorator.
    Returns:
        Callable: A wrapped version of the input function that filters its keyword arguments.
    Example:
        >>> @filter_args
        ... def example_function(a, b, *, c=None):
        ...     return a, b, c
        ...
        >>> example_function(1, 2, c=3, d=4)  # 'd' is ignored
        (1, 2, 3)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        partial_func = functools.partial(func, *args)
        allowed_params = {
            k for k, v in inspect.signature(partial_func).parameters.items()
            if v.kind in (v.KEYWORD_ONLY, v.POSITIONAL_OR_KEYWORD)
        }
        return partial_func(**{k: v for k, v in kwargs.items() if k in allowed_params})

    return wrapper

def filter_output_args(factory_func):
    """
    A decorator that wraps a factory function and applies the `filter_args` 
    function to the result of the factory function.
    Args:
        factory_func (callable): The factory function to be wrapped. It should 
        return a callable that takes arguments to be filtered.
    Returns:
        callable: A wrapped version of the factory function, where the result 
        of the factory function is passed through `filter_args`.
    """
    @functools.wraps(factory_func)
    def wrapper(*args, **kwargs):
        result_func = factory_func(*args, **kwargs)
        return filter_args(result_func)
    return wrapper