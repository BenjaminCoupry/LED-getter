import inspect
import functools


def filter_args(func):
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
    @functools.wraps(factory_func)
    def wrapper(*args, **kwargs):
        result_func = factory_func(*args, **kwargs)
        return filter_args(result_func)
    return wrapper