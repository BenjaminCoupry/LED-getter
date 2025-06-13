import inspect
import functools
import networkx

def force_positional(func):
    sig = inspect.signature(func)
    param_names = [
        p.name for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ordered_args = [kwargs[name] for name in param_names[len(args):]]
        return func(*args, *ordered_args)
    return wrapper


def structured_return(keys):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            anonymous = tuple(v for k, v in zip(keys, results) if k is None)
            named = {k: v for k, v in zip(keys, results) if k is not None}
            return anonymous, named
        return wrapper
    return decorator


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

def generator(inputs, outputs, registry):
    def decorator(fn):
        registry.update({output: {'inputs': set(inputs), 'fn': fn} for output in outputs})
        return fn
    return decorator

def execute_generators(registry, required_outputs, initial_parameters, **kwargs):
    graph = networkx.DiGraph()
    def add_generating_fn(parameter):
        if parameter not in initial_parameters:
            if parameter in registry:
                generating_fn = registry[parameter]['fn']
            else:
                raise ValueError(f"Cannot generate required parameter: {parameter}")
            if generating_fn not in graph:
                graph.add_node(generating_fn)
                for input in registry[parameter]['inputs']:
                    if input in registry and input not in initial_parameters:
                        add_generating_fn(input)
                        graph.add_edge(registry[input]['fn'], generating_fn)
    for parameter in required_outputs:
        add_generating_fn(parameter)
    parameters = {**initial_parameters}
    for fn in networkx.topological_sort(graph):
        output = fn(**parameters, **kwargs)
        parameters = output | parameters 
    return parameters

