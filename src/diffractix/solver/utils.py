
import inspect

def callable_arity(function):
    signature = inspect.signature(function)
    accepts_zero = accepts_one = False
    try:
        signature.bind()
        accepts_zero = True
    except TypeError:
        pass
    try:
        signature.bind(object())
        accepts_one = True
    except TypeError:
        pass
    if accepts_zero == accepts_one:
        raise TypeError("Callable must accept exactly zero or one argument.")
    return 0 if accepts_zero else 1