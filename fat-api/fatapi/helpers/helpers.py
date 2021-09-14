from typing import Callable

def keep_cols(data, cols):
    return data[:, cols]

def not_in_range(x, _list):
    return any((ind >= x or ind < 0) for ind in _list)

def check_type(arg, type_, func):
    if type_==Callable:
        if callable(arg):
            return arg
        else:
            raise ValueError(f"Invalid argument in {func}: {arg} is not a function")
    else:
        if type(arg)==type_:
            return arg
        else:
            raise ValueError(f"Invalid argument in {func}: {arg} is not of type {type_}")