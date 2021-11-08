from typing import Callable, List, TypeVar, Generic, Sequence
import inspect
import numpy as np
import math
from inspect import signature
T = TypeVar('T')

def keep_cols(data, cols):
    return data[:, cols]

def not_in_range(x, _list):
    _list = sum(_list, [])
    return any((ind >= x or ind < 0) for ind in _list)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_volume_of_sphere(d):
    return math.pi**(d/2)/math.gamma(d/2 + 1)

def check_type(arg, func, *types):
    err = f"Invalid argument in {func}: {arg} is not of type "
    vs = []
    for i in range(len(types)):
        if i == 0:
            err += f"{types[i]}"
        else:
            err += f" or {types[i]}"
        if types[i]==Callable:
            vs.append(callable(arg))
        else:
            v = False
            try:
                v = isinstance(arg, types[i])
            except:
                try:
                    vs.append(callable(arg))
                except:
                    v = False
            vs.append(v)
    if sum(vs) >= 1:
        return arg
    else:
        raise ValueError(err)