import numpy as np

# miscellaneous
from itertools import cycle
from copy import deepcopy

import logging
import warnings
import dataclasses
from dataclasses import dataclass
from typing import Optional, List

from functools import reduce

def theta(x):
    return 0.5*(1 + sign(x))


def f(x):
    return np.tanh(x)


def df(x):
    """The derivative of tanh(x) = 1/cosh(x)^2"""
    return 1/np.cosh(10*np.tanh(x/10))**2  # the tanh prevents overflow

def rgetattr(obj, attr):
    """A "recursive" version of getattr that can handle nested objects.
    Args:
        obj (object): Parent object
        attr (string): Address of desired attribute with '.' between child
            objects.
    Returns:
        The attribute of obj referred to."""

    return reduce(getattr, [obj] + attr.split('.'))