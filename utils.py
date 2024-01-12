#!/usr/bin/python

from types import SimpleNamespace as N
from operator import add, sub, and_, mul as multiply
import numpy as np
from numpy import array as V

def first(*args):
    return next(( a for a in args if a is not None ), None)

def select_keys(d, keys):
    return { k:v for k in keys if k in d }

def smart_call(f, x):
    t = type(x)
    if t is dict:
        return f(**x)
    elif t is N:
        return f(**vars(x))
    else:
        return f(x)

# TODO: can we return np array here?
def vector3(v=None,
            x=None, y=None, z=None,
            xy=None, xz=None, yz=None,
            xyz=None,
            default=0, combiner=add):
    if v is None:
        v = (default, default, default)
    elif type(v) in {int, float}:
        v = (v, v, v) # uniform
    assert len(v) == 3 # otherwise v must be vector
    return (combiner(v[0], first(x, xy, xz, xyz, default)),
            combiner(v[1], first(y, xy, yz, xyz, default)),
            combiner(v[2], first(z, xz, yz, xyz, default)))


def vector2(v=None,
            x=None, y=None,
            xy=None,
            default=0, combiner=add):
    if v is None:
        v = (default, default)
    elif type(v) in {int, float}:
        v = (v, v) # uniform
    assert len(v) == 2 # otherwise v must be vector
    return (combiner(v[0], first(x, xy, default)),
            combiner(v[1], first(y, xy, default)))

# an alternative without 0 to np.sign (avoids np.sign(0) = 0)
def sign2(x): return np.where(x >= 0, +1, -1)
