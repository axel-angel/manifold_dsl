#!/usr/bin/python

import numpy as np
from types import SimpleNamespace as N

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

def vector3(v=(0,0,0),
            x=None, y=None, z=None,
            xy=None, xz=None, yz=None):
    if type(v) in {int, float}:
        v = (v, v, v) # uniform
    # otherwise assume v is vector
    return (v[0] + first(x, xy, xz, 0),
            v[1] + first(y, xy, yz, 0),
            v[2] + first(z, xz, yz, 0))


# parse strings to specify orientation, eg: +xy, +x-z, by default =xyz means centered
# = means centered, +/- towards positive/negative respectively
def parse_orient(s):
    signs = dict(x=0, y=0, z=0) # default is centered
    sign = +1
    for x in s:
        if (s := {'=':0, '+':+1, '-':-1}.get(x, None)) != None: # sign symbol
            sign = s
        elif x in 'xyz': # coordinate symbol
            signs[x] = sign
        else:
            raise Exception(f'Invalid symbol {x}')
    return tuple(( signs[c] for c in 'xyz' ))

def compute_orient(sign, min_, max_):
    if   sign  > 0: return -min(min_, max_)
    elif sign == 0: return -(max_ + min_)/2
    else          : return -max(min_, max_)


class Object(N):
    def __init__(self, template=None, **overrides):
        super().__init__(
                **{# defaults
                   'transformations': [],
                   # template and overrides
                   **(vars(template) if template else {}),
                   **overrides})

    def transform(self, type, *args, **kwargs):
        # TODO: recompute bbox
        v = vector3(*args, **kwargs)
        return Object(self, transformations=self.transformations+[(type, v)])

    def translate(self, *args, **kwargs):
        return self.transform('translate', *args, **kwargs)

    def rotate(self, *args, **kwargs):
        return self.transform('rotate', *args, **kwargs)

    def scale(self, *args, **kwargs):
        return self.transform('scale', *args, **kwargs)

    def orient(self, s=''):
        signs = parse_orient(s)
        return self.translate([ compute_orient(sign, min_, max_)
                               for sign, min_, max_ in zip(signs, *self.bbox) ])


def Cube(*args, orient='', size=None, **kwargs):
    o = (Object(geometry='cube',
                bbox=np.array(((0, 0, 0), (1, 1, 1))))
         .orient(orient))
    if size: o = smart_call(o.scale, size)
    return o


def Cylinder(*args, orient='', h=None, d=None, d1=None, d2=None, **kwargs):
    h = first(h, 1)
    d1 = first(d1, d, 1)
    d2 = first(d2, d, 1)
    max_d = max(d1, d2)
    o = (Object(geometry='cylinder',
                constructor_args=dict(h=h, d1=d1, d2=d2), # ensure uniform
                bbox=np.array(((-max_d/2, -max_d/2, 0), (max_d/2, max_d/2, h))))
         .orient(orient))
    return o
