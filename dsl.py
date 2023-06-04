#!/usr/bin/python

from types import SimpleNamespace as N


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
            x=0, y=0, z=0,
            xy=0, xz=0, yz=0):
    if type(v) in {int, float}:
        v = (v, v, v) # uniform
    # otherwise assume v is vector
    return (v[0] + (x or xy or xz),
            v[1] + (y or xy or yz),
            v[2] + (z or xz or yz))


# parse strings to specify orientation, eg: +xy, +x-z, by default =xyz means centered
 # = means centered, +/- towards positive/negative respectively
orient_signs = { '=': -0.5, '+': 0, '-': -1} # unit offset to orient a certain way
def orient_to_offset(s):
    sign = +1
    orient = dict(x=-0.5, y=-0.5, z=-0.5) # default is centered
    for x in s:
        if (s := orient_signs.get(x, None)) != None: # is it a sign symbol?
            sign = s
        elif x in 'xyz': # is it a coordinate?
            orient[x] = sign
        else:
            raise Exception(f'Invalid symbol {x}')
    return orient


class Object(N):
    def __init__(self, template=None, **overrides):
        super().__init__(
                **{# defaults
                   'transformations': [],
                   # template and overrides
                   **(vars(template) if template else {}),
                   **overrides})

    def transform(self, type, *args, **kwargs):
        v = vector3(*args, **kwargs)
        return Object(self, transformations=self.transformations+[(type, v)])

    def translate(self, *args, **kwargs):
        return self.transform('translate', *args, **kwargs)

    def rotate(self, *args, **kwargs):
        return self.transform('rotate', *args, **kwargs)

    def scale(self, *args, **kwargs):
        return self.transform('scale', *args, **kwargs)

    def orient(self, s=''):
        # TODO: we suppose object has unitary dimensions xyz
        offset = orient_to_offset(s)
        return self.translate(**{ k: offset[k] for k in 'xyz' })


def Cube(*args, orient='', scale=None, **kwargs):
    o = Object(geometry='cube').orient(orient)
    if scale:
        o = smart_call(o.scale, scale)
    return o
