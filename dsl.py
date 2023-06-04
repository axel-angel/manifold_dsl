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
                   'function': None,
                   'args': {},
                   'transformations': [],
                   'children': [],
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


# TODO: add $fa, $fs, $fn
# TODO: store geometric description/points list so we can compute bbox with arbitrary transformations
# can compute position of points or face, build stuff on it, relative to it, etc
def Sphere(*args, orient='', d=1, **kwargs):
    o = (Object(function='sphere',
                args=dict(d=d),
                bbox=np.array((3*(-d/2,), 3*(d/2,))))
         .orient(orient))
    return o


def Cube(*args, orient='', size=None, **kwargs):
    o = (Object(function='cube',
                bbox=np.array(((0, 0, 0), (1, 1, 1))))
         .orient(orient))
    if size: o = smart_call(o.scale, size)
    return o


def Cylinder(*args, orient='', h=None, d=None, d1=None, d2=None, **kwargs):
    h = first(h, 1)
    d1 = first(d1, d, 1)
    d2 = first(d2, d, 1)
    max_d = max(d1, d2)
    o = (Object(function='cylinder',
                args=dict(h=h, d1=d1, d2=d2),
                bbox=np.array(((-max_d/2, -max_d/2, 0), (max_d/2, max_d/2, h))))
         .orient(orient))
    return o


# TODO: add operators: union, difference, etc
# idea: use Object so we can group objects and apply transforms to all of them?
# also useful to reorient that group (computing bbox recursively)
def union(*objects): return Object(function='union', children=objects)
def difference(*objects): return Object(function='difference', children=objects)
def intersection(*objects): return Object(function='intersection', children=objects)
def hull(*objects): return Object(function='hull', children=objects)
def minkowski(*objects): return Object(function='minkowski', children=objects)


def to_scad_(object, ident=True):
    ts = object.transformations
    if len(ts) > 0:
        (t_name, args) = ts[-1]
        # openscad wrap in reverse order (last operation first)
        return to_scad_(Object(function=t_name, args=args, children=[Object(object, transformations=ts[:-1])]))

    args = object.args
    if type(args) is dict:
        args_str = ', '.join(( f'{k} = {v}' for k,v in object.args.items() ))
    else:
        args_str = '['+ ', '.join(map(str, args)) + ']'
    xs = [f'{object.function}({args_str})']
    if (cnt := len(object.children)) > 0:
        if cnt > 1: xs.append('{')
        for o in object.children:
            xs.extend(to_scad_(o))
        if cnt > 1: xs.append('}')
    else:
        xs[-1] += ';'

    # let's add the current indent of course
    ident_str = '  ' if ident else ''
    return [ f'{ident_str}{x}' for x in xs ]

def to_scad(object):
    # TODO: force $fn to a reasonable value for now
    return '\n'.join( ['$fn=20;'] + to_scad_(object, ident=False) )
