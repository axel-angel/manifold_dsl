#!/usr/bin/python

import pymanifold, numpy as np, trimesh
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh
from types import SimpleNamespace as N
from functools import reduce
import operator as ops
from collections.abc import Iterable

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


class Solid():
    def __init__(s, other=None):
        s.manifold = other or pymanifold.Manifold()

    # expose wrapped methods
    def to_mesh(s): return s.manifold.to_mesh()
    def as_original(s): return Solid(s.manifold.as_original())
    def __add__(s, t): return Solid(s.manifold + t.manifold)
    def __sub__(s, t): return Solid(s.manifold - t.manifold)
    def __xor__(s, t): return Solid(s.manifold ^ t.manifold)

    bounding_box = property(lambda s: s.manifold.bounding_box)
    edges = property(lambda s: s.manifold.num_edge())
    triangles = property(lambda s: s.manifold.num_tri())
    vertices = property(lambda s: s.manifold.num_vert())
    genus = property(lambda s: s.manifold.genus())
    area = property(lambda s: s.manifold.get_surface_area())
    volume = property(lambda s: s.manifold.get_volume())
    is_empty = property(lambda s: s.manifold.is_empty())
    rounding_error = property(lambda s: s.manifold.precision())

    # TODO: implement
    decompose = NotImplemented
    compose = NotImplemented
    split = NotImplemented
    split_by_plane = NotImplemented
    trim_by_plane = NotImplemented
    transform = NotImplemented
    warp = NotImplemented
    mirror = NotImplemented
    refine = NotImplemented


    # methods with extended features
    def translate(s, *args, **kwargs):
        return Solid(s.manifold.translate(vector3(*args, **kwargs)))

    def rotate(s, *args, **kwargs):
        return Solid(s.manifold.rotate(vector3(*args, **kwargs)))

    def scale(s, *args, **kwargs):
        return Solid(s.manifold.scale(vector3(*args, **kwargs)))

    def orient(s, orient=''):
        signs = parse_orient(orient)
        bbox = np.array(s.manifold.bounding_box).reshape((2,3))
        return s.translate([ compute_orient(sign, min_, max_)
                            for sign, min_, max_ in zip(signs, *bbox) ])

# transmutate proxy for methods returning Solid (wrap returned type)
# -- class methods
for n in 'cube tetrahedron cylinder sphere smooth from_mesh'.split():
    (lambda n:
     setattr(Solid, n,
             lambda *args, **kwargs: Solid(getattr(pymanifold.Manifold, n)(*args, **kwargs))))(n)


# TODO: implement 'Surface' or 'Plane' corresponding to pymanifold.CrossSection


def Cube(size=1, orient=''):
    return (Solid
            .cube(smart_call(vector3, size))
            .orient(orient))

def Sphere(*args, orient='', d=1, fn=0):
    return (Solid
            .sphere(d/2, circular_segments=fn)
            .orient(orient))


def Cylinder(*args, orient='', h=None, d=None, d1=None, d2=None, fn=0):
    h = first(h, 1)
    d1 = first(d1, d, 1)
    d2 = first(d2, d, 1)
    return (Solid
            .cylinder(h, d1/2, d2/2, circular_segments=fn)
            .orient(orient))


# allows operator to accepts varargs or single parameter (list or object)
def operator_varargs(f):
    def f2(*args):
        if len(args) == 1:
            if isinstance(args[0], Iterable):
                return f(args[0])
            else:
                return args[0] # operator doesn't do anything with single argument, simplify
        else:
            return f(args) # unpack varargs to 1 list arg
    return f2


union = operator_varargs(lambda objects: reduce(ops.add, objects))
difference = operator_varargs(lambda objects: reduce(ops.sub, objects))
intersection = operator_varargs(lambda objects: reduce(ops.xor, objects))


def to_stl(fout, manifold):
    mesh = manifold.to_mesh()
    return export_mesh(Trimesh(vertices=mesh.vert_pos, faces=mesh.tri_verts, process=False),
                       fout, 'stl')
