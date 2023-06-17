#!/usr/bin/python

import pymanifold, numpy as np, trimesh
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh
from types import SimpleNamespace as N
from functools import reduce, cached_property
from operator import add, sub, and_, mul as multiply
from collections.abc import Iterable
from numpy.linalg import norm
import operators

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
    # otherwise assume v is vector
    return (combiner(v[0], first(x, xy, xz, xyz, default)),
            combiner(v[1], first(y, xy, yz, xyz, default)),
            combiner(v[2], first(z, xz, yz, xyz, default)))


def parse_orient(x):
    if type(x) == str:
        return parse_orient_str(x)
    if isinstance(x, Iterable): # assume it's 3 coordinates
        return x
    else:
        raise Exception("Cannot handle orientation of type {type(x)}")

# parse strings to specify orientation, eg: +xy, +x-z, by default =xyz means centered
# = means centered, +/- towards positive/negative respectively
def parse_orient_str(s):
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
    if   sign  < 0: return min(min_, max_)
    elif sign == 0: return (max_ + min_)/2
    else          : return max(min_, max_)


class Solid():
    fn = 0 # TODO: is this a good idea?

    def __init__(s, other=None):
        s.manifold = other or pymanifold.Manifold()

    # expose Manifold methods
    def from_mesh(m): return Solid(pymanifold.Manifold.from_mesh(m))
    def to_mesh(s): return s.manifold.to_mesh()
    def to_trimesh(s, process=False):
        m = s.to_mesh()
        return Trimesh(vertices=m.vert_pos, faces=m.tri_verts, process=process)
    def as_original(s): return Solid(s.manifold.as_original())
    def __add__(s, t): return Solid(s.manifold + t.manifold)
    def __sub__(s, t): return Solid(s.manifold - t.manifold)
    def __and__(s, t): return Solid(s.manifold ^ t.manifold) # now & (and), originally ^ (xor)

    # some handy tools
    def from_vertices(vertices, faces): # vertices=positions/points and faces=points indices
        m = pymanifold.Mesh(vertices.astype(np.float32),
                            faces.astype(np.int32),
                            # FIXME: Manifold forces Python binding to send arrays here
                            np.empty(shape=(0,0)), np.empty(shape=(0,0)))
        return Solid.from_mesh(m)

    # TODO: implement getters for orient-specific edge/triangle/vertex?
    edge_count = cached_property(lambda s: s.manifold.num_edge())
    triangle_count = cached_property(lambda s: s.manifold.num_tri())
    vertex_count = cached_property(lambda s: s.manifold.num_vert())

    # manifold geometric properties
    genus = cached_property(lambda s: s.manifold.genus())
    area = cached_property(lambda s: s.manifold.get_surface_area())
    volume = cached_property(lambda s: s.manifold.get_volume())
    is_empty = cached_property(lambda s: s.manifold.is_empty())
    rounding_error = cached_property(lambda s: s.manifold.precision())

    # TODO: implement
    decompose = NotImplemented
    compose = NotImplemented
    split = NotImplemented
    split_by_plane = NotImplemented
    trim_by_plane = NotImplemented
    transform = NotImplemented
    warp = NotImplemented
    refine = NotImplemented

    # methods with extended features
    def translate(s, *args, **kwargs):
        return Solid(s.manifold.translate(vector3(*args, **kwargs)))

    def rotate(s, *args, **kwargs):
        return Solid(s.manifold.rotate(vector3(*args, **kwargs)))

    def scale(s, *args, **kwargs):
        return Solid(s.manifold.scale(vector3(*args, **kwargs, default=1, combiner=multiply)))

    def mirror(s, *args, **kwargs):
        return Solid(s.manifold.mirror(vector3(*args, **kwargs)))

    def orient(s, orient='', at=(0,0,0)):
        signs = parse_orient(orient)
        at = smart_call(vector3, at)
        return s.translate([ v - compute_orient(-1*sign, min_, max_)
                            for v, sign, min_, max_ in zip(at, signs, *s.bounding_box) ])
    def center(s, at=(0,0,0)):
        return s.orient(at=at)

    # this returns the min and max points in 3D
    bounding_box = cached_property(lambda s: np.array(s.manifold.bounding_box).reshape((2,3)))
    center_point = cached_property(lambda s: s.bounding_box.mean(axis=0))
    # return the point on the bounding box that satisfying the given orientation
    def extent(s, orient=''): # default to the center, yes an exception: it's inside actually
        signs = parse_orient(orient)
        return np.array([ compute_orient(sign, min_, max_)
                         for sign, min_, max_ in zip(signs, *s.bounding_box) ])

    def stick_to(s, other, orient=''):
        e = other.extent(orient)
        return s.orient(orient, at=e)


    # TODO: then implement getters for positions of edges, vertices, faces, faces
    # then we can do relative placement to other objects/edges/vertices: left of X, etc
    # expose Manifold properties
    def bounding_cube(s):
        bb_self = s.bounding_box
        center_self = bb_self.mean(axis=0)
        size_self = bb_self[1] - bb_self[0]
        return (Cube()
                .scale(size_self)
                .translate(center_self))

    def bounding_sphere(s, fn=None):
        c = s.center_point
        r = norm(s.to_mesh().vert_pos - c, axis=1).max()
        return Sphere(2*r, fn=fn or Solid.fn).translate(c)

    # fit bbox of s to other, not the shape! s.fit_bounding(Sphere()) != s.bounding_sphere()
    def fit_bounding_box(s, other):
        bb_self = s.bounding_box
        bb_other = other.bounding_box
        size_self = bb_self[1] - bb_self[0]
        size_other = bb_other[1] - bb_other[0]
        return (other
                .center() # recenter before scaling!
                .scale(size_self / size_other)
                .translate(s.center))


# we aren't done yet, we have a few static methods to expose:
# transmutate proxy for methods returning Solid (wrap returned type)
for n in 'cube tetrahedron cylinder sphere smooth'.split():
    (lambda n:
     setattr(Solid, n,
             lambda *args, **kwargs: Solid(getattr(pymanifold.Manifold, n)(*args, **kwargs))))(n)


# nice shorthands to create most objects with extra goodies
# TODO: implement 'Surface' or 'Plane' corresponding to pymanifold.CrossSection

def Cube(size=1, orient=''):
    return (Solid
            .cube(smart_call(vector3, size))
            .orient(orient))

def Sphere(d=1, fn=None, orient=''):
    return (Solid
            .sphere(d/2, circular_segments=fn or Solid.fn)
            .orient(orient))

def Cylinder(h=1, d=None, d1=None, d2=None, fn=Solid.fn or Solid.fn, orient=''):
    d1 = first(d1, d, 1)
    d2 = first(d2, d, 1)
    return (Solid
            .cylinder(h, d1/2, d2/2, circular_segments=fn or Solid.fn)
            .orient(orient))

# TODO: is this useful? validate formula works
def Prism(faces=3, h=1, d_flat=1, orient=''): # useful to make hex nuts/sockets (faces=6)
    d = d_flat/2 / cos(pi / faces)
    return Cylinder(h=h, d=d, fn=faces, orient=orient)

def Tetrahedron(orient=''):
    return (Solid
            .tetrahedron()
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


# operators
union = operator_varargs(lambda objects: reduce(add, objects))
difference = operator_varargs(lambda objects: reduce(sub, objects))
intersection = operator_varargs(lambda objects: reduce(and_, objects))

def hull(*args):
    m = union(*args).to_mesh()
    vertices, faces = operators.hull(m.vert_pos)
    return Solid.from_vertices(vertices, faces)

# allows to place objects next to/on top of each other (see stick_to)
def stack(orient):
    def go(objects):
        return reduce(lambda a,b: (a + b.stick_to(a, orient=orient)), objects)
    return operator_varargs(go)


# export to file
def to_stl(fout, s): return export_mesh(s.to_trimesh(), fout, 'stl')
def to_3mf(fout, s): return export_mesh(s.to_trimesh(), fout, '3mf')
