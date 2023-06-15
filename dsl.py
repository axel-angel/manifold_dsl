#!/usr/bin/python

import pymanifold, numpy as np, trimesh
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh
from types import SimpleNamespace as N
from functools import reduce
from operator import add, sub, xor, mul as multiply
from collections.abc import Iterable
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

def vector3(v=(0,0,0),
            x=None, y=None, z=None,
            xy=None, xz=None, yz=None,
            xyz=None,
            default=0, combiner=add):
    if type(v) in {int, float}:
        v = (v, v, v) # uniform
    # otherwise assume v is vector
    return (combiner(v[0], first(x, xy, xz, xyz, default)),
            combiner(v[1], first(y, xy, yz, xyz, default)),
            combiner(v[2], first(z, xz, yz, xyz, default)))


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
    fn = 0 # TODO: is this a good idea?

    def __init__(s, other=None):
        s.manifold = other or pymanifold.Manifold()

    # expose Manifold methods
    def from_mesh(m): return pymanifold.Manifold.from_mesh(m)
    def to_mesh(s): return s.manifold.to_mesh()
    def as_original(s): return Solid(s.manifold.as_original())
    def __add__(s, t): return Solid(s.manifold + t.manifold)
    def __sub__(s, t): return Solid(s.manifold - t.manifold)
    def __xor__(s, t): return Solid(s.manifold ^ t.manifold)

    # some handy tools
    def from_vertices(vertices, faces): # vertices=positions/points and faces=points indices
        m = pymanifold.Mesh(vertices.astype(np.float32),
                            faces.astype(np.int32),
                            # FIXME: Manifold forces Python binding to send arrays here
                            np.empty(shape=(0,0)), np.empty(shape=(0,0)))
        return Solid.from_mesh(m)

    # TODO: make bounding_box a Manifold.Cube instead, more useful that way
    # TODO: then implement getters for positions of edges, vertices, faces, faces
    # then we can do relative placement to other objects/edges/vertices: left of X, etc
    # expose Manifold properties
    bounding_box = property(lambda s: np.array(s.manifold.bounding_box).reshape((2,3)))
    center = property(lambda s: s.bounding_box.mean(axis=0))
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
        return s.translate([ v + compute_orient(sign, min_, max_)
                            for v, sign, min_, max_ in zip(at, signs, *s.bounding_box) ])


# we aren't done yet, we have plenty of other static methods to expose:
# transmutate proxy for methods returning Solid (wrap returned type)
# -- class methods
for n in 'cube tetrahedron cylinder sphere smooth from_mesh'.split():
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

def Cylinder(h=None, d=None, d1=None, d2=None, fn=Solid.fn or Solid.fn, orient=''):
    h = first(h, 1)
    d1 = first(d1, d, 1)
    d2 = first(d2, d, 1)
    return (Solid
            .cylinder(h, d1/2, d2/2, circular_segments=fn or Solid.fn)
            .orient(orient))

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
intersection = operator_varargs(lambda objects: reduce(xor, objects))

def hull(*args):
    m = union(*args).to_mesh()
    vertices, faces = operators.hull(m.vert_pos)
    return Solid.from_vertices(vertices, faces)

# allows to place objects next to/on top of each other (see orient)
def stack(orient):
    rsign_map = {'+':'-', '-':'+'}
    def go(objects):
        sign = orient[0]
        axes = orient[1:]
        rsign = rsign_map[sign]
        # TODO: instead of using orient repeatdly(=translation=increas errors), can orient one per object with relative positioning to last object bbox
        return reduce(lambda a,b: (a.orient(rsign+axes) + b.orient(sign+axes)), objects)
    return operator_varargs(go)


# export to file
def to_stl(fout, manifold):
    mesh = manifold.to_mesh()
    return export_mesh(Trimesh(vertices=mesh.vert_pos, faces=mesh.tri_verts, process=False),
                       fout, 'stl')
