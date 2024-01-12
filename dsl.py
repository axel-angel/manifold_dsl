#!/usr/bin/python

from . import operators
from .utils import *

import pymanifold, numpy as np, trimesh
from trimesh import Trimesh
from trimesh.exchange.export import export_mesh as trimesh_export
from trimesh.exchange.load import load as trimesh_load
from trimesh import unitize as normalize
from functools import reduce, cached_property
from operator import add, sub, and_, mul as multiply
from collections.abc import Iterable
from numpy.linalg import norm
from math import pi, cos
import networkx

def parse_orient(x, dimensions=3):
    if type(x) == str:
        return parse_orient_str(x, dimensions)
    if isinstance(x, Iterable): # assume it's 3 coordinates
        return x
    else:
        raise Exception("Cannot handle orientation of type {type(x)}")

# parse strings to specify orientation, eg: +xy, +x-z, by default =xyz means centered
# = means centered, +/- towards positive/negative respectively
def parse_orient_str(s, dimensions):
    signs = dict()
    assert 1 <= dimensions <= 3
    # default is centered at 0
    if dimensions >= 1: signs['x'] = 0
    if dimensions >= 2: signs['y'] = 0
    if dimensions >= 3: signs['z'] = 0
    sign = +1 # default sign is positive
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

def facing_rotation(v): # assuming v is normalized and original facing is upward z=1
    vx, vy, vz = v
    pitch = -np.arccos(vz)
    vxy_norm = norm((vx, vy))
    if np.isclose(vxy_norm, 0):
        yaw = 0
    else:
        yaw = -sign2(vx) * np.arccos(vy / vxy_norm) # get signed angles

    from scipy.spatial.transform import Rotation
    rot = np.concatenate((Rotation.from_euler('xz', (pitch, yaw)).as_matrix(),
                          ([0], [0], [0])), axis=1) # 4th column is translation component

    return rot


class Solid():
    fn = 0 # TODO: is this a good idea?

    def __init__(s, other=None):
        s.manifold = other or pymanifold.Manifold()

    # expose Manifold methods
    def from_mesh(m): return Solid(pymanifold.Manifold.from_mesh(m))
    def from_trimesh(m): return Solid.from_vertices(m.vertices, m.faces)
    def from_stl(path): return Solid.from_trimesh(trimesh_load(path))
    def to_mesh(s): return s.manifold.to_mesh() # TODO: maybe cache if possible?
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

    def vertex(s, orient=''):
        signs = parse_orient(orient)
        v = vector3(signs) # it points to the direction we want
        vs = s.to_mesh().vert_pos
        d = ((vs - s.center_point) * v).sum(axis=1)
        idx = np.argmax(d)
        return vs[idx]

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

    # suppose object is facing upward z, will rotate toward given direction
    def rotate_facing(s, *args, **kwargs):
        v = normalize(vector3(*args, **kwargs))
        m = facing_rotation(np.array(v))
        return Solid(s.manifold.transform(m))

    def scale(s, *args, **kwargs):
        return Solid(s.manifold.scale(vector3(*args, **kwargs, default=1, combiner=multiply)))

    def mirror(s, *args, **kwargs):
        # TODO: manifold mirror acts weird with >1 axes?
        return Solid(s.manifold.mirror(vector3(*args, **kwargs)))

    # TODO: we need a way to keep some axes intact, untouched!
    def orient(s, orient='', at=(0,0,0)):
        signs = parse_orient(orient)
        at = smart_call(vector3, at)
        return s.translate([ v - compute_orient(-1*sign, min_, max_)
                            for v, sign, min_, max_ in zip(at, signs, *s.bounding_box) ])
    # TODO: can we refacor with orient?
    # center object s only for the given axes in orient and at given position
    def center(s, orient='xyz', at=(0,0,0)):
        signs = parse_orient(orient)
        at = smart_call(vector3, at)
        return s.translate([ v - (max_ + min_)/2 if sign != 0
                            else 0
                            for v, sign, min_, max_ in zip(at, signs, *s.bounding_box) ])

    # this returns the min and max points in 3D
    bounding_box = cached_property(lambda s: np.array(s.manifold.bounding_box).reshape((2,3)))
    center_point = cached_property(lambda s: s.bounding_box.mean(axis=0))
    # return the point on the bounding box that satisfying the given orientation
    def extent(s, orient=''): # default to the center, yes an exception: it's inside actually
        signs = parse_orient(orient)
        return np.array([ compute_orient(sign, min_, max_)
                         for sign, min_, max_ in zip(signs, *s.bounding_box) ])

    # return s placed after other in direction described by orient
    # TODO: can we replace stick_to by next_to, including for stack()?
    def stick_to(s, other, orient=''):
        e = other.extent(orient)
        return s.orient(orient, at=e)
    # same as stick_to but doesn't orient/touch axes not present in orient
    def next_to(s, other, orient=''):
        e = other.extent(orient)
        signs = parse_orient(orient)
        d = [ v - compute_orient(-1*sign, min_, max_) if sign != 0
             else 0
             for v, sign, min_, max_ in zip(e, signs, *s.bounding_box) ]
        return s.translate(d)
    # align object s to overlap other only on the given axes
    # TODO: can we refactor this with next_to?
    def overlap_with(s, other, orient=''):
        orient = parse_orient(orient)
        e = other.extent([ -1*v for v in orient ]) # we need opposite side
        signs = parse_orient(orient)
        d = [ v - compute_orient(-1*sign, min_, max_) if sign != 0
             else 0
             for v, sign, min_, max_ in zip(e, signs, *s.bounding_box) ]
        return s.translate(d)


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
    def fit_bounding_box(s, other): # TODO: is this useful?
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

def Cube(size=1, orient=''):
    return (Solid
            .cube(smart_call(vector3, size))
            .orient(orient))

def Sphere(d=1, fn=None, orient=''):
    return (Solid
            .sphere(d/2, circular_segments=fn or Solid.fn)
            .orient(orient))

# generalized Sphere scaled in xyz (a round blob)
def Spheroid(size=1, fn=None, orient=''):
    return (Solid
            .sphere(0.5, circular_segments=fn or Solid.fn)
            .scale(smart_call(vector3, size))
            .orient(orient))

# Cylinder with upper diameter d1, lower diameter d2
def Cylinder(h=1, d=None, d1=None, d2=None, fn=Solid.fn or Solid.fn, orient=''):
    d1 = first(d1, d, 1)
    d2 = first(d2, d, 1)
    return (Solid
            .cylinder(h, d1/2, d2/2, circular_segments=fn or Solid.fn)
            .orient(orient))

# generalized Cylinder scaled in xyz (ellyptic cylinder)
def Cylindric(size=1, fn=Solid.fn or Solid.fn, orient=''):
    if isinstance(size, Iterable) and len(size) == 2:
        scaler = (size[0], size[0], size[1]) # shorthand: (d, d, h)
    else:
        scaler = smart_call(vector3, size)
    return (Solid
            .cylinder(1, 1, 1, circular_segments=fn or Solid.fn)
            .scale(scaler)
            .orient(orient))

# TODO: is this useful? validate formula works
def Prism(faces=3, h=1, d_flat=1, orient=''): # useful to make hex nuts/sockets (faces=6)
    d = d_flat / cos(pi / faces)
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

# TODO: maybe move utils functions into own namespace, eg: spacing
def split(end=1, count=2, start=0, margin=0):
    return np.linspace(start+margin, end-margin, count)
def intervals(interval=1, count=2, skip_start=False, skip_end=False):
    return np.linspace(interval if skip_start else 0, interval*(count-1), count,
                       endpoint=not skip_end)
def spanning(interval=1, extent=1, skip_start=False, skip_end=False):
    return np.arange(interval if skip_start else 0,
                     extent if skip_end else extent+interval,
                     interval)

def grid(x=(0,), y=(0,), z=(0,)):
    def go(s):
        return union(( s.translate((x_, y_, z_))
                      for x_ in x
                      for y_ in y
                      for z_ in z ))
    return go


# export to file
def to_stl(fout, s): return trimesh_export(s.to_trimesh(), fout, 'stl')
def to_3mf(fout, s): return trimesh_export(s.to_trimesh(), fout, '3mf')
