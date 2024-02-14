#!/usr/bin/python

import manifold3d
from .utils import *
from .dsl import Solid
from functools import partial
from trimesh import unitize as normalize
from numpy.linalg import norm


class Surface():
    def __init__(s, other=None):
        s.crosssection = other or manifold3d.CrossSection()

    def from_points(points, *holes):
        return Surface(manifold3d.CrossSection([points, *holes]))

    def extrude(s, h):
        return Solid(s.crosssection.extrude(h))

    # TODO: add other methods


# 2D now tracer pen
# TODO: generalize to 3D
# TODO: add nurbs, b-splines?
class Tracer():
    def __init__(s, origin=None):
        if origin is None:
            s.points = []
        else:
            s.points = [(origin[0], origin[1])]

    def from_points(points):
        s = Tracer()
        s.points = points
        return s

    # relative move
    def rel(s, *args, **kwargs):
        x0, y0 = s.points[-1]
        x, y = vector2(*args, **kwargs)
        s.points.append((x0 + x, y0 + y))
        return s

    # absolute position (only specified components)
    def set(s, *args, **kwargs):
        x, y = vector2(*args, **kwargs, default=None, combiner=lambda a,b: first(a, b))
        if len(s.points) == 0:
            if x is None or y is None:
                raise Exception("No origin, hence all coordinates must be set")
            x0, y0 = x, y
        else:
            x0, y0 = s.points[-1]
            if x is not None: x0 = x
            if y is not None: y0 = y
        s.points.append((x0, y0))
        return s

    # relative direction by angle
    def dir(s, angle, length):
        x, y = s.points[-1]
        x += length * np.cos(angle * np.pi/180)
        y += length * np.sin(angle * np.pi/180)
        s.points.append((x, y))
        return s

    def smoothed(s, rounded, splits=10, cyclic=True):
        return Tracer.from_points(smooth_path(s.points, rounded, splits=splits, cyclic=cyclic))

    def to_surface(s):
        return Surface.from_points(s.points)

# replace each vertex by a rounded arc
# TODO: add support for cyclic shape
def smooth_path(points, rounded, splits=10, cyclic=True):
    ps = points
    zps = list(zip(ps, ps[1:], ps[2:]))
    if cyclic:
        zps.extend([(ps[-2], ps[-1], ps[0]), (ps[-1], ps[0], ps[1])])
    ps2 = [ (x, y)
           for p1, p2, p3 in zps
           for x, y in arc_between(p1, p2, p3, rounded, splits) ]
    if not cyclic:
        ps2.insert(0, ps[0])
        ps2.append(ps[-1])
    return ps2

# replace pt2 by a circle arc connected to pt1 and pt2, making it smoother curve
# rounded is the length of the segments replaced by an arc (on both pt1-pt2 and pt2-pt3)
def arc_between(pt1, pt2, pt3, rounded, splits=10, clip_rounded=True):
    pt1 = V(pt1)
    pt2 = V(pt2)
    pt3 = V(pt3)
    if clip_rounded: # cut short arc to avoid it extending beyond the current (half) segment
        rounded = min(rounded, norm( pt1 - pt2 )/2, norm( pt3 - pt2 )/2)
    v1 = normalize( pt1 - pt2 )
    v2 = normalize( pt3 - pt2 )
    alpha = np.arccos(np.dot(v1, v2)) # angle between the two segments
    d = rounded / np.cos(alpha/2)
    r = rounded * np.tan(alpha/2)
    center = pt2 + d * normalize( v1 + v2 )
    beta = np.pi - alpha # angle to sweep the arc
    v3 = normalize( pt2 - center )
    angle = np.arctan2(v3[1], v3[0]) # get signed angles
    n = (v1[1], -v1[0]) # normal to v1
    spin = np.sign( np.dot(n, v2) ) # determine arc spins positive or negative
    return (center
            + Arc(start_angle=angle - spin * beta/2,
                  final_angle=angle + spin * beta/2,
                  d=2*r,
                  splits=splits,
                  ))


# TODO: implement 'Surface' or 'Plane' corresponding to pymanifold.CrossSection


def Ellipse(final_angle, dx=1, dy=1, start_angle=0, splits=20):
    xs = np.linspace(start_angle, final_angle, splits)
    return np.stack((dx/2 * np.cos(xs), dy/2 * np.sin(xs)), axis=1)

def Arc(final_angle, d=1, start_angle=0, splits=20):
    return Ellipse(final_angle, d, d, start_angle=start_angle, splits=splits)

