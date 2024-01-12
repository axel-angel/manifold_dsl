#!/usr/bin/python

from .utils import *
from functools import partial
from trimesh import unitize as normalize
from numpy.linalg import norm

# TODO: generalize to 3D
class Tracer():
    def __init__(s, origin=(0,0)):
        s.points = []
        x, y = smart_call(vector2, origin)
        s.points.append((x,y))

    # relative move
    def rel(s, *args, **kwargs):
        x0, y0 = s.points[-1]
        x, y = vector2(*args, **kwargs)
        s.points.append((x0 + x, y0 + y))
        return s

    # absolute position (only specified components)
    def set(s, *args, **kwargs):
        x0, y0 = s.points[-1]
        x, y = vector2(*args, **kwargs, default=None, combiner=lambda a,b: first(a, b))
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

    def round(s, rounded, splits=10):
        s.points = round_path(s.points, rounded, splits=splits)
        return s

    # TODO: def to_surface

# replace each vertex by a rounded arc
# TODO: add support for cyclic shape
def smooth_path(points, rounded, splits=10):
    ps = points
    ps = [ps[0],
          *[ (x, y)
            for p1, p2, p3 in zip(ps, ps[1:], ps[2:])
            for x, y in arc_between(p1, p2, p3, rounded, splits) ],
          ps[-1] ]
    return ps

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

