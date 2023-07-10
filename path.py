#!/usr/bin/python

import numpy as np
from numpy import array as V
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from svgpathtools import Path
from shapely import Polygon
from trimesh.creation import sweep_polygon
from trimesh import unitize as normalize
from mapbox_earcut import triangulate_float64 as triangulate
from .dsl import Solid

# TODO: steps should depend on arc type + length to keep constant quality
# TODO: fix issue with sharp corners
def svg_render_path(path: Path, inset=0, steps=3, closed=None):
    pts = []
    lastidx = None
    if closed is None: closed = np.isclose(path[-1].length(), 0) # this would create an invalid manifold
    if closed: lastidx = -1 # remove closing Line, we'll do it ourself
    for x in path[:lastidx]:
        pts.extend(( x.point(t) + x.normal(t)*inset
                    # skip last point because it's part of the previous element
                    for t in list(np.linspace(0, 1, steps)[:-1]) ))
    if not closed:
        pts.append( x.point(1) + x.normal(1)*inset )
    return V([ (pt.real, pt.imag) for pt in pts ])

def svg_thicken_path(profile: Path, width=1, steps=3):
    return np.concatenate((
                    svg_render_path(profile, steps=steps, inset=+width/2),
            np.flip(svg_render_path(profile, steps=steps, inset=-width/2), axis=0)))

def polygon2d_to_3d(pts, z=0):
    xs,zs = pts[:,0], pts[:,1]
    ys = z * np.ones((pts.shape[0],))
    return np.stack((xs, ys, zs), axis=1) # shape (N,2) -> (N,3)


def sweep_faces(polygon, pts, cyclic=False):
    segms, vertices, dims = pts.shape
    assert dims == 3

    # faces for a single 3D hollow segment (we're working with face indexes here)
    idxs_vertices = np.arange(0, vertices)
    segm_idxs = V([ ((vertices + idx1, vertices + idx2, idx1), # first triangle
                     (vertices + idx2, idx2, idx1)) # second
                   for idx1, idx2 in zip(idxs_vertices, np.roll(idxs_vertices, -1)) ]
                  ).reshape(-1, 3) # triangles
    if cyclic: last_segm = segms
    else:      last_segm = segms - 1
    idxs = [ i*vertices + segm_idxs for i in range(last_segm) ] # all segments
    if cyclic:
        # need to fix face indices for last (connecting start-end): wraps N+1 -> 0 in array
        idxs = np.concatenate((*idxs[:-1],
                               V(idxs[-1]) % (segms*vertices)))
    else:
        # starting and ending faces
        cap_idxs = triangulate(polygon, V([len(polygon)])).reshape(-1, 3)
        # putting everything together
        idxs = np.concatenate((*idxs, # all segments
                               cap_idxs, # start cap
                               np.flip((segms-1)*vertices + cap_idxs, axis=1)), # end cap
                              )
    return idxs

# cyclic means the start and end are connected, so it forms a loop
# if not, we put caps on both the start and the end to close the shape
def sweep_path(polygon, along_pts, tangents=None, sweeped_pts=None, cyclic=False):
    if sweeped_pts is None:
        sweeped_pts = polygon2d_to_3d(polygon) # provide a default to place polygon in 3D space
    if tangents is None:
        # let's compute the sweeped shape position + angle at each points on along_pts
        vs = normalize(along_pts[1:] - along_pts[:-1]) # directions
        vs_mid = normalize(vs[1:] + vs[:-1]) # avg of tangents between 2 edges
    else:
        assert tangents.shape == along_pts.shape
        vs_mid = tangents
    vs_midx, vs_midy, vs_midz = vs_mid.T
    yaws = -np.sign(vs_midx) * np.arccos(vs_midy / norm(vs_mid[:,(0,1)], axis=1)) # get signed angles
    pitches = np.arctan2(vs_midz, np.sqrt(vs_midx**2 + vs_midy**2))
    if tangents is None:
        # predict first and last tangents, this could be wrong or inaccurate
        yaws = np.concatenate(([yaws[0] + (yaws[0]-yaws[1])],
                               yaws,
                               [yaws[-1] + (yaws[-1]-yaws[-2])]), axis=0)
        pitches = np.concatenate(([pitches[0] + (pitches[0]-pitches[1])],
                                  pitches,
                                  [pitches[-1] + (pitches[-2]-pitches[-1])]), axis=0)
    # TODO: fix pitch math
    ss = V([ pt + Rotation.from_euler('z', yaw).apply(sweeped_pts)
            for pt, yaw, pitch in zip(along_pts, yaws, pitches) ])

    idxs = sweep_faces(polygon, ss, cyclic=cyclic)

    return Solid.from_vertices(ss.reshape(-1,3), idxs)


def ellipse_2d(final_angle, dx=1, dy=1, start_angle=0, splits=20):
    xs = np.linspace(start_angle, final_angle, splits)
    return np.stack((dx/2 * np.cos(xs), dy/2 * np.sin(xs)), axis=1)

def arc_2d(final_angle, d=1, start_angle=0, splits=20):
    return ellipse_2d(final_angle, d, d, start_angle=start_angle, splits=splits)
