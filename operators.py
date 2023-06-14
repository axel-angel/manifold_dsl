#/usr/bin/python

import pymanifold
import numpy as np
from scipy.spatial import ConvexHull
from types import SimpleNamespace as N


def hull(vertices):
    h = ConvexHull(vertices)
    # the order of face vertices in h aren't all in CCW = wrong normals, therefore we need to fix it

    # all pair of lines are within the convex hull by definition
    # therefore we can use the center (which is inside) as a reference for other points to know where is outside
    center = h.points.mean(axis=0)

    # lookup triangles points into a single array
    xs0 = h.points[ h.simplices[:,0] ]
    xs1 = h.points[ h.simplices[:,1] ]
    xs2 = h.points[ h.simplices[:,2] ]

    # compute actual normals implied by face order
    cross = np.cross(xs0 - xs1,
                     xs2 - xs1)

    # check which normals are pointing outward correctly or inward incorrectly
    outward = xs1 - center
    corrects = (cross * outward).sum(axis=1) < 0

    # final faces with all outward correct normals
    faces = np.concatenate((# those faces are already corrects
                            h.simplices[corrects],
                            # those need to be inverted
                            h.simplices[~corrects][:,(2,1,0)]))

    return h.points, faces
