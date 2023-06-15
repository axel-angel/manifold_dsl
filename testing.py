#/usr/bin/python

from dsl import *

star = union(( Cube(size=1, orient=sign+axis )
              for axis in 'xyz'
              for sign in '+-' )).translate(x=5)
to_stl('out.stl', star)


bounder = star.bounding_sphere(fn=50)
to_stl('out_bounding.stl', bounder)


smooth_cube = hull(( (Sphere(d=0.5, fn=20)
                      .orient(-1*np.array((x,y,z))) # ensure we fit the unit cube
                      .translate((x, y, z)))
                    for x in (-1,+1)
                    for y in (-1,+1)
                    for z in (-1,+1)
                    ))
to_stl('smooth_cube.stl', smooth_cube)

#x = (Cube(orient='+x')
#     .translate(z=1),
#     .rotate(x=45)
#     .place(top_of=base.edges('-z'))
#     )
