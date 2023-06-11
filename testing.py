#/usr/bin/python

from dsl import *

star = union(( Cube(size=1, orient=sign+axis )
           for axis in 'xyz'
           for sign in '+-' ))

to_stl('out.stl',
       ((star.orient('+x') + Sphere(fn=100, orient='-x')).orient('+y')
        + Cylinder(fn=10, orient='-y')))

def stack(orient, *objects):
    sign = orient[0]
    axes = orient[1:]
    rsign = {'+':'-', '-':'+'}[sign]
    return reduce(lambda a,b: (a.orient(rsign+axes) + b.orient(sign+axes)), objects)

x = stack('+z',
        Cube(2).rotate(z=45),
        Cube(4).rotate(x=60),
        stack('+x', Cube(1), Cube(4), Sphere(fn=25)),
        Cylinder(h=10, d1=1, d2=10),
        Cylinder(h=3, d1=10, d2=5),
        Cube(3),
).orient()

to_stl('out2.stl', x)

#x = (Cube(orient='+x')
#     .translate(z=1),
#     .rotate(x=45)
#     .place(top_of=base.edges('-z'))
#     )
