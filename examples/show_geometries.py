import gustaf as gus
import numpy as np

if __name__ == '__main__':

    line = gus.spline.create.line(np.array([[0, 0, 0], [2, 5, 0], [4, 4, 2]]))
    rect = gus.spline.create.rectangle(5, 3)
    box = gus.spline.create.box(3, 2, 4)
<<<<<<< HEAD
    pyramid = gus.spline.create.pyramid(1, 1, 2)

    # Show objects
    gus.show.show_vedo(
        ['Line', line],
        ['Rectangle', rect],
        ['Box', box],
        ['Pyramid', pyramid],
        title='Rectangular objects', resolution=50)

    circ = gus.spline.create.circle(5)
    disk1 = gus.spline.create.disk(3, angle=100, n_knot_spans=4)
    disk2 = gus.spline.create.disk(5, r0=1., angle=360, n_knot_spans=10)

    gus.show.show_vedo(
        ['Line Circle', circ],
        ['Disk section', disk1],
        ['Disk', disk2])

    cone = gus.spline.create.cone(5, 10, angle=180)
    gus.show.show_vedo(['Cone', cone])

    torus = gus.spline.create.torus(5, 2, r0=0.5)
    torus2 = gus.spline.create.torus(5, 2, r0=0.5, angle=[100, 210])
    gus.show.show_vedo(
        ['Torus', torus],
        ['Torus section', torus2],
        resolution=50
    )

    sphere = gus.spline.create.sphere(3)
    gus.show.show_vedo(['Sphere', sphere], resolution=50)
=======
    pyramid = gus.spline.create.pyramid(2, 2, 6)

    # gus.show.show_vedo(line, rect, box, pyramid)

    circ = gus.spline.create.circle(5)
    disk1 = gus.spline.create.disk(3, angle=100, n_knot_spans=2)
    disk2 = gus.spline.create.disk(5, angle=360, n_knot_spans=10)

    # gus.show.show_vedo(circ, disk1, disk2)

    cone = gus.spline.create.cone(5, 10, angle=180)
    cone.show()
    sphere = gus.spline.create.sphere(3)
    # sphere.show()

    torus = gus.spline.create.torus(5, 2, r0=0.5)
    torus.show()

    torus2 = gus.spline.create.torus(5, 2, r0=0.5, angle=[100, 210])
    torus2.show()
>>>>>>> add example for shape creation routines
