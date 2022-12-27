import gustaf as gus

s_nurbs = gus.Bezier(
    degrees=[2, 1],
    control_points=[
        [0, 0],
        [0.5, -0.5],
        [1, 0],
        [0, 1],
        [0.5, 0.5],
        [1, 1],
    ],
).nurbs
s_nurbs.insert_knots(0, [0.5])
# s_nurbs.show()
line = s_nurbs.extract.spline({1: 0.7, 0: [0.4, 0.8]})
gus.show.show_vedo(s_nurbs, [line, s_nurbs], line)
