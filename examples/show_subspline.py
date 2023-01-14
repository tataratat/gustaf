import gustaf as gus

if __name__ == "__main__":
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
    line = s_nurbs.extract.spline({0: [0.4, 0.8], 1: 0.7})
    gus.show(
        ["Source spline", s_nurbs],
        ["extract.spline({0: [0.4, 0.8], 1: 0.7})", line, s_nurbs],
        ["Extracted spline", line],
    )
