import gustaf as gus

if __name__ == "__main__":
    # starts interactive session with IgaNet's BSpline
    # currently created BSplineSurface
    p = gus.interactive.iganet_bspline.IganetBSpline(
        "ws://localhost:9001",
        degree=3,
        ncoeffs=[5, 5],
        control_point_ids=False,
    )
    p.start()
