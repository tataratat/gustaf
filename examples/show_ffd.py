import gustaf as gus



if __name__ == "__main__":
    gus.utils.log.configure(debug=True)

    # Lets start with a 2D example
    try:
        mesh_2d = gus.io.mixd.load(
            mxyz="geometry_files/mxyz.space",
            mien="geometry_files/mien")
    except FileNotFoundError as err:
        print("Geometry files not accessible creating stand in mesh.")
        temp_spline = gus.BSpline(
            [2,2], 
            [
                [0,0,0,0.33,0.66,1,1,1],
                [0,0,0,1,1,1]
            ],
            [
                [0,0],[0.25,0],[0.5,0.05],[0.75,0.1],[1,0.1],
                [0,0.2],[0.25,0.2],[0.5,0.2],[0.75,0.2],[1,0.2],
                [0,0.4],[0.25,0.4],[0.5,0.35],[0.75,0.3],[1,0.3]
            ])
        d2_resolution = [50, 25]
        sampled_2d = temp_spline.sample(d2_resolution)
        connec_2d = gus.utils.connec.make_quad_faces(d2_resolution)
        mesh_2d = gus.Faces(sampled_2d, connec_2d)


    spline_2d = gus.BSpline(
        [2,2],
        [[0,0,0,4,4,4],[2,2,2,3,3,3]],
        [[0,0],[0.5,-0.2],[1,0],[0,0.2],
        [0.5,0.2],[1,0.2],[0,0.4],[0.5,0.4],[1,0.4]])
    
    ffd_2d = gus.ffd.FFD(mesh_2d, spline_2d)
    ffd_2d.show()



    # Now 3D
    v_res = [20, 30, 40]
    vertices = gus.create.vertices.raster(
        bounds=[[-1, 0, 5], [4, 1, 10]],
        resolutions=v_res)
    connections = gus.utils.connec.make_hexa_volumes(v_res)
    volume_3d = gus.Volumes(vertices.vertices, connections)

    # create controlpoint grid
    control_points = list()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                control_points.append([i*0.5,j*0.5,k*0.5])

    # change a control point so that there is a deformation
    # control_points[16] = [0.5, .75, .5]
    control_points[22] = [1.5, .5, .5]
    control_points[2] = [0.25, .25, .75]

    spline_3d = gus.BSpline(
            [2,2,2],
            [[0,0,0,4,4,4],[0,0,0,1,1,1],[5,5,5,10,10,10]],
             control_points)

    ffd_3d = gus.ffd.FFD(volume_3d, spline_3d)

    ffd_3d.show()


    # RationalBezier in 2D

    spline_2d_bez = gus.RationalBezier(
        [2,2],
        [[0,0],[0.5,-0.2],[1,0],[0,0.2],
        [0.5,0.2],[1,0.2],[0,0.4],[0.5,0.4],[1,0.4]],
        [1,1,1,1,1,1,1,1,1])
        

    spline_2d_bez = gus.ffd.FFD(mesh_2d, spline_2d)
    spline_2d_bez.show()