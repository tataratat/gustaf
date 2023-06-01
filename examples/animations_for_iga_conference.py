import imageio
import numpy as np
import vedo

import gustaf as gus

# Settings
vedo.settings.screenshot_transparent_background = 1
export_resolution = [1440, 1080]
fps = 15
animation1 = False
animation2 = False
animation3 = True
animation4 = False
animation5 = False
ctps_on = True

## ANIMATION 1 ##
# Animation to show macro-modifications
if animation1:
    print("Starting to export animation 1")
    np.random.seed = 830942978
    sample_resolution = 3
    write_ms = imageio.get_writer(
        "animation1_ms.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )
    write_df = imageio.get_writer(
        "animation1_df.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )
    deformation_function = gus.Bezier(
        degrees=[2, 1],
        control_points=[
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
    ).bspline

    generator = gus.spline.microstructure.Microstructure()
    generator.microtile = gus.spline.microstructure.tiles.DoubleLatticeTile()
    generator.tiling = [1, 1]

    duration = 1
    n_frames = fps * duration
    for i in range(n_frames):
        dx = 0.3 * i / n_frames
        generator.deformation_function = deformation_function.copy()
        generator.deformation_function.insert_knots(0, [0.5 + dx])
        generator.deformation_function.insert_knots(1, [0.5])
        list = generator.create()
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            generator.deformation_function,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))

    print("Section complete : 1")
    for i in range(n_frames):
        dx = 0.3 * (n_frames - i) / n_frames
        dy = -0.2 * i / n_frames
        generator.deformation_function = deformation_function.copy()
        generator.deformation_function.insert_knots(0, [0.5 + dx])
        generator.deformation_function.insert_knots(1, [0.5 + dy])
        list = generator.create()
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            generator.deformation_function,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))

    print("Section complete : 2")
    for i in range(n_frames):
        dy = -0.2 * (n_frames - i) / n_frames
        generator.deformation_function = deformation_function.copy()
        generator.deformation_function.insert_knots(0, [0.5])
        generator.deformation_function.insert_knots(1, [0.5 + dy])
        list = generator.create()
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            generator.deformation_function,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=3,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))

    print("Section complete : 3")
    write_ms.close()
    write_df.close()
    plt = gus.show(
        generator.microtile.create_tile(),
        knots=True,
        control_points=False,
        c="lightgray",
        lighting="off",
        offscreen=True,
        resolution=3,
        size=export_resolution,
    )
    plt.screenshot(filename="microtile.png")

if animation2:
    print("Starting to export animation 2")
    np.random.seed = 830942978
    sample_resolution = 5
    write_ms = imageio.get_writer(
        "animation2_ms.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )
    write_df = imageio.get_writer(
        "animation2_df.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )
    deformation_function = gus.Bezier(
        degrees=[2, 1],
        control_points=[
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
    ).bspline
    deformation_function.insert_knots(0, [0.5])
    deformation_function.insert_knots(1, [0.5])

    generator = gus.spline.microstructure.Microstructure()
    generator.microtile = gus.spline.microstructure.tiles.DoubleLatticeTile()
    generator.tiling = [1, 1]
    generator.deformation_function = deformation_function

    duration = 2
    n_frames = fps * duration
    offset = np.random.rand(*deformation_function.cps.shape) * 0.25
    offset2 = -np.random.rand(*deformation_function.cps.shape) * 0.2
    offset *= 1 / n_frames
    offset2 *= 1 / n_frames
    for i in range(n_frames):
        generator.deformation_function.cps += offset
        list = generator.create()
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            generator.deformation_function,
            knots=True,
            control_points=ctps_on,
            control_mesh=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))
    print("Section complete : 1")
    for i in range(n_frames):
        generator.deformation_function.cps += offset2
        list = generator.create()
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            generator.deformation_function,
            knots=True,
            control_points=ctps_on,
            control_mesh=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))

    print("Section complete : 2")
    for i in range(n_frames):
        generator.deformation_function.cps -= offset + offset2
        list = generator.create()
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            generator.deformation_function,
            knots=True,
            control_points=ctps_on,
            control_mesh=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))

    print("Section complete : 3")
    write_ms.close()
    write_df.close()
    plt = gus.show(
        generator.microtile.create_tile(),
        knots=True,
        control_points=False,
        c="lightgray",
        lighting="off",
        offscreen=True,
        resolution=sample_resolution,
        size=export_resolution,
    )
    plt.screenshot(filename="microtile.png")

if animation3:
    print("Starting to export animation 3.")
    np.random.seed = 2049410
    sample_resolution = 10
    writer_ffd = imageio.get_writer(
        "animation3_ffd.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )
    writer_composition = imageio.get_writer(
        "animation3_composition.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )
    # Create MacroSpline
    macrospline = gus.Bezier(
        degrees=[2, 2],
        control_points=[
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
    )

    line0 = gus.Bezier(degrees=[1], control_points=[[0.8, 0], [0.9, 1]])
    line1 = gus.Bezier(degrees=[1], control_points=[[0.3, 1], [1.0, 0.3]])
    rectangle = gus.Bezier(
        degrees=[1, 1],
        control_points=[[0.0, 0.2], [0.3, 0.0], [0.4, 0.9], [0.6, 0.7]],
    )

    line0_ffd = gus.Bezier(degrees=[1], control_points=[[0.8, 0], [0.9, 1]])
    line1_ffd = gus.Bezier(degrees=[1], control_points=[[0.3, 1], [1.0, 0.3]])
    rectangle_ffd = gus.Bezier(
        degrees=[1, 1],
        control_points=[[0.0, 0.2], [0.3, 0.0], [0.4, 0.9], [0.6, 0.7]],
    )
    macrospline.show_options["knots"] = True
    macrospline.show_options["alpha"] = 0.1
    macrospline.show_options["c"] = "lightgray"

    duration = 2
    n_frames = fps * duration
    offset = np.random.rand(*macrospline.cps.shape) * 0.3
    offset2 = np.random.rand(*macrospline.cps.shape) * 0.3
    offset *= 1 / n_frames
    offset2 *= 1 / n_frames

    # Dummy plot to access camera
    plt = gus.show(
        [macrospline],
        knots=True,
        control_points=False,
        lighting="off",
        offscreen=True,
        resolution=sample_resolution,
        lw=2,
        size=export_resolution,
    )
    camera_defaults = plt.camera

    for i in range(n_frames):
        macrospline.cps += offset

        # Compositions
        line0_c = macrospline.compose(line0)
        line1_c = macrospline.compose(line1)
        rectangle_c = macrospline.compose(rectangle)
        line0_c.show_options["knots"] = False
        line1_c.show_options["knots"] = False
        rectangle_c.show_options["c"] = "blue"

        # FFD
        line0_ffd.cps[:] = macrospline.evaluate(line0.cps)
        line1_ffd.cps[:] = macrospline.evaluate(line1.cps)
        rectangle_ffd.cps[:] = macrospline.evaluate(rectangle.cps)

        plt = gus.show(
            [macrospline, line0_c, line1_c, rectangle_c],
            knots=True,
            control_points=False,
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            lw=2,
            size=export_resolution,
            camera=camera_defaults,
        )
        writer_composition.append_data(plt.screenshot(asarray=True))
        macrospline.show_options["c"] = "green"
        plt = gus.show(
            [macrospline, line0_ffd, line1_ffd, rectangle_ffd],
            knots=True,
            control_points=False,
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            lw=2,
            size=export_resolution,
            camera=camera_defaults,
        )
        writer_ffd.append_data(plt.screenshot(asarray=True))

    print("Section 1 complete")
    for i in range(n_frames):
        macrospline.cps += offset2 - offset

        # Compositions
        line0_c = macrospline.compose(line0)
        line1_c = macrospline.compose(line1)
        rectangle_c = macrospline.compose(rectangle)
        line0_c.show_options["knots"] = False
        line1_c.show_options["knots"] = False
        rectangle_c.show_options["c"] = "blue"

        # FFD
        line0_ffd.cps[:] = macrospline.evaluate(line0.cps)
        line1_ffd.cps[:] = macrospline.evaluate(line1.cps)
        rectangle_ffd.cps[:] = macrospline.evaluate(rectangle.cps)

        plt = gus.show(
            [macrospline, line0_c, line1_c, rectangle_c],
            knots=True,
            control_points=False,
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            lw=2,
            size=export_resolution,
            camera=camera_defaults,
        )
        writer_composition.append_data(plt.screenshot(asarray=True))
        macrospline.show_options["c"] = "green"
        plt = gus.show(
            [macrospline, line0_ffd, line1_ffd, rectangle_ffd],
            knots=True,
            control_points=False,
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            lw=2,
            size=export_resolution,
            camera=camera_defaults,
        )
        writer_ffd.append_data(plt.screenshot(asarray=True))

    print("Section 2 complete")
    for i in range(n_frames):
        macrospline.cps -= offset2

        # Compositions
        line0_c = macrospline.compose(line0)
        line1_c = macrospline.compose(line1)
        rectangle_c = macrospline.compose(rectangle)
        line0_c.show_options["knots"] = False
        line1_c.show_options["knots"] = False
        rectangle_c.show_options["c"] = "blue"

        # FFD
        line0_ffd.cps[:] = macrospline.evaluate(line0.cps)
        line1_ffd.cps[:] = macrospline.evaluate(line1.cps)
        rectangle_ffd.cps[:] = macrospline.evaluate(rectangle.cps)

        plt = gus.show(
            [macrospline, line0_c, line1_c, rectangle_c],
            knots=True,
            control_points=False,
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            lw=2,
            size=export_resolution,
            camera=camera_defaults,
        )
        writer_composition.append_data(plt.screenshot(asarray=True))
        macrospline.show_options["c"] = "green"
        plt = gus.show(
            [macrospline, line0_ffd, line1_ffd, rectangle_ffd],
            knots=True,
            control_points=False,
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            lw=2,
            size=export_resolution,
            camera=camera_defaults,
        )
        writer_ffd.append_data(plt.screenshot(asarray=True))
    print("Section 3 complete")

    # Extract screenshots with control points
    macrospline.cps += offset * n_frames

    # Compositions
    line0_c = macrospline.compose(line0)
    line1_c = macrospline.compose(line1)
    rectangle_c = macrospline.compose(rectangle)
    line0_c.show_options["knots"] = False
    line0_c.show_options["control_points"] = True
    line0_c.show_options["control_mesh"] = True
    line0_c.show_options["control_mesh_c"] = "darkblue"
    line0_c.show_options["control_points_c"] = "black"
    line1_c.show_options["knots"] = False
    line1_c.show_options["control_points"] = True
    line1_c.show_options["control_mesh"] = True
    line1_c.show_options["control_mesh_c"] = "darkblue"
    line1_c.show_options["control_points_c"] = "black"
    rectangle_c.show_options["c"] = "blue"
    rectangle_c.show_options["control_points"] = True
    rectangle_c.show_options["control_mesh"] = True
    rectangle_c.show_options["control_mesh_c"] = "darkblue"
    rectangle_c.show_options["control_points_c"] = "black"
    plt = gus.show(
        [macrospline, line0_c, line1_c, rectangle_c],
        lighting="off",
        offscreen=True,
        resolution=sample_resolution,
        lw=2,
        size=export_resolution,
        camera=camera_defaults,
    )
    plt.screenshot(filename="animation3_composition_w_cps.png")

    # FFD
    line0_ffd.control_points[:] = macrospline.evaluate(line0.cps)
    line1_ffd.cps[:] = macrospline.evaluate(line1.cps)
    rectangle_ffd.cps[:] = macrospline.evaluate(rectangle.cps)
    line0_ffd.show_options["knots"] = False
    line0_ffd.show_options["control_points"] = True
    line0_ffd.show_options["control_mesh"] = True
    line0_ffd.show_options["control_mesh_c"] = "darkgreen"
    line0_ffd.show_options["control_points_c"] = "black"
    line1_ffd.show_options["knots"] = False
    line1_ffd.show_options["control_points"] = True
    line1_ffd.show_options["control_mesh"] = True
    line1_ffd.show_options["control_mesh_c"] = "darkgreen"
    line1_ffd.show_options["control_points_c"] = "black"
    rectangle_ffd.show_options["c"] = "green"
    rectangle_ffd.show_options["control_points"] = True
    rectangle_ffd.show_options["control_mesh"] = True
    rectangle_ffd.show_options["control_mesh_c"] = "darkgreen"
    rectangle_ffd.show_options["control_points_c"] = "black"
    plt = gus.show(
        [macrospline, line0_ffd, line1_ffd, rectangle_ffd],
        lighting="off",
        offscreen=True,
        resolution=sample_resolution,
        lw=2,
        size=export_resolution,
        camera=camera_defaults,
    )
    plt.screenshot(filename="animation3_ffd_w_cps.png")
    print("Animation Complete")

if animation4:
    print("Starting to export animation 4")
    np.random.seed = 934857904
    sample_resolution = 3
    write_ms = imageio.get_writer(
        "animation4_ms.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )
    write_df = imageio.get_writer(
        "animation4_tile.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )

    deformation_function = gus.Bezier(
        degrees=[2, 2],
        control_points=[
            [0.0, 0.0],
            [0.5, 0.2],
            [1.0, 0.0],
            [0.2, 0.5],
            [0.5, 0.5],
            [0.8, 0.5],
            [0.0, 1.0],
            [0.5, 1.2],
            [1.0, 1.0],
        ],
    )

    generator = gus.spline.microstructure.Microstructure()
    generator.microtile = gus.spline.microstructure.tiles.CrossTile2D()
    generator.tiling = [4, 4]
    generator.deformation_function = deformation_function

    thickness = 0.05

    def parametrization_function(x):
        return np.ones((x.shape[0], 1)) * thickness

    generator.parametrization_function = parametrization_function

    duration = 2
    n_frames = fps * duration
    dx = 0.3 / n_frames
    for i in range(n_frames):
        thickness += dx
        list = generator.create(center_expansion=1.2)
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            generator.microtile.create_tile(
                center_expansion=1.2,
                parameters=parametrization_function(
                    generator.microtile.evaluation_points
                ),
            ),
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))

    print("Section complete : 1")
    for i in range(n_frames):
        thickness -= dx
        list = generator.create(center_expansion=1.2)
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            generator.microtile.create_tile(
                center_expansion=1.2,
                parameters=parametrization_function(
                    generator.microtile.evaluation_points
                ),
            ),
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))

    print("Section complete : 2")
    plt = gus.show(
        generator.deformation_function,
        knots=True,
        control_points=False,
        c="lightgray",
        lighting="off",
        offscreen=True,
        resolution=sample_resolution,
        size=export_resolution,
    )
    plt.screenshot(filename="animation4_microtile.png")

if animation5:
    print("Starting to export animation 5")
    np.random.seed = 69239683
    sample_resolution = 3
    write_ms = imageio.get_writer(
        "animation5_ms.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )
    write_df = imageio.get_writer(
        "animation5_para.mp4",
        codec="mjpeg",
        mode="I",
        fps=fps,
        quality=10,
        pixelformat="yuvj444p",
    )

    deformation_function = gus.Bezier(
        degrees=[2, 1],
        control_points=[
            [0.0, 0.0],
            [0.5, 0.2],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.2],
            [1.0, 1.0],
        ],
    )

    parametrization_spline = gus.Bezier(
        degrees=[2, 1], control_points=np.ones((6, 1)) * 0.25
    )

    para_geometric_base = gus.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
    )
    para_geometric_base.spline_data["field"] = parametrization_spline
    para_geometric_base.show_options["data_name"] = "field"
    para_geometric_base.show_options["control_points"] = False
    para_geometric_base.show_options["vmin"] = 0.05
    para_geometric_base.show_options["vmax"] = 0.45
    para_geometric_base.show_options["cmap"] = "jet"
    para_geometric_base.show_options["lighting"] = "off"

    def parametrization_function(x):
        return parametrization_spline.evaluate(x)

    generator = gus.spline.microstructure.Microstructure()
    generator.microtile = gus.spline.microstructure.tiles.CrossTile2D()
    generator.tiling = [5, 5]
    generator.deformation_function = deformation_function
    generator.parametrization_function = parametrization_function

    duration = 2
    n_frames = fps * duration

    para_offset = np.random.rand(6, 1)
    para_offset -= np.min(para_offset)
    para_offset /= np.max(para_offset)
    para_offset = (para_offset - 0.5) * 0.4 / n_frames

    for i in range(n_frames):
        parametrization_spline.cps += para_offset
        list = generator.create()
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            para_geometric_base,
            knots=True,
            control_points=False,
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))

    print("Section complete : 1")
    for i in range(n_frames):
        parametrization_spline.cps -= para_offset
        list = generator.create()
        plt = gus.show(
            list,
            knots=True,
            control_points=False,
            c="lightgray",
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_ms.append_data(plt.screenshot(asarray=True))
        plt = gus.show(
            para_geometric_base,
            knots=True,
            control_points=False,
            lighting="off",
            offscreen=True,
            resolution=sample_resolution,
            size=export_resolution,
        )
        write_df.append_data(plt.screenshot(asarray=True))

    print("Section complete : 2")
    plt = gus.show(
        generator.deformation_function,
        knots=True,
        control_points=False,
        c="lightgray",
        lighting="off",
        offscreen=True,
        resolution=sample_resolution,
        size=export_resolution,
    )
    plt.screenshot(filename="deformation_function.png")
    plt = gus.show(
        generator.microtile.create_tile(),
        knots=True,
        control_points=False,
        c="lightgray",
        lighting="off",
        offscreen=True,
        resolution=sample_resolution,
        size=export_resolution,
    )
    plt.screenshot(filename="animation5_deformation_function.png")
    write_df.close()
    write_ms.close()
