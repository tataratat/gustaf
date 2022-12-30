import numpy as np

import gustaf as gus

if __name__ == "__main__":
    # turn on debug logs
    # gus.utils.log.configure(debug=True)
    # surface
    # define degrees
    ds2 = [2, 2]

    # define knot vectors
    kvs2 = [
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ]

    # define control points
    cps2 = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1.5, 0],
            [3, 1.5, 0],
            [-1, 0, 0],
            [-1, 2, 0],
            [1, 4, 0],
            [3, 4, 0],
            [-2, 0, 0],
            [-2, 2, 0],
            [1, 5, 0],
            [3, 5, 2],
        ]
    )

    # init bspline
    b = gus.BSpline(
        degrees=ds2,
        knot_vectors=kvs2,
        control_points=cps2,
    )

    # define splinedata
    # 1. see coordinates's norm
    b.splinedata["me"] = b
    b.show_options["dataname"] = "me"
    b.show()
    # 1.1 default scalarbar
    b.show_options["scalarbar"] = True
    b.show()
    b.show_options["scalarbar"] = False

    # 2. see coordinate's norm and as arrow
    b.show_options["arrowdata"] = "me"
    b.show()

    # 3. see coordinates norm and as arrows only on specified places
    b.show_options["arrowdata_on"] = np.random.random((100, 2))  # para_coords
    b.show()

    # 4. see 3. with parametric_view
    b.show(parametric_space=True)

    # 5. plot function that supports both resolutions and on
    def plot_func(data, resolutions=None, on=None):
        """
        callback to evaluate derivatives
        """
        if resolutions is not None:
            q = gus.create.vertices.raster(
                [[0, 0], [1, 1]], resolutions
            ).vertices
            return data.derivative(q, [0, 1])
        elif on is not None:
            return data.derivative(on, [0, 1])

    plot_func_data = gus.spline.SplineDataAdaptor(b, function=plot_func)
    b.splinedata["der01"] = plot_func_data
    b.show_options["arrowdata"] = "der01"
    b.show()  # arrowdata_on from 3.

    # remove on to sample same way as spline.
    # however, gold
    b.show_options.pop("arrowdata_on")  # remove
    b.show_options["arrowdata_color"] = "gold"
    b.show()  # resolutions

    # 6. plot on predefined place - this will be only available as arrow data
    locations = np.repeat(np.linspace(0, 1, 15), 2).reshape(-1, 2)
    values = np.repeat(np.linspace(1, 2, 15), 3).reshape(-1, 3) + [0, 0, 2]
    fixed_data = gus.spline.SplineDataAdaptor(values, locations=locations)
    b.splinedata["fixed"] = fixed_data
    b.show_options["arrowdata"] = "fixed"
    # let's turn off scalar field
    b.show_options.pop("dataname")
    b.show()

    # fixed location data can't be shown in other requested locations.
    # followings won't work
    # b.show_options["arrowdata_on"] = locations[:5]
    # b.show()

    # 7. plot any data with a function
    # some manually defined deformed spline
    deformed = b.copy()  # minimal copy - properties and cached data only
    deformed.cps[11, -1] -= 4
    deformed.cps *= [6, 6, 6]
    deformed.cps += [-5, 0, 8]
    deformed.cps[0, [0, -1]] += 4
    deformed.show_options["c"] = "hotpink"

    # define callback
    def func(self_and_deformed, resolutions=None, on=None):
        """
        callback to sample displacements.
        """
        # unpack data
        self, deformed = self_and_deformed
        if resolutions is not None:
            return deformed.sample(resolutions) - self.sample(resolutions)
        elif on is not None:
            return deformed.evaluate(on) - self.evaluate(on)

    # use adaptor - data is used as the first arg for callback
    deformed_data = gus.spline.SplineDataAdaptor(
        data=(b, deformed),
        function=func,
    )
    b.splinedata["deformed"] = deformed_data
    b.show_options["arrowdata"] = "deformed"
    # arrows are always automatically scaled. for this one, let's not
    b.show_options["arrowdata_scale"] = 1
    b.show_options["arrowdata_on"] = locations
    # let's see in parametric space
    p_view = b.create.parametric_view()  # shallow copies data and options
    p_view.show_options.pop("arrowdata_scale")  # too big for p_view
    p_view.show_options["dataname"] = "deformed"
    # plot side by side
    gus.show([b, deformed], ["Parametric view of displacements", p_view])
    # say, deformed has changed again - plots should adapt automatically
    deformed.cps[:, [1, 2]] = deformed.cps[:, [2, 1]]
    gus.show([b, deformed], ["Parametric view of displacements", p_view])

    # 8. fixed location data that uses callback
    # predefind some locations
    bottom = gus.spline.create.arc(radius=0.5, angle=-180)  # zero centered
    bottom.cps += [0.5, 0.55]
    circle1 = gus.spline.create.circle(radius=0.1)
    circle2 = circle1.copy()
    circle1.cps += [0.25, 0.75]
    circle2.cps += [0.75, 0.75]
    n = 30
    locations = np.vstack(
        (bottom.sample(n), circle1.sample(n), circle2.sample(n))
    )

    # callback
    def heights(bot_c1_c2_n):
        bot, c1, c2, n_repeat = bot_c1_c2_n

        values = np.repeat(
            [
                [0, 0, bot],
                [0, 0, c1],
                [0, 0, c2],
            ],
            n_repeat,
            axis=0,
        )
        return values

    nice_data = gus.spline.SplineDataAdaptor(
        data=(4, 2, 1, n),
        locations=locations,
        function=heights,
    )

    disc = gus.spline.create.disk(2, angle=123)
    disc.normalize_knot_vectors()
    disc.splinedata["nice"] = nice_data
    disc.show_options["arrowdata"] = "nice"
    disc.show_options["arrowdata_color"] = "jet"
    disc.show()
