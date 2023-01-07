import gustaf as gus
import numpy as np
import vedo


class Interactive2DControlPoints(vedo.Plotter):
    """
    Self contained interactive plotter for 2P2D splines.
    Can move control points on the right plot and the left plot will
    show the displacement field in parametric space view.
    """

    def __init__(self, spl=None, **kwargs):
        """
        Creates spline and initialize all callbacks

        Parameters
        ----------
        spl: spline
        **kwargs: kwargs
          No effect for now.

        Returns
        -------
        None
        """
        # N = kwargs.get("N", 2)
        # interactive = kwargs.get("interactive", False)
        # sharecam = kwargs.get("sharecam", False)
        N = 2
        interactive = False
        sharecam = False

        super().__init__(
            N=N, interactive=interactive, sharecam=sharecam, **kwargs
        )

        # plotter mode
        self.mode = "TrackballActor"

        # setup spline if defined
        if spl is not None:
            self.spline = spl
            self.o_spline = spl.copy()
            # right button reset only for spl, for now.
            self.add_callback("RightButtonPress", self._right_click)
        else:
            # nothing defined - demo mode!
            # define a spline
            self.spline = gus.spline.create.disk(
                outer_radius=2, inner_radius=1, angle=175, n_knot_spans=3
            )
            # call demo disk - double work here, but setup properly
            self._new_demo_disk(175)

        self.spline.show_options["control_points"] = False
        self.spline.show_options["control_mesh"] = True
        self.spline.show_options["lighting"] = "off"

        # add callbacks
        self.add_callback("Interaction", self._update)
        self.add_callback("LeftButtonPress", self._left_click)
        self.add_callback("LeftButtonRelease", self._left_release)

    def _new_demo_disk(self, angle):
        """
        Prepares disk with given angle.

        Parameters
        ----------
        angle: float

        Returns
        -------
        None
        """
        # create disk
        d = gus.spline.create.disk(
            outer_radius=2, inner_radius=1, angle=angle, n_knot_spans=3
        )
        # just update control points and weights
        self.spline.cps[:] = d.cps
        self.spline.ws = d.ws
        self.o_spline = self.spline.copy()
        self.w_val = angle

    def _left_click(self, evt):
        """
        Left click selects and registers control point id to follow.

        Parameters
        ----------
        evt: vedo.Event

        Returns
        -------
        None
        """
        # nothing picked return
        if not evt.actor:
            return None
        # only cps should be pickable and they have ids assigned
        self.picked = evt.actor.id

    def _left_release(self, evt):
        """
        Left click release - resets registered control point id.

        Parameters
        ----------
        evt: vedo.Event

        Returns
        -------
        None
        """
        self.picked = -1

    def _right_click(self, evt):
        """
        Restart. Currently some issue with demo

        Parameters
        ----------
        evt: vedo.Event

        Returns
        -------
        None
        """
        # clear
        self.clear(at=0, deep=True, render=True)
        self.clear(at=1, deep=True, render=True)

        # reassign
        self.spline = self.o_spline.copy()
        self.spline.show_options["control_points"] = False
        self.spline.show_options["control_mesh"] = True
        self.spline.show_options["lighting"] = "off"

        # restart
        self.start()

    def _update(self, evt):
        """
        Interaction / Drag event. Cleans and add updated ones

        Parameters
        ----------
        evt: vedo.Event

        Returns
        -------
        None
        """
        if self.picked < 0:
            return None
        if evt.title != "Interactive 2D Control Points":
            return None

        # get new control point location
        self.p = self.compute_world_position(evt.picked2d)[:2]

        # remove
        self.at(1).remove(*self.spline_showable.values())
        self.at(0).remove(self.sampled_para)

        # update cp
        self.spline.control_points[self.picked] = self.p

        # setup showable based on updated cp
        self._setup_showable()

        # add set showables
        self.at(1).add(*self.spline_showable.values())
        self.at(0).add(self.sampled_para)

    def _setup_showable(self):
        """
        """
        self.spline.show_options["resolutions"] = [30, 100]
        self.para_spline.show_options["resolutions"] = [30, 100]
        self.spline_showable = self.spline.showable()
        for v in self.spline_showable.values():
            v.pickable(False)

        # setup cps
        for i, cp in enumerate(self.spline.cps):
            s = vedo.Sphere(cp, r=0.1, res=5)
            s.id = i
            self.spline_showable[f"cp{i}"] = s

        # set up p_view
        p_showable = self.para_spline.showable()
        self.sampled_para = p_showable["spline"].pickable(False).add_scalarbar()

    def _slider(self, widget, event):
        self.at(1).remove(*self.spline_showable.values())
        self.at(0).remove(
            self.sampled_para,
        ) 
        self._new_demo_disk(widget.value)
        self.spline.show_options["control_points"] = False
        self.spline.show_options["control_mesh"] = True
        self.spline.show_options["lighting"] = "off"
        #self.start()
        disp_data = gus.spline.SplineDataAdaptor(
            data=(self.spline.copy(), self.spline),
            function=self._disp_callback,
        )
        self.para_spline.splinedata["disp"] = disp_data

        self._setup_showable()
        self.at(1).add(*self.spline_showable.values()).reset_camera()

 
    # let's plot displacements
    def _disp_callback(self, orig_deformed, resolutions=None, on=None):
        """
        callback to sample disp.
        """
        orig, deformed = orig_deformed
        if resolutions is not None:
            return deformed.sample(resolutions) - orig.sample(resolutions)
        elif on is not None:
            return deformed.evaluate(on) - orig.evaluate(on)

    def start(self):
        # setup paraspline and already setup
        self.para_spline = self.spline.create.parametric_view()
        self.para_spline.show_options["control_points"] = False
        self.para_spline.show_options["control_mesh"] = False


        disp_data = gus.spline.SplineDataAdaptor(
            data=(self.spline.copy(), self.spline),
            function=self._disp_callback,
        )
        self.para_spline.splinedata["disp"] = disp_data
        self.para_spline.show_options["dataname"] = "disp"
        p_showable = self.para_spline.showable()
        sampled = p_showable.pop("spline")
        self.sampled_para = sampled.actors[-1]
        p_showable["spline"] = sampled.actors[:-1]

        self.show(
            *p_showable.values(), title="parametric", at=0, mode=self.mode
        )
        self.at(0).show("Displacement field in parametric space view")
        self.para_spline.show_options["knots"] = False
        self.para_spline.show_options["axes"] = False

        self._setup_showable()

        self.show(
            *self.spline_showable.values(),
            at=1,
            title="Interactive 2D Control Points",
            mode=self.mode,
        )
        if hasattr(self, "w_val"):
            self.add_slider(
                self._slider,
                xmin=10,
                xmax=350,
                value=self.w_val,
                pos=([0.1, 0.1], [0.4, 0.1]),
                title="Angle in degrees",
                title_size=2,
            )

        self.at(1).interactive()
        return self


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
        [0, 0],
        [0, 1],
        [1, 1.5],
        [3, 1.5],
        [-1, 0],
        [-1, 2],
        [1, 4],
        [3, 4],
        [-2, 0],
        [-2, 2],
        [1, 5],
        [3, 5],
    ]
)

# init bspline
b = gus.BSpline(
    degrees=ds2,
    knot_vectors=kvs2,
    control_points=cps2,
)

#plt = Interactive2DControlPoints(b).start()

# demo
plt = Interactive2DControlPoints().start()
