import uuid

import gustaf as gus
import numpy as np
import vedo


def new_uuid():
    return str(uuid.uuid4())

def req_template():
    return {"id": new_uuid()}

def sendable_str(dict_):
    return str(dict_).replace("'", '"').encode()

def iganet_to_gus(data_dict):
    dict_spline = data_dict["data"]
    for_gus = {}
    for_gus["degrees"] = dict_spline["degrees"]
    coeffs = dict_spline["coeffs"]
    for_gus["control_points"] = [
        [x, y] for x, y in zip(coeffs[0], coeffs[1])
    ]
    for_gus["knot_vectors"] = dict_spline["knots"]
    return gus.BSpline(**for_gus)

class IganetBSpline(vedo.Plotter):
    """
    Self contained interactive plotter for 2P2D splines.
    Can move control points on the right plot and the left plot will
    show the displacement field in parametric space view.
    """

    def __init__(self, url):
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
        N = 2
        interactive = False
        sharecam = False

        super().__init__(
            N=N, interactive=interactive, sharecam=sharecam,
        )

        # plotter mode
        self.mode = "TrackballActor"

        # add callbacks
        self.add_callback("Interaction", self._update)
        self.add_callback("LeftButtonPress", self._left_click)
        self.add_callback("LeftButtonRelease", self._left_release)
        self.at(1).add_button(
            self._iganet_eval_button,
            pos=(0.7, 0.05),  # x,y fraction from bottom left corner
            states=["sync & evaluate"],
            c=["w", "w"],
            bc=["dg", "dv"],  # colors of states
            font="courier",   # arial, courier, times
            size=25,
            bold=True,
            italic=False,
        )


        # connect
        self.host = gus.utils.comm.WebSocketClient(url)

        # basic setup
        session_req = {
            "request": "create/session",
            "id": new_uuid(),
        }
        self.session_id = eval(
            self.host.send_recv(sendable_str(session_req))
        )["data"]["id"]
        # once the communication is truely online, we should assert UUID match

        # create spline at host
        spline_create_req = req_template()
        spline_create_req["request"] = f"create/{self.session_id}/BSplineSurface"
        spline_create_req["data"] = {"degree" : 2}
        self.spline_id = int(eval(
            self.host.send_recv(sendable_str(spline_create_req))
        )["data"]["id"])

        # create same spline local
        host_spline_req = req_template()
        host_spline_req["request"] = f"get/{self.session_id}/{self.spline_id}"
        self.host_spline_raw = eval(
            self.host.send_recv(sendable_str(host_spline_req))
        )
        self.spline = iganet_to_gus(self.host_spline_raw)

    def _iganet_eval_button(self):
        # clear
        if hasattr(self, "eval_v"): 
            self.at(1).remove(self.eval_v)

        # sync
        cp_update_req = req_template()
        cp_update_req["request"] = f"put/{self.session_id}/{self.spline_id}/coeffs"
        cp_update_req["data"] = {
            "indices": [i for i in range(len(self.spline.cps))],
            "coeffs": self.spline.cps.tolist()
        }
        self.host.send_recv(sendable_str(cp_update_req))

        # evaluate
        eval_req = req_template()
        eval_req["request"] = f"eval/{self.session_id}/{self.spline_id}"
        eval_req["data"] = {"resolution" : [11,11]}

        evaluated_raw = eval(
            self.host.send_recv(sendable_str(eval_req))
        )["data"]
        #self.eval_v = vedo.Points(
        self.eval_v = gus.Vertices(
            [[x, y] for x, y in zip(evaluated_raw[0], evaluated_raw[1])],
            #r=10,
        )
        self.eval_v.show_options["r"] = 10
        self.eval_v.show_options["vertex_ids"] = True
        self.eval_v = self.eval_v.showable()
        self.at(1).show(self.eval_v)

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
        if not hasattr(evt.actor, "id"):
            return None
        self.picked = evt.actor.id

        if hasattr(self, "eval_v"): 
            self.at(1).remove(self.eval_v)

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
        self.spline = iganet_to_gus(self.host_spline_raw)
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
        if evt.title != self.title:
            return None

        # get new control point location
        self.p = self.compute_world_coordinate(evt.picked2d)[:2]

        # remove
        self.at(0).remove(*self.spline_showable.values())

        # update cp
        self.spline.control_points[self.picked] = self.p

        # setup showable based on updated cp
        self._setup_showable()

        # add set showables
        self.at(0).add(*self.spline_showable.values())

    def _setup_showable(self):
        """
        """
        res = [50,50]
        self.spline.show_options["resolutions"] = res
        self.spline.show_options["control_points"] = False
        self.spline.show_options["control_mesh"] = True
        self.spline.show_options["lighting"] = "off"

        self.spline_showable = self.spline.showable()

        for v in self.spline_showable.values():
            v.pickable(False)

        # setup cps
        for i, cp in enumerate(self.spline.cps):
            s = vedo.Sphere(cp, r=0.05, res=5)
            s.id = i
            self.spline_showable[f"cp{i}"] = s


    def start(self):
        self._setup_showable()
        self.title = "Hello IgaNets, Hello Matthias"
        self.show(
            "Server response:", at=1)
        self.show(
            *self.spline_showable.values(),
            at=0,
            title=self.title,
            interactive=True,
            mode=self.mode,
        )

        # disconnect
        del self.host

        return self
