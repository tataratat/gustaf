import uuid

from gustaf import utils
from gustaf.show import vedo
from gustaf.spline import BSpline
from gustaf.vertices import Vertices

try:
    VedoPlotter = vedo.Plotter

except BaseException:
    # ModuleImportRaiser can't be parent class.
    class VedoPlotter:
        pass


def new_uuid():
    """
    Returns a new uuid in str.

    Parameters
    ----------
    None

    Returns
    -------
    uuid_str: str
    """
    return str(uuid.uuid4())


def req_template():
    """
    Creates minimal request template.

    Parameters
    ----------
    None

    Returns
    -------
    basic_template: dict
      basic dict template with configured "id"(uuid)
    """
    return {"id": new_uuid()}


def sendable_str(dict_):
    """Given dict, prepares a str that's readable from iganet server.

    Parameters
    ----------
    dict_: dict
      json request. No further conversion is required as long as all the values
      are python native type

    Returns
    -------
    sendable_str: str
      string encoded in bytes.
    """
    return str(dict_).replace("'", '"').encode()


def iganet_to_gus(data_dict):
    """
    Creates spline based on iganet's response dict.
    Note that you have to eval(raw_string_response)!

    Parameters
    ----------
    data_dict: dict

    Returns
    -------
    gustaf_bspline: gus.BSpline
    """
    dict_spline = data_dict["data"]

    # prepare dict for gus.BSpline init
    for_gus = {}

    # degrees
    for_gus["degrees"] = dict_spline["degrees"]

    # coeffs (control_points) are raw major, so let's reorganize that
    coeffs = dict_spline["coeffs"]
    for_gus["control_points"] = [[*x] for x in zip(*coeffs)]

    # knot_vectors
    for_gus["knot_vectors"] = dict_spline["knots"]

    return BSpline(**for_gus)


class IganetBSpline(VedoPlotter):
    """
    Self contained interactive plotter for 2P2D splines.
    Can move control points on the right plot and the left plot will
    show the displacement field in parametric space view.
    """

    def __init__(self, uri):
        """
        Creates spline and initialize all callbacks

        Parameters
        ----------
        uri: str

        Returns
        -------
        None
        """
        # temporary solution to replace ModuleImportRaiser
        vedo.Plotter  # call vedo.Plotter to see if vedo is properly imported

        # plotter intialization constants
        N = 2
        interactive = False
        sharecam = False

        # set title for window
        self.window_title = "Hello IgaNets, Hello Matthias"

        super().__init__(
            N=N,
            interactive=interactive,
            sharecam=sharecam,
            title=self.window_title,
        )

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
            font="courier",  # arial, courier, times
            size=25,
            bold=True,
            italic=False,
        )

        # connect
        self.server = utils.comm.WebSocketClient(uri)

        # basic setup
        session_req = {
            "request": "create/session",
            "id": new_uuid(),
        }
        self.session_id = eval(
            self.server.send_recv(sendable_str(session_req))
        )["data"]["id"]
        # once the communication is truely online, we should assert UUID match

        # create spline at server
        # let's ask for BSplineSurface of degree 2
        spline_create_req = req_template()
        spline_create_req[
            "request"
        ] = f"create/{self.session_id}/BSplineSurface"
        spline_create_req["data"] = {"degree": 2}
        # from this request, all we need is created spline's id.
        self.spline_id = int(
            eval(self.server.send_recv(sendable_str(spline_create_req)))[
                "data"
            ]["id"]
        )

        # create same spline local
        server_spline_req = req_template()
        server_spline_req[
            "request"
        ] = f"get/{self.session_id}/{self.spline_id}"
        self.server_spline_raw = eval(
            self.server.send_recv(sendable_str(server_spline_req))
        )
        self.spline = iganet_to_gus(self.server_spline_raw)
        print(f"created {self.spline.whatami}")

        # plotter mode
        if self.spline.dim == 2:
            # self.default_mode = vtk.vtkInteractorStyleTrackballActor()
            self.default_mode = "TrackballActor"
        elif self.spline.dim == 3:
            # self.default_mode = vtk.vtkInteractorStyleTrackballCamera()
            self.default_mode = "TrackballCamera"
        else:
            raise ValueError(
                "Interactive mode supports splines of 2 and 3 dim."
            )

        # initialize picked
        self.picked = -1

    def _iganet_eval_button(self):
        """
        Button callback - asks for evaluation
        """
        # clear
        if hasattr(self, "eval_v"):
            self.at(1).remove(self.eval_v)

        # sync
        cp_update_req = req_template()
        cp_update_req[
            "request"
        ] = f"put/{self.session_id}/{self.spline_id}/coeffs"
        cp_update_req["data"] = {
            "indices": [i for i in range(len(self.spline.cps))],
            "coeffs": self.spline.cps.tolist(),
        }
        self.server.send_recv(sendable_str(cp_update_req))

        # evaluate
        eval_req = req_template()
        eval_req["request"] = f"eval/{self.session_id}/{self.spline_id}"
        eval_req["data"] = {"resolution": [11, 11]}

        evaluated_raw = eval(self.server.send_recv(sendable_str(eval_req)))[
            "data"
        ]
        self.eval_v = Vertices(
            [[*x] for x in zip(*evaluated_raw)],
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
        Restart.

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
        self.spline = iganet_to_gus(self.server_spline_raw)
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
        self.p = self.compute_world_coordinate(evt.picked2d)[: self.spline.dim]

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
        Prepares vedo-show-ready objects, as well as control points that we can
        pick.
        """
        res = [50, 50]  # sample resolution
        self.spline.show_options["resolutions"] = res
        self.spline.show_options["control_points"] = False
        self.spline.show_options["control_mesh"] = True
        self.spline.show_options["lighting"] = "off"

        # splines return showable as dict
        self.spline_showable = self.spline.showable()

        # we don't want to pick any of spline
        for v in self.spline_showable.values():
            v.pickable(False)

        # setup cps - those are pickable!
        for i, cp in enumerate(self.spline.cps):
            s = vedo.Sphere(cp, r=0.05, res=5)
            s.id = i
            self.spline_showable[f"cp{i}"] = s

    def start(self):
        """
        Start the interative mode.

        Returns
        -------
        plot: IganetBSpline
        """
        self._setup_showable()
        self.show("Server response:", at=1)
        self.show(
            *self.spline_showable.values(),
            at=0,
            interactive=True,
            mode=self.default_mode,
        )

        # close once interactive session is over.
        self.server.websocket.close()

        return self
