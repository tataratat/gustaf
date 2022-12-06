"""gustaf/gustaf/ffd.py.

Freeform Deformation!


Adaptation of previous implementation in internal python package gustav by
Jaewook Lee.
"""
from typing import Any, List, Optional, Union
import numpy as np
from gustaf._base import GustafBase
from gustaf.show import show_vedo
from gustaf._typing import SPLINE_TYPES, MESH_TYPES
from gustaf.create.spline import with_bounds
from gustaf import settings


class FFD(GustafBase):

    def __init__(
            self,
            mesh: Optional[MESH_TYPES] = None,
            spline: Optional[SPLINE_TYPES] = None
    ):
        """
        Free-form deformation is a method used to deform an object by a
        deformation function. In our case the object is given via a mesh, the
        currently supported mesh-types are given by the variable
        :py:const:`gustaf._typing.MESH_TYPES`, and the deformations function
        by a spline, supported splines are given by the variable
        :py:const.:`gustaf._typing.SPLINE_TYPES`. The splines parametric
        dimension will be scaled in to a unit-hypercube as well as the
        original meshes vertices. The outline of the resulting mesh is given
        by the physical space of the spline.

        The FFD class provides functions to modify the spline by completely
        overwriting the spline whole spline or parts of it. To obtain the
        deformed mesh mapped into the latest spline, retrieve the mesh
        attribute.

        Please not that even though an object of the class can be initiated
        without a mesh, it is not possible to compute the deformation without
        one. Please ensure that at least a mesh is defined before retrieving
        the (deformed) mesh. If only a mesh is provided a default spline where
        the geometric dimensions have the bounds of the mesh is defined.

        A previously available partial FFD is currently not implemented, and
        is planned to be implemented in a separate class (LocalFFD).

        Parameters
        ----------
        mesh: Optional[MESH_TYPES]
            Mesh used in the FFD. Defaults to None.
        spline: Optional[SPLINE_TYPES]
            Spline used in the FFD. Defaults to None.

        Class Attributes
        ----------------
        _spline: SPLINE_TYPES
            Internal current spline
        _mesh: MESH_TYPES
            unscaled base mesh
        _q_vertices: np.ndarray (n, dim)
            Scaled vertices of the base mesh

        Returns
        -------
        None
        """
        # Use property definitions to store the values
        self._spline: SPLINE_TYPES = None
        self._mesh: MESH_TYPES = None
        self._o_mesh: MESH_TYPES = None
        self._q_vertices: np.ndarray = None

        if spline is not None:
            self.spline = spline
        if mesh is not None:
            self.mesh = mesh

        # self._is_calculated = False

    @property
    def mesh(self, ) -> MESH_TYPES:
        """Returns copy of current mesh. Before copying, it applies
        deformation.

        Returns
        -------
        MESH_TYPES
            Current Mesh with the deformation according to the current spline.
        """
        self._deform()
        return self._mesh.copy()

    @mesh.setter
    def mesh(self, mesh: MESH_TYPES):
        """Sets mesh. If it is first time, the copy of it will be saved as
        original mesh. If spline is already defined and in transformed status,
        it applies transformation directly.

        Parameters
        -----------
        mesh: MESH_TYPES
            Mesh used for the FFD

        Returns
        --------
        None
        """
        if self._spline is None:
            # Define a default spline if mesh is given but no spline
            par_dim = mesh.vertices.shape[1]
            self.spline = with_bounds(
                    [[0] * par_dim, [1] * par_dim], mesh.bounds()
            )

        self._logi("Setting mesh.")
        self._logi("Mesh Info:")
        self._logi("  Vertices: {v}.".format(v=mesh.vertices.shape))
        self._logi("  Bounds: {b}.".format(b=mesh.bounds()))
        self._o_mesh = mesh.copy()  # we keep original copy for visualization
        self._mesh = mesh.copy()  # another copy for current status.

        self._check_dimensions()

        self._scale_mesh_vertices()
        if self._spline:
            self._spline._data["gustaf_ffd_computed"] = False

    @property
    def spline(self):
        """Returns a copy of the spline. Please use the setter to explicitly
        make changes to the spline.

        Parameters
        -----------
        None

        Returns
        --------
        self._spline: Spline
        """
        return self._spline

    @spline.setter
    def spline(self, spline: SPLINE_TYPES):
        """Sets spline. The spline parametric range bounds will be converted
        into the bounds [0,1]^para_dim.

        Parameters
        -----------
        spline: SPLINE_TYPES
            New Spline for the next deformation

        Returns
        --------
        None
        """
        self._spline = spline

    def _check_dimensions(self) -> bool:
        """Checks if the dimension of the spline and the mesh match and

        Returns:
            bool: _description_
        """
        messages = []
        # Checks dimensions and ranges critical for a correct FFD calculation
        if self._spline and not self._spline.para_dim == self._spline.dim:
            messages.append(
                    "The parametric and geometric dimensions of the "
                    "spline are not the same."
            )
        if (
                self._spline and self._mesh
                and not self._spline.dim == self._mesh.vertices.shape[1]
        ):
            messages.append(
                    "The geometric dimensions of the spline and the "
                    "dimension of the mesh are not the same."
            )
        if len(messages) > 0:
            raise RuntimeError(
                    "Can not perform FFD due to spline and mesh "
                    "dimension mismatch. The following dimension mismatches:"
                    f"{messages}."
            )

    def _scale_mesh_vertices(self):
        """Scales the mesh vertices into the dimension of a hypercube and save
        them in self._q_vertices."""
        self._logd("Fitting mesh into spline's parametric space.")

        self._q_vertices = self._mesh.vertices.copy()

        original_mesh_bounds = self._mesh.bounds()

        # save mesh offset and scale for reasons
        self._mesh_offset = original_mesh_bounds[0]
        self._mesh_scale = 1 / (
                original_mesh_bounds[1] - original_mesh_bounds[0]
        )

        # scale and offset vertices coordinates
        self._q_vertices -= self._mesh_offset
        self._q_vertices *= self._mesh_scale

        self._logd("Successfully scaled and transformed mesh vertices!")

    def _deform(self):
        """Deforms mesh if spline or mesh changes were detected since last
        calculation. Meant for internal use.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        if self._mesh is None or self._spline is None:
            raise RuntimeError(
                    "Can't perform deformation for the FFD, since either the "
                    "spline or(and) the mesh are not yet defined. "
                    "Please set either spline or mesh."
            )
        if self._spline._data.get("gustaf_ffd_computed", False):
            return None

        spline = self._spline.copy()
        if spline.has_knot_vectors:
            spline.normalize_knot_vectors()

        self._check_dimensions()

        self._logd("Applying FFD: Transforming vertices")

        # Here, we take _q_vertices, due to possible scale/offset.
        self._mesh.vertices = spline.evaluate(self._q_vertices)
        self._logd("FFD successful.")

        self._spline._data["gustaf_ffd_computed"] = True

    @property
    def control_points(self):
        """Returns current spline's control points. The control points can be
        directly updated with this.

        Returns
        --------
        self._spline.control_points: np.ndarray
        """
        return self._spline.control_points

    @control_points.setter
    def control_points(
            self, control_points: Union[List[List[float]], np.ndarray]
    ):
        """Sets control points and deforms mesh.

        Parameters
        -----------
        control_points: np.ndarray

        Returns
        --------
        None
        """
        assert self._spline.control_points.shape == control_points.shape,\
            "Given control points' shape does not match current ones!"

        self._spline.control_points = control_points.copy()
        self._logd("Set new control points.")

    def elevate_degree(self, *args, **kwargs):
        """Wrapper for Spline.elevate_degree.

        Parameters
        -----------
        *args:
        **kwargs:

        Returns
        --------
        None
        """
        self._spline.elevate_degree(*args, **kwargs)

    def insert_knots(self, parametric_dimension, knots):
        """Wrapper for Spline.insert_knots.

        Parameters
        -----------
        *args:
        **kwargs:

        Returns
        --------
        None
        """
        if "knot_vectors" in self._spline.required_properties:
            raise NotImplementedError(
                    "Can not perform knot insertion on Bezier spline."
            )
        self._spline.insert_knots(parametric_dimension, knots)

    def remove_knots(self, parametric_dimension, knots, tolerance=1e-8):
        """Wrapper for Spline.remove_knots.

        Parameters
        -----------
        *args:
        **kwargs:

        Returns
        --------
        None
        """
        if "knot_vectors" in self._spline.required_properties:
            raise NotImplementedError(
                    "Can not perform knot insertion on Bezier spline."
            )
        self._spline.remove_knots(
                parametric_dimension, knots, tolerance=tolerance
        )

    def reduce_degree(self, *args, **kwargs):
        """Wrapper for Spline.reduce_degree.

        Parameters
        -----------
        *args:
        **kwargs:

        Returns
        --------
        None
        """
        self._spline.reduce_degree(*args, **kwargs)

    def show(self, **kwargs) -> Any:
        """Visualize. Shows the deformed mesh and the current spline. Currently
        visualization is limited to vedo.

        Parameters
        ----------
        title: str
            Title of the vedo window. Defaults to "gustaf - FFD".
        return_showable: bool
            If true returns a dict of the showable items. Defaults to False.
        return_discrete: bool
            Return dict of gustaf discrete objects, for example,
            {Vertices, Edges, Faces}, instead of opening a window.
            Defaults to False.
        kwargs: Any
            Arbitrary keyword arguments. These are passed onto the vedo
            functions. Please be aware, that no checking of these are performed
            in this function.

        Returns
        -------
        Any:
            Returns, if applicable, the vedo plotter. 'close=False' as argument
            to get the plotter.
        """
        backend = kwargs.pop("backend", None)
        return_showable = kwargs.pop("return_showable", False)
        return_discrete = kwargs.pop("return_discrete", False)
        title = kwargs.pop("title", "gustaf - FFD")

        if return_discrete and return_showable:
            raise ValueError(
                    "Either one of following params can be True: "
                    "{return_discrete, return_showable} "
                    "You've set both True."
            )

        if backend is None:
            backend = settings.VISUALIZATION_BACKEND

        # prepare originals
        o_mesh = self._o_mesh.copy()
        # prepare deformed
        d_mesh = self.mesh  # copies

        things_to_show = dict()
        # let's show faces at most, since volumes can take awhile
        if o_mesh.kind == "volume":
            # only outer faces. overwrite
            o_mesh = o_mesh.tofaces(unique=False)
            o_mesh.update_faces(o_mesh.single_faces())
            d_mesh = d_mesh.tofaces(unique=False)
            d_mesh.update_faces(d_mesh.single_faces())

        # update meshes
        things_to_show.update(original_mesh=o_mesh)
        things_to_show.update(original_description="Original Mesh")
        things_to_show.update(deformed_mesh=d_mesh)
        things_to_show.update(deformed_description="Deformed Mesh with Spline")

        # update spline
        things_to_show.update(deformed_spline=self.spline)

        if return_discrete or not backend.startswith("vedo"):
            # spline is strictly not discrete.
            return things_to_show

        if return_showable:
            # let's turn everthing into showable and return
            for k, v in things_to_show.items():
                if isinstance(v, GustafBase):
                    things_to_show[k] = v.showable()

            return things_to_show

        # current workaround to set spline's surface alpha correctly
        # TODO: support this situation better
        spl = things_to_show.pop("deformed_spline")
        spl_showable = spl.showable(surface_alpha=.3)

        return show_vedo(
                [
                        things_to_show[k]
                        for k in things_to_show.keys()
                        if k.startswith("original")
                ],
                [
                        *[
                                things_to_show[k]
                                for k in things_to_show.keys()
                                if k.startswith("deformed")
                        ],
                        *spl_showable.values(),
                ],
                title=title,
        )

    def showable(self, **kwargs):
        """Returns a dictionary of showable items to describe the FFD at the
        current state.

        See show() for more information. This function redirects to it
        directly with the return_showable keyword set to True.
        """
        return self.show(return_showable=True, **kwargs)
