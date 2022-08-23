"""
gustaf/gustaf/ffd.py

Freeform Deformation!


Adaptation of previous implementation in internal python package gustav by
Jaewook Lee. 
"""
from typing import List, Optional, Union
import numpy as np
from gustaf._base import GustafBase
from gustaf.faces import Faces
from gustaf.show import show_vedo
from gustaf._typing import SPLINE_TYPES, MESH_TYPES
from gustaf.create.spline import with_bounds
from gustaf import settings


class FFD (GustafBase):

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
        _is_calculated: bool 
            Attribute tracking if changes are present since last calculation

        Returns
        -------
        None
        """
        # Use property definitions to store the values
        self._spline: SPLINE_TYPES = None
        self._mesh: MESH_TYPES = None
        self._q_vertices: np.ndarray = None
        self._is_calculated: bool = False

        if spline is not None:
            self.spline = spline
        if mesh is not None:
            self.mesh = mesh

        self._is_calculated = False

    def _check_dimensions(self):
        """
        Checks dimensions and ranges critical for a correct FFD calculation

        TODO: range of spline is same as geometric range of mesh
        """
        if not self._spline.para_dim == self._spline.dim:
            self._logw("The parametric and geometric dimensions of the "
                       "spline are not the same.")
        if not self._spline.dim == self._mesh.vertices.shape[1]:
            self._logw("The geometric dimensions of the spline and the "
                       "dimension of the mesh are not the same.")

    @property
    def mesh(self,) -> MESH_TYPES:
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
        """
        Sets mesh. If it is first time, the copy of it will be saved as 
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
                [[0]*par_dim, [1]*par_dim],
                mesh.bounds)

        self._logi("Setting mesh.")
        self._logi("Mesh Info:")
        self._logi(
            "  Vertices: {v}.".format(v=mesh.vertices.shape)
        )
        self._logi(
            "  Bounds: {b}.".format(b=mesh.get_bounds())
        )
        self._mesh = mesh.copy()  # copy to make sure given mesh stays untouched
        self._check_dimensions()
        self._scale_mesh_vertices()
        self._is_calculated = False

    @property
    def spline(self):
        """
        Returns a copy of the spline. Please use the setter to explicitly make
        changes to the spline.

        Parameters
        -----------
        None

        Returns
        --------
        self._spline: Spline 
        """
        self._logd("Returning copy of current spline.")
        return self._spline.copy() if self._spline is not None else None

    @spline.setter
    def spline(self, spline: SPLINE_TYPES):
        """
        Sets spline. The spline parametric range bounds will be converted 
        into the bounds [0,1]^para_dim.

        Parameters
        -----------
        spline: SPLINE_TYPES
            New Spline for the next deformation

        Returns
        --------
        None
        """
        self._spline = spline.copy()
        self._scale_parametric_dimension_to_hypercube()

        self._logi("Setting Spline.")
        self._logi("Spline Info:")
        self._logi(
            f"  Parametric dimensions: {spline.para_dim}."
        )
        self._is_calculated = False

    def _scale_parametric_dimension_to_hypercube(self):
        """Scales all knot_vectors of the spline to a range of [0,1]
        """
        if not self._is_parametric_room_hypercube(self._spline):
            knv = []
            for knot_vector in self._spline.knot_vectors:
                knot_vector = [kv_v - knot_vector[0] for kv_v in knot_vector]
                knot_vector = [kv_v / knot_vector[-1] for kv_v in knot_vector]
                knv.append(knot_vector)
            self._spline.knot_vectors = knv

    def _is_parametric_room_hypercube(
        self,
        spline: SPLINE_TYPES
    ) -> bool:
        """Checks if the parametric range of the spline is a hypercube.

        Parameters
        ----------
        spline : SPLINE_TYPES
            New spline for which to check for hypercubeness

        Returns
        -------
        bool
            True if parametric room is hypercube, else False
        """
        if "knot_vectors" in spline.required_properties:
            for knot_vector in spline.knot_vectors:
                # check if knot_vectors fist element is 0 and last element is 1
                if not (
                        np.isclose(knot_vector[0], 0, atol=settings.TOLERANCE)
                        and
                        np.isclose(knot_vector[-1], 1, atol=settings.TOLERANCE)
                ):
                    return False
        return True

    def _scale_mesh_vertices(self):
        """
        Scales the mesh vertices into the dimension of a hypercube and save 
        them in self._q_vertices.
        """
        self._logd("Fitting mesh into spline's parametric space.")

        self._q_vertices = self._mesh.vertices.copy()

        original_mesh_bounds = self._mesh.get_bounds()

        # save mesh offset and scale for reasons
        self._mesh_offset = original_mesh_bounds[0]
        self._mesh_scale = 1/(original_mesh_bounds[1] -
                              original_mesh_bounds[0])

        # scale and offset vertices coordinates
        self._q_vertices -= self._mesh_offset
        self._q_vertices *= self._mesh_scale

        self._logd("Successfully scaled and transformed mesh vertices!")

    def _deform(self):
        """
        Deforms mesh if spline or mesh changes were detected since last 
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
                "Can not perform deformation for the FFD, since either the "
                "spline and/or the mesh are not yet defined. Please define "
                "both a spline and mesh before deforming the mesh.")
            return
        if self._is_calculated:
            return

        self._logd("Applying FFD: Transforming vertices")

        # Here, we take _q_vertices, due to possible scale/offset.
        self._mesh.vertices = self._spline.evaluate(
            self._q_vertices
        )
        self._logd("FFD successful.")

        self._is_calculated = True

    @property
    def control_points(self):
        """
        Returns current spline's control points. Can't and don't use this to
        directly manipulate control points: it won't update

        Returns
        --------
        self._spline.control_points: np.ndarray
        """
        return self._spline.control_points.copy()

    @control_points.setter
    def control_points(self,
                       control_points: Union[List[List[float]], np.ndarray]
                       ):
        """
        Sets control points and deforms mesh.

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
        self._is_calculated = False

    def deform_for_given_cp(self, values: np.ndarray, mask=None) -> MESH_TYPES:
        """
        Set the new control_point values according to the mask and returns the 
        deformed mesh.

        #TODO I am not sure if this works since CPs might not always be ndarray

        Parameters
        -----------
        values: np.ndarray
          Values to translate. If it is 1D array, it is applied to all masked
          control points.
        mask: list-like
          (Optional) Default is None. If None, it will be applied to all.
          It can be bool or int.

        Returns
        --------
        deformed_mesh: MESH_TYPES
        """
        if mask is None:
            self.control_points += values
        else:
            cp = self.control_points
            cp[mask] += values
            self.control_points = cp

        self._is_calculated = False  # should already be set, but to be safe
        return self.mesh

    def elevate_degree(self, *args, **kwargs):
        """
        Wrapper for Spline.elevate_degree

        Parameters
        -----------
        *args:
        **kwargs:

        Returns
        --------
        None
        """
        self._spline.elevate_degree(*args, **kwargs)

    def insert_knots(self, *args, **kwargs):
        """
        Wrapper for Spline.insert_knots

        Parameters
        -----------
        *args:
        **kwargs:

        Returns
        --------
        None
        """
        self._spline.insert_knots(*args, **kwargs)

    def remove_knots(self, *args, **kwargs):
        """
        Wrapper for Spline.remove_knots

        Parameters
        -----------
        *args:
        **kwargs:

        Returns
        --------
        None
        """
        self._spline.remove_knots(*args, **kwargs)
        self._is_calculated = False

    def reduce_degree(self, *args, **kwargs):
        """
        Wrapper for Spline.reduce_degree

        Parameters
        -----------
        *args:
        **kwargs:

        Returns
        --------
        None
        """
        self._spline.reduce_degree(*args, **kwargs)
        self._is_calculated = False

    def show(self, title: str = "Gustaf - FFD", **kwargs):
        """
        Visualize. Shows the deformed mesh and the current spline.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        original_mesh = self._mesh.copy()
        original_mesh.vertices = self._q_vertices
        if original_mesh.kind == "volume":
            orig_mesh_outer_faces = Faces(
                original_mesh.vertices,
                original_mesh.get_faces()[
                    original_mesh.get_surfaces()]
            )
            mesh_outer_faces = Faces(
                self.mesh.vertices,
                self.mesh.get_faces()[
                    self.mesh.get_surfaces()]
            )
            show_vedo(
                ["Original Mesh",
                 original_mesh,
                 orig_mesh_outer_faces.toedges(unique=True)],
                ["Deformed Mesh with Spline",
                 #  self.mesh.showable(),
                 mesh_outer_faces.toedges(unique=True),
                 *self._spline.showable().values()],
                title=title, **kwargs
            )
        else:
            show_vedo(
                ["Original Mesh",
                 original_mesh,
                 original_mesh.toedges(unique=True)],
                ["Deformed Mesh with Spline",
                 #  self.mesh.showable(),
                 self.mesh.toedges(unique=True),
                 *self._spline.showable().values()],
                title=title, **kwargs
            )
