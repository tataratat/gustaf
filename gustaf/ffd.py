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
from gustaf.spline.base import Bezier
from gustaf.show import show_vedo
from gustaf._typing import SPLINE_TYPES, MESH_TYPES
from gustaf.create.spline import with_bounds


class FFD (GustafBase):

    def __init__(
        self,
        mesh: MESH_TYPES,
        spline: Optional[SPLINE_TYPES],
        # is_partial: bool=False,
    ):
        """
        FreeFormDeformation. Here, spline is always the boss. Meaning, once
        spline is defined using setter, we always fit mesh into the spline.
        For partial FFD, no one needs to be the boss. If no spline is defined,
        a simple spline wraps given mesh as default.

        Parameters
        ----------
        mesh: MESH_TYPES
            Base mesh of the FFD
        spline: SPLINE_TYPE
            Original Spline of the FFD
        is_partial: bool
            True if mesh is only partially deformed (currently not supported)

        Attributes
        ----------
        _spline: SPLINE_TYPES
        _mesh: MESH_TYPES
        _original_spline_ranges: List[List[float]]
        _q_vertices: np.ndarray (n, dim)
        _is_partial: bool
        calculated: bool

        Returns
        -------
        None
        """
        # Use property definitions to store the values
        self._spline = None
        self._mesh: MESH_TYPES = None
        self._original_spline_ranges = None
        self._q_vertices = None
        self._is_partial = False
        self.calculated = False

        self.mesh = mesh
        if spline is None:
            # if no spline is given create a very naive spline with physical
            # bounds defined bny the mesh and parametric bounds defined by a 
            # hypercube
            self.spline = with_bounds(
                [[0]*self._q_vertices.shape[1],[1]*self._q_vertices.shape[1]],
                 self._mesh.bounds)
        else:
            self.spline = spline
        self.is_partial = False



        self._check_dimensions()

    
    def _check_dimensions(self):
        """Checks dimensions and ranges critical for a correct FFD calculation

        The following things are checked:
        is_partial:

            Currently nothing since not implemented

        else:
            parametric dimension of spline is equal to geometric dimension of 
            spline

            geometric dimension of spline is the same as mesh dimension

            # can currently not be checked
            geometric range of spline is same as geometric range of mesh
        """
        if self._is_partial:
            pass
        else:
            if not self._spline.para_dim == self._spline.dim:
                self._logw("The parametric and geometric dimensions of the "
                           "spline are not the same.")
            if not self._spline.dim == self._mesh.vertices.shape[1]:
                self._logw("The geometric dimensions of the spline and the "
                           "dimension of the mesh are not the same.")
            # No easy way to compare the bounds of the spline and mesh
            # since the spline could already be deformed when initializing
            
            # if not self._spline.control_point_bounds == self._mesh.bounds:
            #     self._logw("The bounds of the control "
            #                "spline are not the same.")


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
        if self._mesh is not None:
            self._logw("Resetting original Mesh, this is not intended "
            "behavior. Please create a new FFD object when replacing the mesh"
            ".")

        self._logi("Setting mesh.")
        self._logi("Mesh Info:")
        self._logi(
            "  Vertices: {v}.".format(v=mesh.vertices.shape)
        )
        self._logi(
            "  Bounds: {b}.".format(b=mesh.get_bounds())
        )
        self._mesh = mesh.copy() # copy to make sure given mesh stays untouched

        self._scale_mesh_vertices()
        self.calculated = False

    @property
    def spline(self):
        """
        Returns a copy of current spline. This is to prevent unnecessary
        tangling

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
        Sets spline after doing some tests to determine what kind of situation
        we are currently facing. Checks if spline is naive by comparing control
        points after re-creating naive version of given spline. Here, naive
        means un-deformed from box stage.

        Parameters
        -----------
        spline: SPLINE_TYPES
            New Spline for the next deformation

        Returns
        --------
        None
        """
        # This is also False if no spline was given before
        if self._is_parametric_range_changed_vs_original_spline(spline):
            self._spline = spline.copy()
            self._scale_parametric_dimension_to_hypercube()

        self._logi("Setting Spline.")
        self._logi("Spline Info:")
        self._logi(
            f"  Parametric dimensions: {spline.para_dim}."
        )
        self._logi(
            f"  Parametric dimension before scaling:"
            f" {self._original_spline_ranges}."
        )
        self.calculated = False

    @property
    def is_partial(self):
        """
        Returns is_partial.

        Parameters
        -----------
        None

        Returns
        --------
        self._is_partial: bool
        """
        if self._is_partial:
            self._logw("Partial FFD is currently not supported.")
        return self._is_partial

    @is_partial.setter
    def is_partial(self, is_partial):
        """
        Setter for is_partial. Do you want partial FFD of your mesh?

        Parameters
        -----------
        is_partial: bool

        Returns
        --------
        None
        """
        if is_partial:
            self._logw("Partial FFD is currently not supported.")
        # self._is_partial = is_partial
        # self._logd("This FFD will be:")
        # if is_partial:
        #     self._logw("  Partial!")
        #     self._logw("  Be careful! Else, mesh might tangle!")
        #     self._offset = None
        #     self._scale = None

        # else:
        #     self._logd("  Non-partial!")


    def _scale_parametric_dimension_to_hypercube(self):
        """Scales all knot_vectors of the spline to a range of [0,1]
        """
        self._original_spline_ranges = list()
        if self._is_parametric_room_hypercube(self._spline):
            for _ in range(self.spline.para_dim):
                self._original_spline_ranges.append((0,1))
        else:
            knv = []
            for knot_vector in self._spline.knot_vectors:
                self._original_spline_ranges.append(
                    (knot_vector[0],knot_vector[-1]))
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
        if not type(spline) is Bezier:
            for knot_vector in spline.knot_vectors:
                # check if knot_vectors fist element is 0 and last element is 1
                if not (knot_vector[0] == 0 and knot_vector[-1] == 1):
                    return False
        return True

    def _is_parametric_range_changed_vs_original_spline(
            self,
            spline: SPLINE_TYPES
        ) -> bool:
        """Checks if the parametric range of the spline has a different 
        range in the parametric dimension as the original spline used for this 
        FFD.

        Parameters
        ----------
        spline : SPLINE_TYPES
            New spline for which to check for changes in the parametric 
            dimension

        Returns
        -------
        bool
            True if parametric room is changed or these have not been 
            calculated yet, else False
        """
        if self._original_spline_ranges is None:
            return True
        if not type(spline) is Bezier:
            for knot_vector, orig_parametric_range in zip(
                                            spline.knot_vectors,
                                            self._original_spline_ranges):
                # check if knot_vectors fist element is 0 and last element is 1
                if not (
                    knot_vector[0] == orig_parametric_range[0] and
                    knot_vector[-1] == orig_parametric_range[-1]
                ):
                    return True
        return False
               
            
    def _is_mesh_dimension_same_as_spline_parametric_dimension(self) -> bool:
        """Checks if the mesh and parametric dimension of the spline are the 
        same.

        Returns
        -------
        bool
            True if the dimensions are the same, else False.
        """   
        if self._mesh.vertices.shape[1] == len(self._spline_offset):
            True
        False

    def _scale_mesh_vertices(self):
        """
        Scales the mesh vertices into the dimension of a hypercube and save 
        them in self._q_vertices.
        """        
        self._logd("Fitting mesh into spline's parametric space.")
        if self._is_partial:
            # currently not supported
            # should add functionality here for it
            # define the q_vertices here
            pass
        else:
            self._q_vertices = self._mesh.vertices.copy()
        
        original_mesh_bounds = self._mesh.get_bounds()
        self._mesh_offset = list()
        self._mesh_scale = list()
        for mesh_bound  in original_mesh_bounds.T:
            self._mesh_offset.append(-mesh_bound[0])
            self._mesh_scale.append(1/(mesh_bound[-1]-mesh_bound[0]))

        # transform vertices
        self._q_vertices += self._mesh_offset
        # scale vertices
        scale_matrix = np.eye(len(self._mesh_scale))*self._mesh_scale
        self._q_vertices = self._q_vertices @ scale_matrix
        self._logd("Successfully scaled and transformed mesh vertices!")

    def _deform(self):
        """
        Deform mesh. Partially deform mesh if `is_partial` is True. Meant for
        internal use.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        # no changes since last evaluation/deformation
        if self.calculated:
            return

        if self._mesh is None or self._spline is None:
            raise RuntimeError(
                "Either `mesh` or `spline` is not defined for FFD."
            )

        # One thing to keep in mind is that in this case, user needs to make
        # sure that part of the mesh that they want to deform is positive
        # definite.
        if self.is_partial:
            self._logw("The partial FFD is currently not available.")
            # self._logd("Applying partial FFD...")

            # # Instead of checking for "collision", just try to select vertices.
            # # Note: range is (), not []!
            # v_ind = self._original_mesh.select_vertices(
            #     "xyz_range",
            #     criteria=utils.bounds_to_ranges(
            #         self._naive_spline.knot_vectors_bounds
            #     ),
            # )

            # if len(v_ind) > 0:
            #     self._logd(
            #         "  There are {v}/{vt} vertices to deform.".format(
            #             v=len(v_ind),
            #             vt=len(self._mesh.vertices),
            #         )
            #     )

            #     # Add new axis if current spline maps to higher dim
            #     if (
            #         self._spline.physical_dim != self._mesh.vertices.shape[1]
            #         and self._spline.physical_dim > self._spline.parametric_dim
            #     ):
            #         for _ in range(
            #             self.spline.physical_dim 
            #             - self.spline.parametric_dim
            #         ):
            #             self._mesh.vertices = np.hstack(
            #                 (
            #                     self._original_mesh.vertices,
            #                     np.zeros((self._mesh.vertices.shape[0], 1)),
            #                 )
            #             )

            #     # Here, no need to use q_vertices, since we don't deform
            #     # original
            #     self._mesh.vertices[v_ind] = self._spline.evaluate(
            #         self._original_mesh.vertices[v_ind]
            #     )
            #     self._logd("partial FFD successful.")
            # else:
            #    self._logw(
            #         "Spline does not contain any mesh. Nothing to deform."
            #     )

            return

        self._logd("Applying FFD...")
        # Here, we take _q_vertices, due to possible scale/offset.
        self._mesh.vertices = self._spline.evaluate(
            self._q_vertices
        )
        self._logd("FFD successful.")
        self.calculated = True
    

    @property
    def control_points(self):
        """
        Returns current spline's control points. Can't and don't use this to
        directly manipulate control points: it won't update

        Returns
        --------
        self._spline.control_points: np.ndarray
        """
        return self._spline.control_points

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
            self.control_points[mask] += values

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
#         self._naive_spline.elevate_degree(*args, **kwargs)

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
#         self._naive_spline.insert_knots(*args, **kwargs)

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
#         self._naive_spline.remove_knots(*args, **kwargs)

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
#         self._naive_spline.reduce_degree(*args, **kwargs)

        
    def show(self):
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
        if original_mesh.kind is "volume":
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
                )