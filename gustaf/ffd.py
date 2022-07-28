"""
gustaf/gustaf/ffd.py

Freeform Deformation!


Adaptation of previous implementation in internal python package gustav by
Jaewook Lee. 
"""
from typing import List, Union
import numpy as np
from gustaf import utils
from gustaf.edges import Edges

from gustaf.faces import Faces
from gustaf.spline.base import NURBS, BSpline, Bezier
from gustaf.vertices import Vertices
from gustaf.volumes import Volumes
from gustaf._base import GustavBase


class FFD (GustavBase):

    def __init__(
        self,
        mesh: Union[Edges, Faces, Volumes],
        spline: Union[Bezier, BSpline, NURBS],
        # is_partial: bool=False,
    ):
        """
        FreeFormDeformation. Here, spline is always the boss. Meaning, once
        spline is defined using setter, we always fit mesh into the spline.
        For partial FFD, no one needs to be the boss. If no spline is defined,
        a simple spline wraps given mesh as default.

        Parameters
        ----------
        mesh (Union[Edges, Faces, Volumes]): Base mesh of the FFD
        spline (Union[Bezier, BSpline, NURBS]): Original Spline of the FFD
        is_partial (bool): True if mesh is only partialy deformed (currently 
            not supported)

        Attributes
        ----------
        _mesh: Mesh
        _original_mesh: Mesh
        _q_vertices: np.ndarray
        _spline: BSpline or NURBS
        _naive_spline: BSpline or NURBS
        _offset : (dim,) np.ndarray
        _scale : (dim,) np.ndarray
        _is_partial: bool

        Returns
        -------
        None
        """
        # Use property definitions to store the values
        self._spline = None
        self._mesh = None
        self._spline_range = None
        self._spline_offset = None
        self._mesh_offset = None
        self._mesh_scale = None
        self._q_vertices = None
        self._is_partial = False

        self.calucated = False
        self.mesh = mesh
        print(f"Spline is as follows: {spline.degrees} {spline.knot_vectors} {spline.control_points}")
        self.spline = spline
        self.is_partial = False
        

        # functionality

        self._scale_mesh()


    @property
    def mesh(self,) -> Union[Edges, Faces, Volumes]:
        """Returns copy of current mesh. Before copying, it applies 
        deformation.

        Returns
        -------
        Union[Edges, Faces, Volumes]
            Current Mesh with the deformation according to the current spline.
        """        
        self._deform()
        return self._mesh.copy()

    @mesh.setter
    def mesh(self, mesh: Union[Edges, Faces, Volumes]):
        """
        Sets mesh. If it is first time, the copy of it will be saved as 
        original mesh. If spline is already defined and in transformed status, 
        it applies transformation directly.

        Parameters
        -----------
        mesh (Union[Edges, Faces, Volumes]): Mesh used for the FFD

        Returns
        --------
        None
        """
        needs_recalulating: bool = False
        if self._mesh is not None:
            self._logw("Resetting original Mesh, this is not intended "
            "behaviour. Please create a new FFD object when replacing the mesh"
            ".")
            needs_recalulating = True

        self._logi("Setting mesh.")
        self._logi("Mesh Info:")
        self._logi(
            "  Veritces: {v}.".format(v=mesh.vertices.shape)
        )
        self._logi(
            "  Bounds: {b}.".format(b=mesh.get_bounds())
        )
        self._mesh = mesh.copy() # copy to make sure given mesh stays untouched
        self._original_mesh = self._mesh.copy()
        self._logd("Saved a copy of mesh for book-keeping.")

        if needs_recalulating:
            self._mesh_into_spline()
        self.calucated = False

    @property
    def spline(self):
        """
        Returns a copy of current spline. This is to prevent unncessary
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
    def spline(self, spline: Union[Bezier, BSpline, NURBS]):
        """
        Sets spline after doing some tests to determine what kind of situation
        we are currently facing. Checks if spline is naive by comparing control
        points after re-creating naive version of given spline. Here, naive
        means undeformed from box stage.

        Parameters
        -----------
        spline: Spline

        Returns
        --------
        None
        """
        # needs recalculating only applies if the mesh needs to be rescaled 
        # due to changes in parametric dimension ranges of the spline
        # this does trigger a calculation of the FFD
        needs_recalulating: bool = False

        self._spline = spline.copy()

        if self._spline is None:
            self._calculate_spline_parametric_range()
        elif self._check_spline_ranges_changed(spline):
            # parametric dimension ranges have changed
            self._logw("Parametric dimensions has changed. Rescaling "
                       "the mesh.")
            self._calculate_spline_parametric_range()
            needs_recalulating = True

        self._logi("Setting Spline.")
        self._logi("Spline Info:")
        self._logi(
            f"  Parametric dimensions: {spline.para_dim}."
        )
        self._logi(
            f"  Parametric dimension offsets: {self._spline_offset}."
        )
        self._logi(
            f"  Parametric dimension bounds with offset:"
            f" {self._spline_range}."
        )

        if needs_recalulating:
            self._mesh_into_spline()
        self.calucated = False

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

    def _calculate_spline_parametric_range(self):
        """Calculates the parametric range and needed offset for the given 
        spline.

        For the BSpline and NURBS the parametric dimension is defined by the 
        knot_vectors. For Bezier the parametric dimension is a unit hypercube.

        Parameters
        ----------
        spline : Union[Bezier, BSpline, NURBS]
            Spline for which the parametric dimension is to be calculated.
        """                
        self._spline_offset = list()
        self._spline_range = list()
        if type(self._spline) is Bezier:
            for _ in range(self._spline.para_dim):
                self._spline_offset.append(0)
                self._spline_range.append((0,1))
        else:
            for knot_vector in self._spline.knot_vectors:
                # check if minimum value of the knot_vector is negativ, if yes
                # define offset so that the minimum value is zero
                self._spline_offset.append(np.max([knot_vector[0]*-1,0, 0]))
                # calculate the spline range in this dimension 
                self._spline_range.append(
                    list(map(knot_vector.__getitem__,[0,-1]))
                    +self._spline_offset[-1]
                )
            self._scale_spline()
        print(self._spline.knot_vectors, self._spline._knot_vectors)

    def _check_spline_ranges_changed(
            self,
            spline: Union[Bezier,BSpline, NURBS]
        ) -> bool:
        """Checks if the parametric range of the spline has changed.

        Parameters
        ----------
        spline : Union[Bezier,BSpline, NURBS]
            New spline for which to check the previously calculated ranges 
            against.

        Returns
        -------
        bool
            Whether or not the ranges have changed. True if ranges have 
            changed or check could not be completed.
        """
        if self._spline_range is None or self._spline_range is None:
            self._logw("No previously calculated spline parametric ranges.")
            return True
        if not len(self._spline_offset) == spline.para_dim:
            self._logw("Dimension of the spline has changed.")
            return True
        # if spline is Bezier the parametric dimensions can not change
        if type(spline) is Bezier:
            return False
        
        for knot_vector, offset, range in zip(
                                        spline.knot_vectors,
                                        self._spline_offset,
                                        self._spline_range):
            # check if offset is the same
            # check if the spline range is the same
            if not (
                np.max(knot_vector[0]*-1,0) == offset and 
                np.isclose(np.subtract(range[::-1]),
                           np.subtract(*knot_vector[[-1,0]]+offset))
               ):
                break
        else:
            # Ranges for all parametric dimensions are the same
            return False
        return True
            
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

    def _scale_mesh(self):
        """
        Scales the mesh into the spline's parametric dimension
        """
        if self._mesh is None or self._spline is None:
            self._logw("Either spline or the mesh are not yet correctly "
            "defined, please do this before preceeding.")
            return
        
        # check if the parameteric dimension of the spline is the same as the 
        # dimension of the mesh
        if self._is_mesh_dimension_same_as_spline_parametric_dimension():
            self._logw("The dimensions of the spline and mesh differ. Can not "
            "transform the mesh into the parameteric room of the spline.")
            return

        # scale mesh so that it fits into the spline's parametric dimension
        self._mesh_into_spline()

    def _scale_spline(self):
        """Translates the parametric dimension of the spline into a positiv 
        range.

        Sould only ever called on a BSpline or NURBS.
        """        
        for knot_vector_idx in range(self._spline.para_dim):
            # if lowest value of knot vector is negativ translate vector such
            # that he lowest value is zero
            if self._spline.knot_vectors[knot_vector_idx][0] < 0:    
                self._spline.knot_vectors[knot_vector_idx] = \
                    list(map(self._spline_offset[knot_vector_idx].__add__, 
                         self._spline.knot_vectors[knot_vector_idx]))
        


    def _deform(self, reverse_scale_and_offset=True):
        """
        Deform mesh. Partially deform mesh if `is_partial` is True. Meant for
        internal use.

        Parameters
        -----------
        reverse_scale_and_offset: bool
          (Optional) Default is True. For visualization, 

        Returns
        --------
        None
        """
        # no changes since last evaluation/deformation
        if self.calucated:
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

        # Here, we take q_vertice, due to possible scale/offset.
        print("directly before", self._spline.knot_vectors, self._spline._knot_vectors)
        evaluated_vertices = self._spline.evaluate(
            self._q_vertices
        )
        self._logd("FFD successful.")

        if reverse_scale_and_offset:
            descale_matrix = (np.eye(len(self._mesh_scale))/self._mesh_scale)
            evaluated_vertices -= self._mesh_offset
            self._mesh.vertices = (evaluated_vertices @ descale_matrix)

        self.calucated = True
        



#     @property
#     def control_points(self):
#         """
#         Returns current spline's control points. Can't and don't use this to
#         directly manipulate control points: it won't update

#         Parameters
#         -----------
#         None

#         Returns
#         --------
#         self._spline.control_points: np.ndarray
#         """
#         return self._spline.control_points

#     @control_points.setter
#     def control_points(self, control_points):
#         """
#         Sets control points and deforms mesh.

#         Parameters
#         -----------
#         control_points: np.ndarray

#         Returns
#         --------
#         None
#         """
#         assert self._spline.control_points.shape == control_points.shape,\
#             "Given control points' shape does not match current ones!"

#         self._spline.control_points = control_points.copy()
#         self._logd("Set new control points.")

#     def translate(self, values, mask=None):
#         """
#         Translates control points of given mask.

#         Parameters
#         -----------
#         values: np.ndarray
#           Values to translate. If it is 1D array, it is applied to all masked
#           control points.
#         mask: list-like
#           (Optional) Default is None. If None, it will be applied to all.
#           It can be bool or int.

#         Returns
#         --------
#         deformed_mesh: Mesh
#         """
#         if mask is None:
#             self.control_points += values
#         else:
#             self.control_points[mask] += values

#         return self.mesh

#     def elevate_degree(self, *args, **kwargs):
#         """
#         Wrapper for Spline.elevate_degree

#         Parameters
#         -----------
#         *args:
#         **kwrags:

#         Returns
#         --------
#         None
#         """
#         self._spline.elevate_degree(*args, **kwargs)
#         self._naive_spline.elevate_degree(*args, **kwargs)

#     def insert_knots(self, *args, **kwargs):
#         """
#         Wrapper for Spline.insert_knots

#         Parameters
#         -----------
#         *args:
#         **kwargs:

#         Returns
#         --------
#         None
#         """
#         self._spline.insert_knots(*args, **kwargs)
#         self._naive_spline.insert_knots(*args, **kwargs)

#     def remove_knots(self, *args, **kwargs):
#         """
#         Wrapper for Spline.remove_knots

#         Parameters
#         -----------
#         *args:
#         **kwargs:

#         Returns
#         --------
#         None
#         """
#         self._spline.remove_knots(*args, **kwargs)
#         self._naive_spline.remove_knots(*args, **kwargs)

#     def reduce_degree(self, *args, **kwargs):
#         """
#         Wrapper for Spline.reduce_degree

#         Parameters
#         -----------
#         *args:
#         **kwargs:

#         Returns
#         --------
#         None
#         """
#         self._spline.reduce_degree(*args, **kwargs)
#         self._naive_spline.reduce_degree(*args, **kwargs)

    def _mesh_into_spline(self):
        """
        Scales and transforms the vertices of the original mesh, into the
        parametric dimension of the spline.
        """        
        self._logd("Fitting mesh into spline's parametric space.")
        if self._is_partial:
            # currently not supported
            # shoudl add functionality here for it
            # define the q_vertices here
            pass
        else:
            self._q_vertices = self._original_mesh.vertices.copy()
        
        original_mesh_bounds = self._original_mesh.get_bounds()
        self._mesh_offset = list()
        self._mesh_scale = list()
        for mesh_bound, spline_range  in zip(original_mesh_bounds.T,
                                             self._spline_range):
            self._mesh_offset.append(spline_range[0]-mesh_bound[0])
            self._mesh_scale.append(
                np.subtract(*spline_range[::-1])/
                np.subtract(*mesh_bound[::-1]))
        
        # scale vertices
        scale_matrix = np.eye(len(self._mesh_scale))*self._mesh_scale
        self._q_vertices = self._q_vertices @ scale_matrix
        # transform vertices
        self._q_vertices += self._mesh_offset
        self._logd("Successfully scaled and transformed mesh vertices!")


    # def _make_mesh_positive_definite(self):
    #     """
    #     Checks if mesh vertices are positive definite. If not, apply offset
    #     to make them positive definite. Meant for internal use.

    #     Parameters that are changed: _scale, _offset, _q_vertices

    #     Parameters
    #     -----------
    #     None

    #     Returns
    #     --------
    #     None
    #     """
    #     # TODO: is this fine?
    #     self._scale = None

    #     if self._original_mesh.vertices.min() >= 0.0:
    #         self._logd("Vertices of mesh is positive definite.")
    #         self._offset = None
    #         self._q_vertices = self._mesh.vertices.copy()
    #         return

    #     self._logd("Vertices of mesh is not positive definite.")

    #     self._offset = self._mesh.vertices.min(axis=0)
    #     self._q_vertices = self._mesh.vertices.copy() - self._offset

    #     self._logd("  Applied offset to query vertices.")

#     def _reverse_scale_and_offset(self):
#         """
#         Reverse scale and offset, if they exist.

#         Parameters
#         -----------
#         None

#         Returns
#         --------
#         None
#         """
#         if self._scale is None and self._offset is None:
#             return

#         self._logd("Reversing scale and offset.")

#         if self._scale is not None:
#             # Get current mesh bounds before any manipulation
#             c_bounds = self._mesh.bounds

#             tmp_offset = self._mesh.vertices.min(axis=0)
#             self._mesh.vertices -= tmp_offset
#             self._mesh.vertices /= self._mesh.vertices.max(axis=0)

#             # To allow any kind of scale, we need scale difference of current
#             # /original mesh. Otherwise, mesh will always be 
#             # scaled into the initial bounding box.
#             o_bounds = utils.bounds(self._q_vertices)


#             ob_diff = o_bounds[1] - o_bounds[0]
#             cb_diff = c_bounds[1] - c_bounds[0]

#             # In case spline maps to higher degree, add some zeros.
#             s_dim_diff = int(
#                 self._spline.physical_dim - self._spline.parametric_dim
#             )
#             if s_dim_diff < 0:
#                 ob_diff = ob_diff.tolist()
#                 for _ in range(s_dim_diff):
#                     ob_diff.append(0)
#                 ob_diff = np.asarray(ob_diff.tolist() + [0]) 

#             spline_scale_factor = cb_diff / ob_diff

#             # Sometimes this scale factor can have inf or 0.
#             # inf:
#             #  spline maps to higher dim. There's 0 in o_bounds difference.
#             #  -> replace inf with c_bounds diff
#             # 0:
#             #  spline maps to a somewhere co-planar
#             #  -> replace 0 with 1
#             tolerance = 1e-10
#             zero_mask = abs(cb_diff) < tolerance
#             if zero_mask.any():
#                 spline_scale_factor[zero_mask] = 1.0

#             inf_mask = abs(ob_diff) < tolerance
#             if inf_mask.any():
#                 spline_scale_factor[inf_mask] = cb_diff[inf_mask]

#             # If spline maps to higher dim.
#             if self._spline.physical_dim > self._spline.parametric_dim:
#                 scale = np.asarray(self._scale.tolist() + [1.0])

#             else: 
#                 scale = self._scale

#             self._mesh.vertices *= scale * spline_scale_factor
#             self._mesh.vertices += tmp_offset

#         if self._offset is not None:
#             # If spline maps to higher dim.
#             if self._spline.physical_dim > self._spline.parametric_dim:
#                 offset = np.asarray(self._offset.tolist() + [0.0])

#             else: 
#                 offset = self._offset

#             self._mesh.vertices += offset
        
    # def show(self):
    #     """
    #     Visualize. Shows naive state and current state, based on params.

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     -------
    #     None
    #     """
    #     import vedo

    #     # Same style as `Spline.show()`

    #     # We always need naive as it is
    #     things_to_show_naive = []
    #     spline_obj_n = self._spline.show(offscreen=True)
    #     spline_obj_n. # wanna see through the spline.
    #     things_to_show_naive.extend(spline_obj_n)

    #     mesh_obj_n = self._original_mesh.vedo_mesh # vedo.Mesh or vedo.UGrid
    #     if self._q_vertices is not None:
    #         qv = self._q_vertices.copy()

    #         if self._q_vertices.shape[1] < 3:
    #             qv = np.hstack(
    #                 (qv, np.zeros((len(qv), int(3 - qv.shape[1]))))
    #             )
                
    #         mesh_obj_n.points(qv)

    #     if self._mesh.elements is not None: # In this case, it is vedo.UGrid
    #         mesh_obj_n = mesh_obj_n.tomesh(shrink=0.8).color("hotpink")
    #     else:
    #         mesh_obj_n.wireframe(True)
    #     things_to_show_naive.append(mesh_obj_n)

    #     if (self._scale is None and self._offset is None) or self.is_partial:
    #         things_to_show_current = []

    #         spline_obj_c = self._spline.show(offscreen=True)
    #         spline_obj_c[0].alpha(.5) # wanna see through the spline.
    #         things_to_show_current.extend(spline_obj_c)
    
    #         mesh_obj = self.mesh.vedo_mesh # either vedo.Mesh or vedo.UGrid
    #         if self._mesh.elements is not None: # In this case, it is vedo.UGrid
    #             mesh_obj = mesh_obj.tomesh(shrink=0.9).color("hotpink")

    #         else:
    #             mesh_obj.wireframe(True)

    #         things_to_show_current.append(mesh_obj)

    #         plt = vedo.Plotter(N=2, sharecam=False)
    #         plt.show(things_to_show_naive, "Naive State", at=0)
    #         plt.show(
    #             things_to_show_current,
    #             "Current State",
    #             at=1,
    #             interactive=True
    #         ).close()

    #     else: # scale or offset and not partial
    #         om = self._original_mesh.vedo_mesh
    #         cm = self.mesh.vedo_mesh

    #         things_to_show_current_no_reverse = []
    #         spline_obj_cnr = self._spline.show(offscreen=True)
    #         spline_obj_cnr[0].alpha(.5)
    #         things_to_show_current_no_reverse.extend(spline_obj_cnr)
    #         # Deform again, but don't reverse
    #         # TODO: ^ more efficiently
    #         self._deform(reverse_scale_and_offset=False)
    #         cnrm = self._mesh.vedo_mesh

    #         # If it is volume mesh, shrink it
    #         if self._original_mesh.elements is not None:
    #             om = om.tomesh(shrink=0.9).color("hotpink")
    #             cm = cm.tomesh(shrink=0.9).color("hotpink")
    #             cnrm = cnrm.tomesh(shrink=0.9).color("hotpink")

    #         else:
    #             om.wireframe()
    #             cm.wireframe()
    #             cnrm.wireframe()

    #         things_to_show_current_no_reverse.append(cnrm)
            
    #         plt = vedo.Plotter(N=4, sharecam=False)
    #         plt.show(om, "Original Mesh", at=0)
    #         plt.show(cm, "Current Mesh", at=1)
    #         plt.show(things_to_show_naive, "Naive Spline Bounds", at=2)
    #         plt.show(
    #             things_to_show_current_no_reverse,
    #             "Current Spline Bounds",
    #             at=3,
    #             interactive=True
    #         ).close()
