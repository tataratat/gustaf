"""gustav/gustav/mesh.py

Mesh.
"""
import logging
import os

import numpy as np

from gustav import utils
from gustav._abstract_base import AB
from gustav.utils.errors import InvalidSetterCallError

class Mesh(AB):

    def __init__(
            self,
            vertices=None,
            faces=None,
            elements=None,
            edges=None
    ):
        """
        Mesh.
        It can be tri, quad, tet, and hexa.
        Even lines and points.
        At any time, it can only have max 2 properties including vertices.
        Some connectivity information is "cached" and this will be eraised
        each time one of the connectivity properties is newly set.

        Parameters
        -----------
        vertices: (n,m) np.ndarray
          float. `m` is either 2 or 3.
        faces: (k,l) np.ndarray
          int. `l` is either 3 or 4.
          This is also often referred as cells.
        elements: (p,q) np.ndarray
          int. `q` is either 4 or 8.
        edges: (u, 2) np.ndarray
          int.

        Returns
        --------
        None

        Attributes
        -----------
        vertices: np.ndarray
        faces: np.ndarray
        edges: np.ndarray
        unique_edges: np.ndarray
        outlines: list
        boundary_conditions: dict
        elements: np.ndarray
        surfaces: list
        faces_center: np.ndarray
        unique_faces: np.ndarray
        """
        self._logd("Init")

        self._properties = dict()
        self._cached = dict()

        self.vertices = vertices
        self.faces = faces
        self.elements = elements
        self.edges = edges

    @property
    def whatami(self):
        """
        Sends `Mesh` on a journey to fetch an answer to a deep question:
        whatami?

        Parameters
        -----------
        None

        Returns
        --------
        whatami: str
        """
        # whatami is not a property, since mesh only has 2 properties.
        whatami = self._get_cached("whatami")
        if whatami is None:
            return "Nothing"

        else:
            return whatami

    @property
    def vertices(self):
        """
        Returns vertices.

        Parameters
        -----------
        None

        Returns
        --------
        vertices: (n, d) np.ndarray
        """
        return self._get_property("vertices")

    @vertices.setter
    def vertices(self, vertices):
        """
        Sets vertices.

        Parameters
        -----------
        vertices: (n, d) np.ndarray

        Returns
        --------
        None
        """
        if (
            len(self._properties) == 2
            and self._get_property("vertices") is None
        ):
            raise InvalidSetterCallError(self)

        self._logd("Setting vertices.")
        vertices = utils.arr.make_c_contiguous(vertices, np.float64)
        self._update_property("vertices", vertices)

        new_vertices = self.vertices
        if new_vertices is not None and len(self._properties) == 1:
            self._update_cached("whatami", "points")

        elif new_vertices is None:
            self._update_cached("whatami", None)

    @property
    def edges(self):
        """
        Returns edges.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        # If edges is a property, return it
        edges = self._get_property("edges")
        if edges is not None:
            return edges

        # If "I" am points or nothing, return None
        whatami = self.whatami
        if whatami == "points" or whatami == "Nothing":
            return None

        # By this time, "I" am not a line nor points.
        # Maybe it is cached
        edges = self._get_cached("edges")
        if edges is not None:
            return edges

        # So, not cached, let's compute
        faces = self.faces
        if faces is not None:
            if faces.shape[1] == 3 or faces.shape[1] == 4:
                new_edges = utils.connec.faces_to_edges(faces)
                self._update_cached("edges", new_edges)
                return new_edges

        # If function hasn't ended by now, something went wrong.
        raise RuntimeError(
            "Something went wrong while trying to get edges"
        )

    @edges.setter
    def edges(self, edges):
        """
        Set edges.
        If this setter lets you to set edges, this mesh will be a line.

        Parameters
        -----------
        edges: (n, 2) np.ndarray

        Returns
        --------
        None
        """
        if len(self._properties) >= 2:
            raise InvalidSetterCallError(self)

        # If there are 1 property and it is not vertices, error!
        if len(self._properties) == 1 and self.vertices is None:
            raise InvalidSetterCallError(self)

        self._logd("Setting edges.")
        edges = utils.arr.make_c_contiguous(edges, np.int32)
        self._update_property("edges", edges)
        self._clear_cached()

        # This mesh will be one of the following: {Nothing, points, line}
        new_edges = self.edges
        if new_edges is not None and len(self._properties) == 2:
            self._update_cached("whatami", "line")
            return None

        elif new_edges is None and len(self._properties) == 1:
            self._update_cached("whatami", "points")
            return None

        elif len(self._properties) == 0:
            self._update_cached("whatami", None)
            return None

        else:
            raise RuntimeError(
                "Something went wrong while setting edges."
            )

    @property
    def faces(self):
        """
        Returns faces.

        Parameters
        -----------
        None

        Returns
        --------
        faces: (n, d) np.ndarray
        """
        # If faces is a property, return it
        faces = self._get_property("faces")
        if faces is not None:
            return faces

        whatami = self.whatami
        if (
            whatami == "points"
            or whatami == "Nothing"
            or whatami == "line"
        ):
            return None

        # if "I" am tet or hexa, return either cached or generated faces
        elements = self.elements
        if elements is not None:
            faces = self._get_cached("faces")
            if faces is not None:
                return faces

            if self.whatami == "tet" and elements.shape[1] == 4:
                new_faces = utils.connec.tet_to_tri(elements)
                self._update_cached("faces", new_faces)
                return new_faces 

            elif self.whatami == "hexa" and elements.shape[1] == 6:
                new_faces = utils.connec.hexa_to_quad(elements)
                self._update_cached("faces", new_faces)
                return new_faces

        else:
            raise RuntimeError(
                "Something went wrong while trying to get faces"
            )

    @faces.setter
    def faces(self, faces):
        """
        Sets faces.

        Parameters
        -----------
        faces: (n, d) np.ndarray

        Returns
        --------
        None
        """
        if len(self._properties) >= 2:
            raise InvalidSetterCallError(self)

        # If there are 1 property and it is not vertices, error!
        if len(self._properties) == 1 and self.vertices is None:
            raise InvalidSetterCallError(self)

        self._logd("Setting faces.")
        faces = utils.arr.make_c_contiguous(faces, np.int32)
        self._update_property("faces", faces)
        self._clear_cached()

        # Update whatami
        new_faces = self.faces
        if new_faces is not None and len(self._properties) == 2:
            if new_faces.shape[1] == 3:
                self._update_cached("whatami", "tri")
                return None

            elif new_faces.shape[1] == 4:
                self._update_cached("whatami", "quad")
                return None

        elif new_faces is None and len(self._properties) == 1:
            self._update_cached("whatami", "points")
            return None

        elif len(self._properties) == 0:
            self._update_cached("whatami", None)
            return None

        else:
            raise RuntimeError("Something went wrong during setting faces")

    @property
    def elements(self):
        """
        Returns elements.

        Parameters
        -----------
        None

        Returns
        --------
        elements: (n, d)_ np.ndarray
        """
        # elements is either property or nothing.
        return self._get_property("elements")

    @elements.setter
    def elements(self, elements):
        """
        Sets elements.

        Parameters
        -----------
        elements: (n, d) np.ndarray

        Returns
        --------
        None
        """
        if len(self._properties) >= 2:
            raise InvalidSetterCallError(self)

        # If there are 1 property and it is not vertices, error!
        if len(self._properties) == 1 and self.vertices is None:
            raise InvalidSetterCallError(self)

        self._logd("Setting elements")
        elements = utils.arr.make_c_contiguous(elements, np.int32)
        self._update_property("elements", elements)
        self._clear_cached()

        # Update whatami
        new_elements = self.elements
        if new_elements is not None and len(self._properties) == 2:
            if new_elements.shape[1] == 4:
                self._update_cached("whatami", "tet")
                return None

            elif new_elements.shape[1] == 6:
                self._update_cached("whatami", "hexa")
                return None

        elif new_elements is None and len(self._properties) == 1:
            self._update_cached("whatami", "points")
            return None

        elif len(self._properties) == 0:
            self._update_cached("whatami", None)
            return None

        else:
            raise RuntimeError("Something went wrong during setting faces")

    @property
    def sorted_edges(self):
        """
        Returns sorted edges.

        Parameters
        -----------
        None

        Returns
        --------
        sorted_edges: (n, 2) np.ndarray
        """
        sorted_edges = self._get_cached("sorted_edges")
        if sorted_edges is not None:
            return sorted_edges

        # If edges is not defined, return 
        sorted_edges = self.edges.copy() if self.edges is not None else None
        if sorted_edges is None:
            return None

        # Sort, and save
        sorted_edges.sort(axis=1)
        self._update_cached("sorted_edges", sorted_edges)

        return sorted_edges

    @property
    def unique_edges(self):
        """
        Returns unique edges.
        They are sorted along axis=1.
        If you want unsorted, use:
        mesh.edges[mesh.unique_edge_ids]

        Parameters
        -----------
        None

        Returns
        --------
        unique_edges: (n, 2) np.ndarray
        """
        unique_edges = self._get_cached("unique_edges")
        if unique_edges is not None:
            return unique_edges

        sorted_edges = self.sorted_edges
        if sorted_edges is None:
            return None

        unique_stuff = utils.arr.unique_rows(
            sorted_edges,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            dtype_name="int32",
        )
        # Unpack
        unique_edges = unique_stuff[0]
        ue_ids = unique_stuff[1]
        ue_inverses = unique_stuff[2]
        ue_counts = unique_stuff[3]

        # Save appropriate info
        self._update_cached("unique_edges", unique_edges) # sorted!
        self._update_cached("unique_edge_ids", ue_ids)
        self._update_cached("outlines", ue_indices[ue_counts == 1])
        self._update_cached("unique_edge_inverses", ue_inverses)

        return unique_edges

    @property
    def unique_edge_ids(self):
        """
        Returns ids of unique edges


        Parameters
        -----------
        None

        Returns
        --------
        unique_edge_ids: (n,) np.ndarray
        """
        unique_edge_ids = self._get_cached("unique_edge_ids")
        if unique_edge_ids is not None:
            return unique_edge_ids

        edges = self.edges
        if edges is None:
            return None

        _ = self.unique_edges # Should compute unique_edge_ids and save it

        return self.unique_edge_ids

    @property
    def outlines(self):
        """
        Returns ids of outline edges.
        Get outline edges:
          mesh.edges[mesh.outlines]

        Parameters
        -----------
        None

        Returns
        --------
        outlines: (n,) np.ndarray
        """
        outlines = self._get_cached("outlines")
        if outlines is not None:
            return unique_edge_ids

        edges = self.edges
        if edges is None:
            return None

        _ = self.unique_edges # Should compute outlines and save it

        return self.outlines

    @property
    def unique_edge_inverses(self):
        """
        Returns ids that can be used to reconstruct sorted_edges with
        unique_edges.

        Good to know:
          mesh.sorted_edges == mesh.unique_edges[mesh.unique_edge_inverses]

        Parameters
        -----------
        None

        Returns
        --------
        unique_edge_inverses: (n,) np.ndarray
        """
        unique_edge_inverses = self._get_cached("unique_edge_inverses")
        if unique_edge_inverses is not None:
            return unique_edge_inverses

        edges = self.edges
        if edges is None:
            return None

        _ = self.unique_edges # Should compute unique_edge_inverses and save it

        return self.unique_edge_inverses

    @property
    def sorted_faces(self):
        """
        Returns sorted faces.

        Parameters
        -----------
        None

        Returns
        --------
        sorted_faces: (n, 3) or (n, 4) np.ndarray
        """
        sorted_edges = self._get_cached("sorted_faces")
        if sorted_faces is not None:
            return sorted_faces

        # If edges is not defined, return 
        sorted_faces = self.faces.copy() if self.faces is not None else None
        if sorted_faces is None:
            return None

        # Sort, and save
        sorted_edges.sort(axis=1)
        self._update_cached("sorted_faces", sorted_faces)

        return sorted_faces

    @property
    def unique_faces(self):
        """
        Returns unique faces.
        They are sorted along axis=1.
        If you want unsorted, use:
        mesh.faces[mesh.unique_face_ids]

        Parameters
        -----------
        None

        Returns
        --------
        unique_faces: (n, 3) or (n, 4) np.ndarray
        """
        unique_faces = self._get_cached("unique_faces")
        if unique_faces is not None:
            return unique_faces

        sorted_faces = self.sorted_faces
        if sorted_faces is None:
            return None

        unique_stuff = utils.arr.unique_rows(
            sorted_faces,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            dtype_name="int32",
        )
        # Unpack
        unique_faces = unique_stuff[0]
        ue_ids = unique_stuff[1]
        ue_inverses = unique_stuff[2]
        ue_counts = unique_stuff[3]

        # Save appropriate info
        self._update_cached("unique_faces", unique_edges) # sorted!
        self._update_cached("unique_face_ids", ue_ids)
        self._update_cached("surfaces", ue_indices[ue_counts == 1])
        self._update_cached("unique_face_inverses", ue_inverses)

        return unique_faces

    @property
    def unique_face_ids(self):
        """
        Returns ids of unique faces


        Parameters
        -----------
        None

        Returns
        --------
        unique_face_ids: (n,) np.ndarray
        """
        unique_face_ids = self._get_cached("unique_face_ids")
        if unique_face_ids is not None:
            return unique_face_ids

        faces = self.faces
        if faces is None:
            return None

        _ = self.unique_faces # Should compute unique_edge_ids and save it

        return self.unique_face_ids

    @property
    def surfaces(self):
        """
        Returns ids of surfaces.
        Get outer faces:
          mesh.faces[mesh.surfaces]

        Parameters
        -----------
        None

        Returns
        --------
        surfaces: (n,) np.ndarray
        """
        surfaces = self._get_cached("surfaces")
        if surfaces is not None:
            return surfaces

        faces = self.faces
        if faces is None:
            return None

        _ = self.unique_faces # Should compute surfaces and save it

        return self.surfaces

    @property
    def unique_face_inverses(self):
        """
        Returns ids that can be used to reconstruct sorted_faces with
        unique_faces.

        Good to know:
          mesh.sorted_faces == mesh.unique_faces[mesh.unique_face_inverses]

        Parameters
        -----------
        None

        Returns
        --------
        unique_face_inverses: (n,) np.ndarray
        """
        unique_face_inverses = self._get_cached("unique_face_inverses")
        if unique_face_inverses is not None:
            return unique_face_inverses

        faces = self.faces
        if faces is None:
            return None

        _ = self.unique_faces # Should compute unique_face_inverses and save it

        return self.unique_face_inverses
