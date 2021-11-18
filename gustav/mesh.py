"""gustav/gustav/mesh.py

Mesh.
"""
import logging
import os

import numpy as np

from gustav import utils
from gustav import settings
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
        On this journey, it also finds its "kind".
        {"Nothing", "points", "line", ""}

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
            self._update_cached("kind", None)
            return "Nothing"

        else:
            if whatami == "points":
                self._update_cached("kind", "points")

            elif whatami == "line":
                self._udpate_cached("kind", "line")

            elif whatami == "tri" or whatami == "quad":
                self._update_cached("kind", "surface")

            elif whatami == "tet" or whatami == "hexa":
                self._update_cached("kind", "volume")

            return whatami

    @property
    def kind(self):
        """
        Returns its kind. It is one of the followings:
        {"Nothing", "points", "line", "surface", "volume"}

        Parameters
        -----------
        None

        Returns
        --------
        kind: str
        """
        _ = self.whatami # cheap operation
        kind = self._get_cached("kind")
        if kind is None:
            return "Nothing"

        return kind

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

    def _connectivity(self, copy=True):
        """
        Smart copied connectivity returner.

        Parameters
        -----------
        return_kind: bool
        copy: bool
          Default is True

        Returns
        --------
        connectivity: (n, 2) or (n, 3) or (n, 4) or (n, 6) np.ndarray
        kind: str
        """
        self._logd("Copying connectivity info")
        kind = self.kind
        if kind == "points" or whatami == "Nothing":
            connectivity = None

        elif kind == "line":
            connectivity = self.edges.copy()

        elif kind == "surface":
            connectivity = self.faces.copy()

        elif kind == "volume":
            connectivity = self.elements.copy()

        else:
            raise RuntimeError(
                "Something went wrong during `_connectivity()`, "
                + "because `kind` is False."
            )

        return connectivity

    def _reset_connectivity(self, connectivity, inplace=True):
        """
        Smart connectivity re-setter.
        Only works if somesort of connectivity exists.

        Parameters
        -----------
        connectivity: 
        """
        self._logd("Resetting connectivity")
        kind = self.kind
        if kind == "points" or kind == "Nothing":
            raise ValueError(
                "Sorry, I can't reset connectivity that does not exist."
            )

        if inplace:
            if kind == "line":
                self.edges = connectivity
            elif kind == "surface":
                self.faces = connectivity
            elif kind == "volume":
                self.elements = connectivity

            return None

        else:
            new_mesh = Mesh(vertices=self.vertices.copy())
            if kind == "line":
                new_mesh.edges = connectivity
            elif kind == "surface":
                new_mesh.faces = connectivity
            elif kind == "volume":
                new_mesh.elements = connectivity

            return new_mesh

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

    @property
    def bounds(self,):
        """
        Returns bounds of the vertices.
        Not saved.

        Parameters
        -----------
        None

        Returns
        --------
        bounds: (2, d) np.ndarray
        """
        return utils.arr.bounds(self.vertices)

    @property
    def bounding_box_center(self):
        """
        Returns center of the bounding box.
        Not saved.

        Parameters
        -----------
        None

        Returns
        --------
        bounding_box_center: (n) np.ndarray
        """
        return utils.arr.bounding_box_center(self.ver)

    def bounding_box(self):
        """
        Returns bounding box in corresponding dimension.
        note: use create
        """
        pass

    def bounding_sphere(self):
        """
        Returns bounding sphere in corresponding dimension
        note: use create
        """
        pass

    @property
    def bounds_norm(self):
        pass

    @property
    def bounding_box_center(self):
        pass

    def update_vertices(self, mask, inverse=None, inplace=True):
        """
        Update vertices with a mask.
        Adapted from `github.com/mikedh/trimesh`.
        Inplace operation.

        Parameters
        -----------
        mask: (len(self.vertices)) np.ndarray
          bool or int
        inverse: (len(self.vertices)) np.ndarray
          int
        inplace: bool

        Returns
        --------
        new_mesh: Mesh
          Only returned if inplaces=False
        """
        vertices = self.vertices
        if vertices is None:
            return None

        vertices = vertices.copy()

        # make sure mask is a numpy array
        mask = np.asanyarray(mask)

        if (
            (mask.dtype.name == 'bool' and mask.all())
            or len(mask) == 0
        ):
            # mask doesn't remove any vertices so exit early
            return None

        # create the inverse mask if not passed
        if inverse is None:
            inverse = np.zeros(len(vertices), dtype=np.int64)
            if mask.dtype.kind == 'b':
                inverse[mask] = np.arange(mask.sum())
            elif mask.dtype.kind == 'i':
                inverse[mask] = np.arange(len(mask))
            else:
                inverse = None

        # re-index connectivity from inverse
        # TODO: preserve BC, maybe. Easier way is to destory it
        whatami = self.whatami
        connectivity = None
        if inverse is not None and whatami != "points":
            connectivity = self._connectivity(copy=True)
            connectivity = inverse[connectivity.reshape(-1)].reshape(
                (-1, connectivity.shape[1])
            )

        # apply the mask
        vertices = vertices[mask]

        # update
        if inplace:
            self.vertices = vertices
            if connectivity is not None:
                self._reset_connectivity(connectivity, inplace=inplace) # True
                return None

        else:
            if connectivity is None:
                return Mesh(vertices=vertices)

            else:
                new_mesh = self.copy()

                return new_mesh._reset_connectivity(
                    connectivity,
                    inplace=inplace, # False
                )

    def select_vertices(
            self,
            method,
            **kwargs
    ):
        pass

    def remove_vertices(self, ids, inplace=True):
        """
        Given ids of vertices, remove them.
        Similar to update_vertices, but inverted version of it.

        Parameters
        -----------
        ids: (n,) np.ndarray
        inplace: bool

        Returns
        --------
        new_mesh: Mesh
          iff `inplace=False`.
        """
        # Make mask
        mask = np.ones(len(self.vertices), dtype=bool)
        mask[ids] = False

        return self.update_vertices(mask, inplace=inplace)

    def remove_unreferenced_vertices(self, inplace=True):
        """
        Remove all the vertices that aren't referenced by connectivity.
        Adapted from `github.com/mikedh/trimesh`.

        Parameters
        -----------
        inplace: bool

        Returns
        --------
        new_mesh: Mesh
         iff `inplace=True`
        """
        kind = self.kind
        # Return if there's no connectivity
        if kind == "points" or kind == "Nothing":
            return None

        referenced = np.zeros(len(self.vertices), dtype=bool)

        connectivity = self._connectivity(copy=False)
        referenced[connectivity] = True

        inverse = np.zeros(len(self.vertices), dtype=np.int64)
        inverse[referenced] = np.arange(referenced.sum())

        return self.update_vertices(
            mask=referenced,
            inverse=inverse,
            inplace=inplace,
        )

    def merge_vertices(self, tolerance=settings.TOLERANCE, inplace=True):
        """
        """
        pass

    def update_connectivity():
        """
        """
        pass

    def copy(self):
        """
        """
        new_mesh = Mesh()
        new_mesh._properties = copy.deepcopy(self._properties)
        new_mesh._cached = copy.deepcopy(self._cached)
        return new_mesh
