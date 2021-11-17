"""gustav/gustav/mesh.py

Mesh.
"""
import logging
import os

import numpy as np

from gustav import utils
from gustav.utils.errors import InvalidSetterCallError

class Mesh:

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

    def _logd(self, *log):
        """
        Debug logger wrapper for Mesh.

        Parameters
        -----------
        *log: *str

        Returns
        --------
        None
        """
        utils.log._debug("Mesh -", *log)

    def _logi(self, *log):
        """
        Info logger wrapper for Mesh.

        Parameters
        -----------
        *log: *str

        Returns
        --------
        None
        """
        utils.log._info("Mesh -", *log)

    def _logw(self, *log):
        """
        Warning logger wrapper for Mesh.

        Parameters
        -----------
        *log: *str

        Returns
        --------
        None
        """
        utils.log._warning("Mesh -", *log)

    def _get_property(self, key):
        """
        Checks if property is defined with given key.

        Parameters
        -----------
        key: str

        Returns
        --------
        property: obj or None
        """
        return utils._dict._get_property(
            self._properties,
            key,
            "Mesh",
        )

    def _update_property(self, key, value):
        """
        Updates property with given value.

        Parameters
        -----------
        key: str
        value: obj

        Returns
        --------
        None
        """
        utils._dict._update_property(
            self._properties,
            key,
            value,
            "Mesh",
        )

    def _get_cached(self, key):
        """
        Checks if obj is cached with given key.

        Parameters
        -----------
        key: str

        Returns
        --------
        cached_property: obj or None
        """
        return utils._dict._get_cached(
            self._cached,
            key,
            "Mesh",
        )

    def _update_cached(self, key, value):
        """
        Updates cached dict with given key and value.

        Parameters
        -----------
        key: str
        value: obj

        Returns
        --------
        None
        """
        utils._dict._update_cached(
            self._cached,
            key,
            value,
            "Mesh",
        )

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
        if len(self._properties) >= 2:

        vertices = utils.arr.make_c_contiguous(vertices, np.float64)
        self._update_property("vertices", vertices)

        if self.vertices is not None and len(self._properties) == 1:
            self._update_cached("whatami", "points")

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
        if len(self._properties) <= 1:
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
                return self.edges

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

        edges = utils.arr.make_c_contiguous(edges, np.int32)
        self._update_properties("edges", edges)

        if self.edges is not None and len(self._properties) == 2:
            self._update_cached("whatami", "line")

        elif self.edges is None and len(self._properties) == 1:
            self._update_cached("whatami", "points")

        elif len(self._properties) == 0:
            self._update_cached("whatami", None)

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

        # If "I" am points or nothing, return None
        if len(self._properties) <= 1:
            return None

        # if "I" am tet or hexa, return either cached or generated faces
        elements = self._get_property("elements")
        if elements is not None:
            faces = self._get_cached("faces")
            if faces is not None:
                return faces

            if self.whatami == "tet" and elements.shape[1] == 4:
                new_faces = utils.connec.tet_to_tri(elements)
                self._update_cached("faces") = new_faces
                return self.faces

            elif self.whatami == "hexa" and elements.shape[1] == 6:
                new_faces = utils.connec.hexa_to_quad(elements)
                self._update_cached("faces") = new_faces
                return self.faces

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

        faces = utils.arr.make_c_contiguous(faces, np.int32)
        self._update_properties("faces", faces)

        if self.faces is not None and len(self._properties) == 2:
            if self.faces.shape[1] == 3:
                self._update_cached("whatami", "tri")

            elif self.faces.shape[1] == 4:
                self._update_cached("whatami", "quad")

        elif self.faces is None and len(self._properties) == 1:
            self._update_cached("whatami", "points")

        elif len(self._properties) == 0:
            self._update_cached("whatami", None)

        else:
            raise RuntimeError("Something went wrong during setting faces")

    @property
    def elements(self):
        """
        Returns elements

        Parameters
        -----------
        None

        Returns
        --------
        elements: (n, d)_ np.ndarray
        """
        # If element is a property, return it
        elements = self._get_property("elements")
        if elements is not None:
            return elements

        # If "I" am points or nothing, return None
        if len(self._properties) <= 1:
            return None

        else:
            raise RuntimeError(
                "Something went wrong while trying to get faces"
            )

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

        elements = utils.arr.make_c_contiguous(elements, np.int32)
        self._update_properties("faces", elements)

        if self.elements is not None and len(self._properties) == 2:
            if self.faces.shape[1] == 4:
                self._update_cached("whatami", "tet")

            elif self.faces.shape[1] == 6:
                self._update_cached("whatami", "hexa")

        elif self.elements is None and len(self._properties) == 1:
            self._update_cached("whatami", "points")

        elif len(self._properties) == 0:
            self._update_cached("whatami", None)

        else:
            raise RuntimeError("Something went wrong during setting faces")


