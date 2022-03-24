"""gustaf/gustaf/edges.py

Edges. Also known as 
"""

import numpy as np

from gustaf import settings
from gustaf import utils
from gustaf.vertices import Vertices

class Edges(Vertices):

    kind = "edge"

    __slots__ = [
        "edges",
        "edges_sorted",
        "edges_unique",
        "edges_unique_id",
        "edges_unique_inverse",
        "edges_unique_count",
        "outlines"
    ]

    def __init__(
            self,
            vertices=None,
            edges=None,
            elements=None,
            process=False,
    ):
        if vertices is not None:
            self.vertices = utils.arr.make_c_contiguous(
                vertices,
                settings.FLOAT_DTYPE
            )

        if edges is not None:
            self.edges = utils.arr.make_c_contiguous(edges, settings.INT_DTYPE)
        elif elements is not None:
            self.edges = utils.arr.make_c_contiguous(
                elements,
                settings.INT_DTYPE
            )

        self.whatami = "edges"
        self.vis_dict = dict()

        self.process(everything=process)

    def process(
            self,
            edges_sorted=False,
            edges_unique=False,
            edges_unique_id=False,
            edges_unique_inverse=False,
            edges_unique_count=False,
            everything=False,
    ):
        pass

    def get_edges(self):
        """
        Returns edges. If edges is not its original property, it tries to
        compute it based on existing elements.

        Parameters
        -----------
        None

        Returns
        --------
        edges: (n, 2) np.ndarray
        """

        if self.kind == "edge":
            self.edges = utils.arr.make_c_contiguous(
                self.edges,
                settings.INT_DTYPE,
            )

            return self.edges

        if hasattr(self, "volumes") or hasattr(self, "faces"):
            self.edges = utils.connec.faces_to_edges(self.get_faces())
            return self.edges

        # Shouldn't reach this point, but in case it does
        self._logd("cannot compute/return edges.")
        return None

    def get_edges_sorted(self):
        """
        Sort edges along axis=1.

        Parameters
        -----------
        None

        Returns
        --------
        edges_sorted: (n_edges, 2) np.ndarray
        """
        self.edges_sorted = self.get_edges().copy()
        self.edges_sorted.sort(axis=1)

        return self.edges_sorted

    def get_edges_unique(self):
        """
        Returns unique edges.

        Parameters
        -----------
        None

        Returns
        --------
        edges_unique: (n, 2) np.ndarray
        """
        unique_stuff = utils.arr.unique_rows(
            self.get_edges_sorted(),
            return_index=True,
            return_inverse=True,
            return_counts=True,
            dtype_name=settings.INT_DTYPE,
        )

        # unpack
        self.edges_unique = unique_stuff[0]
        self.edges_unique_id = unique_stuff[1].astype(settings.INT_DTYPE)
        self.edges_unique_inverse = unique_stuff[2].astype(settings.INT_DTYPE)
        self.edges_unique_count = unique_stuff[3].astype(settings.INT_DTYPE)
        self.outlines = self.edges_unique_id[self.edges_unique_count == 1]

        return self.edges_unique

    def get_edges_unique_id(self):
        """
        Returns ids of unique edges.

        Parameters
        -----------
        None

        Returns
        --------
        edges_unique_id: (n,) np.ndarray
        """
        _ = self.get_edges_unique()

        return self.edges_unique_id

    def get_edges_unique_inverse(self):
        """
        Returns ids that can be used to reconstruct sorted edges with unique
        edges.

        Good to know:
          mesh.edges_sorted == mesh.unique_edges[mesh.edges_unique_inverse]

        Parameters
        -----------
        None

        Returns
        --------
        edges_unique_inverse: (len(self.edges),) np.ndarray
        """
        _ = self.get_edges_unique()

        return self.edges_unique_inverse

    def get_outlines(self):
        """
        Returns indices of very unique edges: edges that appear only once.
        For well constructed edges, this can be considered as outlines.

        Parameters
        -----------
        None

        Returns
        --------
        outlines: (m,) np.ndarray
        """
        _ = self.get_edges_unique()

        return self.outlines

    def update_elements(self, mask, inplace=True):
        """
        Similar to update_vertices, but for elements.

        Parameters
        -----------
        inplace: bool

        Returns
        --------
        new_self: type(self)
          iff inplace=False
        """
        new_elements = new_elements[mask]
        if inplace:
            self.elements(new_elements).remove_unreferrenced_vertices(
                inplace=True
            )
            return None

        else:
            return type(self)(
                vertices=self.vertices,
                elements=new_elements
            ).remove_unreferrenced_vertices(inplace=False)

    def update_edges(self, *args, **kwargs):
        """
        Alias to update_elements.
        """
        return self.update_elements(*args, **kwargs)

    def subdivide(self):
        """
        Subdivides elements.
        Edges into 2, faces into 4.
        Not an inplace operation.

        Parameters
        -----------
        None

        Returns
        --------
        subdivided: Edges or Faces
        """
        if self.kind != "face":
            raise NotImplementedError

        else:
            whatami = self.get_whatami()
            if whatami.startswith("tri"):
                return type(self)(
                    **(utils.connec.subdivide_tri(self, return_dict=True))
                )

            elif whatami.startswith("quad"):
                return type(self)(
                    **(utils.connec.subdivide_quad(self, return_dict=True))
                )

            else:
                return None

    def tovertices(self):
        """
        Returns Vertices obj.

        Parameters
        -----------
        None

        Returns
        --------
        vertices: Vertices
        """
        return Vertices(self.vertices)
