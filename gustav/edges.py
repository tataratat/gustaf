"""gustav/gustav/edges.py

Edges. Also known as 
"""

import numpy as np

from gustav import settings
from gustav import utils
from gustav.vertices import Vertices

class Edges(Vertices):

    __slots__ = [
        "edges",
        "_edges_sorted",
        "_edges_unique",
        "_edges_unique_id",
        "_edges_unique_inverse",
        "_edges_unique_count",
    ]

    def __init__(
            self,
            vertices=None,
            edges=None,
            elements=None,
    ):
        if self.vertices is not None:
            self.vertices = utils.make_c_contiguous(
                vertices,
                settings.FLOATS_DTYPE
            )

        if edges is not None:
            self.edges = utils.make_c_contiguous(edges, settings.INT_DTYPE)
        elif elements is not None:
            self.edges = utils.make_c_contiguous(
                elements,
                settings.INT_DTYPE
            )

        self.whatami = "edges"
        self.kind = "edge"

    def process(
            self,
            edges_sorted=True,
            edges_unique=True,
            edges_unique_id=True,
            edges_unique_inverse=True,
            edges_unique_count=True,
            force_process=True,
    ):
        pass

    def edges_sorted(self):
        """
        Sort edges along axis=1.

        Parameters
        -----------
        None

        Returns
        --------
        edges_sorted: (n_edges, 2) np.ndarray
        """
        self.edges = utils.make_c_contiguous(
            self.edges,
            settings.INT_DTYPE,
        )
        self._edges_sorted = self.edges.copy()
        self._edges_sorted.sort(axis=1)

        return self._edges_sorted

    def edges_unique(self):
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
            self.edges_sorted(),
            return_index=True,
            return_inverse=True,
            return_counts=True,
            dtype_name=settings.INT_DTYPE,
        )

        # unpack
        self._edges_unique = unique_stuff[0]
        self._edges_unique_ids = unique_stuff[1]
        self._edges_unique_inverse = unique_stuff[2]
        self._edges_unique_count = unique_stuff[3]
        self._outlines = self._edges_unique_ids[self._edges_unique_count == 1]

        return self._edges_unique

    def edges_unique_id(self):
        """
        Returns ids of unique edges.

        Parameters
        -----------
        None

        Returns
        --------
        edges_unique_id: (n,) np.ndarray
        """
        _ = self.edges_unique()

        return self._edges_unique_id

    def edges_unique_inverse(self):
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
        _ = self.edges_unique()

        return self._edges_unique_inverse

    def outlines(self):
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
        _ = self.edges_unique()

        return self._outlines

    def update_elememts(self, mask, inplace=True):
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
            ).remove_unreferrenced_vertices(inplace=True)

    # alias
    update_edges = self.update_elements

    def subdivide(self):
        """
        """
        pass
