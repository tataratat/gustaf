"""gustav/gustav/vertices.py

Vertices. Base of all "Mesh" geometries.
"""

from gustav import settings
from gustav._abstract_base import AB

class Vertices(AB):

    __slots__ = [
        "vertices",
        "_vertices_unique",
        "_vertices_unique_id",
        "_vertices_unique_inverse",
        "_bounds",
        "_centers",
    ]

    def __init__(
            self,
            vertices=None,
    ):
        """
        Vertices. It has vertices.

        Parameters
        -----------
        vertices: (n, d) np.ndarray

        Returns
        --------
        None
        """
        if vertices is not None:
            self.vertices = utils.arr.make_c_contiguous(
                vertices,
                settings.FLOAT_DTYPE
            )
        self.whatami = "vertices"
        self.kind = "vertex"

    def process(
            self,
            vetices_unique=True,
            vertices_unique_id=True,
            vertices_unique_inverse=True,
            bounds=True,
            centers=True,
            force_process=True,
    ):
        """
        Returns unique vertices.

        Parameters
        -----------
        None

        Returns
        --------
        """
        self.vertices = utils.make_c_contiguous(
            self.vertices,
            settings.FLOAT_DTYPE
        )

        if (
            vertices_unqiue
            or vertices_unique_id
            or vertices_unique_inverse
            or force_process
        ):
            self.vertices_unique()

        if bounbds or force_process:
            self.bounds()

        if centers or force_process:
            self.centers()

    def elements(self, elements=None, **processkwargs):
        """
        Returns current elements.
        Elements mean different things for different classes:
          Vertices -> vertices
          Edges -> edges
          Faces -> faces
          Volumes -> volumes
        Could be understood as connectivity.
        Inplace updates only.

        Parameters
        -----------
        elements: (n, d) np.ndarray
          Only updates if it is not None.

        Returns
        --------
        None
        """
        if hasattr(self, "volumes"):
            if elements is None:
                return self.volumes
            else:
                self.volumes = elements
                self.process(**processkwargs)

        elif hasattr(self, "faces"):
            if elements is None:
                return self.faces
            else:
                self.faces = elements
                self.process(**processkwargs)

        elif hasattr(self, "edges"):
            if elements is None:
                return self.edges
            else:
                self.edges = elements
                self.process(**processkwargs)

        elif hasattr(self, "vertices"):
            return np.arange(
                (self.vertices.shape[0], 1),
                dtype=settings.INT_DTYPE
            )

        else:
            return None

    def vertices_unique(self):
        """
        jjcpp unique
        """
        pass

    def vertices_unique_id(self):
        """
        """
        pass

    def vertices_unique_inverse(self):
        """
        """
        pass

    def bounds(self):
        """
        Returns bounds of the vertices.

        Parameters
        -----------
        None

        Returns
        bounds: (d,) np.ndarray
        """
        self._bounds = utils.arr.bounds(self.vertices)

        return self._bounds

    def centers(self):
        """
        Center of elements.

        Parameters
        -----------
        None

        Returns
        --------
        centers: (n_elements, d) np.ndarray
        """
        elements = self.elements()
        self._centers = self.vertices[elements].mean(axis=1)

        return self._centers

#    def clear(self):
        """
        Clear all properties.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
#        self._logd("clearning attributes")
#        for s in self.__slots__:
#            if hasattr(self, s):
#                delattr(self, s)

#        self._logd("all attributes are cleared!")

    def update_vertices(self, mask, inverse=None, inplace=True):
        """
        Update vertices with a mask.
        Adapted from `github.com/mikedh/trimesh`.

        Parameters
        -----------
        """
        vertices = self.vertices.copy()

        # make mask numpy array
        mask = np.asarray(mask)

        if (
            (mask.dtype.name == "bool" and mask.all())
            or len(mask) == 0
        ):
            # Nothing to do
            return None

        # create inverse mask if not passed
        if inverse is None:
            inverse = np.zeros(len(vertices, dtype=settings.INT_DTYPE))
            if mask.dtype.kind == "b":
                inverse[mask] = np.arange(mask.sum())
            elif mask.dtype.kind == "i":
                inverse[mask] = np.arange(len(mask))
            else:
                inverse = None

        # re-index elements from inverse
        # TODO: Here could be a good place to preserve BCs.
        elements = None
        if inverse is not None and self.kind != "vertex":
            elements = self.elements().copy()
            elements = inverse[elements.reshape(-1)].reshape(
                (-1, elements.shape[1])
            )

        # apply mask
        vertices = vertices[mask]

        # update
        if inplace:
            self.vertices = vertices
            if elements is not None:
                self.elements(elements)
                return None

            else:
                return None

        else:
            if element is None:
                return type(self)(vertices=vertices)

            else:
                return type(self)(vertices=vertices, elements=elements)

    def select_vertices(self, ranges):
        """
        Returns vertices inside the given range.

        Parameters
        -----------
        ranges: (d, 2) array-like
          Takes None.

        Returns
        --------
        ids: (n,) np.ndarray
        """
        return utils.arr.select_with_ranges(self.vertices, ranges)

    def remove_vertices(self, ids, inplace=True):
        """
        Removes vertices with given vertex ids.

        Parameters
        -----------
        ids: (n,) np.ndarray
        inplace: bool

        Returns
        --------
        new_self: type(self)
          iff inplace=True.
        """
        mask = np.ones(len(self.vertices), dtype=bool)
        mask[ids] = False

        return self.update_vertices(mask, inplace=inplace)

    def remove_unreferenced_vertices(self, inplace=True):
        """
        Remove unreferenced vertices.
        Adapted from `github.com/mikedh/trimesh`

        Parameters
        -----------
        inplace: bool

        Returns
        --------
        new_self: type(self)
          iff inplace=True.
        """
        if self.kind == "vertex":
            return None

        referenced = np.zeros(len(self.vertices), dtype=settings.INT_DTYPE)
        referenced[self.elements()] = True

        inverse = np.zeros(len(self.vertices), dtype=settings.INT_DTYPE)
        inverse[referenced] = np.arange(referenced.sum())

        return self.update_vertices(
            mask=referenced,
            inverse=inverse,
            inplace=inpalce,
        )

    def merge_vertices(self):
        """
        implement for element cases with hasattr
        """
        pass
