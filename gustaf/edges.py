"""gustaf/gustaf/edges.py.

Edges. Also known as lines.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as _np

from gustaf import helpers as _helpers
from gustaf import settings as _settings
from gustaf import show as _show
from gustaf import utils as _utils
from gustaf.helpers.options import Option as _Option
from gustaf.vertices import Vertices as _Vertices

if TYPE_CHECKING:
    from typing import Any

    from gustaf.faces import Faces
    from gustaf.helpers.data import TrackedArray, Unique2DIntegers
    from gustaf.volumes import Volumes


class EdgesShowOption(_helpers.options.ShowOption):
    """
    Show options for vertices.
    """

    _valid_options = _helpers.options.make_valid_options(
        *_helpers.options.vedo_common_options,
        _Option(
            "vedo",
            "lw",
            "Width of edges (lines) in pixel units.",
            (float, int),
        ),
        _Option("vedo", "as_arrows", "Show edges as arrows.", (bool,)),
        _Option(
            "vedo",
            "head_radius",
            "Radius of arrow head. Applicable if as_arrows is True",
            (float, int),
        ),
        _Option(
            "vedo",
            "head_length",
            "Length of arrow head. Applicable if as_arrows is True",
            (float, int),
        ),
        _Option(
            "vedo",
            "shaft_radius",
            "Radius of arrow shaft. Applicable if as_arrows is True",
            (float, int),
        ),
    )

    _helps = "Edges"

    def _initialize_showable(self):
        """
        Initializes edges as either vedo.Lines or vedo.Arrows

        Parameters
        ----------
        None

        Returns
        -------
        edges: vedo.Lines or vedo.Arrows
        """
        if self.get("as_arrows", False):
            init_options = ("head_radius", "head_length", "shaft_radius")
            return _show.vedo.Arrows(
                self._helpee.const_vertices[self._helpee.edges],
                **self[init_options],
            )

        else:
            init_options = ("lw",)
            return _show.vedo.Lines(
                self._helpee.const_vertices[self._helpee.edges],
                **self[init_options],
            )


class Edges(_Vertices):
    kind = "edge"

    __slots__ = (
        "_edges",
        "_const_edges",
    )

    __show_option__ = EdgesShowOption
    __boundary_class__ = _Vertices

    def __init__(
        self,
        vertices: list[list[float]] | TrackedArray | _np.ndarray = None,
        edges: _np.ndarray | None = None,
        elements: _np.ndarray | None = None,
    ) -> None:
        """Edges. It has vertices and edges. Also known as lines.

        Parameters
        -----------
        vertices: (n, d) np.ndarray
        edges: (n, 2) np.ndarray
        """
        super().__init__(vertices=vertices)

        if edges is not None:
            self.edges = edges

        elif elements is not None:
            self.edges = elements

    @property
    def edges(self) -> TrackedArray:
        """Returns edges. If edges is not its original property.

        Parameters
        -----------
        None

        Returns
        --------
        edges: (n, 2) np.ndarray
        """
        self._logd("returning edges")
        return self._edges

    @edges.setter
    def edges(self, es: TrackedArray | _np.ndarray) -> None:
        """Edges setter. Similar to vertices, this is a tracked array.

        Parameters
        -----------
        es: (n, 2) np.ndarray

        Returns
        --------
        None
        """
        self._logd("setting edges")

        self._edges = _helpers.data.make_tracked_array(
            es, _settings.INT_DTYPE, copy=False
        )

        # shape check
        if es is not None:
            _utils.arr.is_shape(es, (-1, 2), strict=True)

        # same, but non-writeable view of tracked array
        self._const_edges = self._edges.view()
        self._const_edges.flags.writeable = False

    @property
    def const_edges(self) -> TrackedArray:
        """Returns non-writeable version of edges.

        Parameters
        -----------
        None

        Returns
        --------
        const_edges (n, 2) np.ndarray
        """
        return self._const_edges

    @property
    def whatami(self) -> str:
        """whatami?

        Parameters
        -----------
        None

        Returns
        --------
        whatami: str
        """
        return "edges"

    @_helpers.data.ComputedMeshData.depends_on(["elements"])
    def sorted_edges(self) -> _np.ndarray:
        """Sort edges along axis=1.

        Parameters
        -----------
        None

        Returns
        --------
        edges_sorted: (n_edges, 2) np.ndarray
        """
        edges = self._get_attr("edges")

        return _np.sort(edges, axis=1)

    @_helpers.data.ComputedMeshData.depends_on(["elements"])
    def unique_edges(self) -> Unique2DIntegers:
        """Returns a named tuple of unique edge info. Info includes unique
        values, ids of unique edges, inverse ids, count of each unique values.

        Parameters
        -----------
        None

        Returns
        --------
        unique_info: Unique2DIntegers
          valid attributes are {values, ids, inverse, counts}
        """
        unique_info = _utils.connec.sorted_unique(
            self.sorted_edges(), sorted_=True
        )

        edges = self._get_attr("edges")

        # tuple is not assignable, but entry is mutable...
        unique_info.values[:] = edges[unique_info.ids]

        return unique_info

    @_helpers.data.ComputedMeshData.depends_on(["elements"])
    def single_edges(self) -> _np.ndarray:
        """Returns indices of very unique edges: edges that appear only once.
        For well constructed faces, this can be considered as outlines.

        Parameters
        -----------
        None

        Returns
        --------
        outlines: (m,) np.ndarray
        """
        unique_info = self.unique_edges()

        return unique_info.ids[unique_info.counts == 1]

    @property
    def elements(self) -> TrackedArray:
        """Returns current connectivity. A short cut in FE friendly term.
        Elements mean different things for different classes: Vertices ->
        vertices Edges -> edges Faces -> faces Volumes -> volumes.

        Parameters
        -----------
        None

        Returns
        --------
        elements: (n, d) np.ndarray
          int. iff elements=None
        """
        elem_name = type(self).__qualname__.lower()
        self._logd(f"returning {elem_name}")

        return getattr(self, elem_name)

    @elements.setter
    def elements(self, elements: TrackedArray | _np.ndarray) -> Any | None:
        """Calls corresponding connectivity setter. A short cut in FEM friendly
        term. Vertices -> vertices Edges -> edges Faces -> faces Volumes ->
        volumes.

        Parameters
        -----------
        elements: (n, d) np.ndarray

        Returns
        --------
        None
        """
        # naming rule in gustaf
        elem_name = type(self).__qualname__.lower()
        self._logd(f"Setting {elem_name}'s connectivity.")
        return setattr(self, elem_name, elements)

    @property
    def const_elements(self) -> TrackedArray:
        """Returns non-mutable version of elements.

        Parameters
        -----------
        None

        Returns
        --------
        non_mutable_elements: (n, d) TrackedArray
        """
        self._logd("returning const_elements")
        return getattr(self, "const_" + type(self).__qualname__.lower())

    @_helpers.data.ComputedMeshData.depends_on(["vertices", "elements"])
    def centers(self) -> TrackedArray:
        """Center of elements.

        Parameters
        -----------
        None

        Returns
        --------
        centers: (n_elements, d) np.ndarray
        """
        self._logd("computing centers")

        return self.const_vertices[self.const_elements].mean(axis=1)

    @_helpers.data.ComputedMeshData.depends_on(["vertices", "elements"])
    def referenced_vertices(
        self,
    ) -> _np.ndarray:
        """Returns mask of referenced vertices.

        Parameters
        -----------
        None

        Returns
        --------
        referenced: (n,) np.ndarray
        """
        referenced = _np.zeros(len(self.const_vertices), dtype=bool)
        referenced[self.const_elements] = True

        return referenced

    def remove_unreferenced_vertices(self) -> Edges | Faces | Volumes:
        """Remove unreferenced vertices. Adapted from
        `github.com/mikedh/trimesh`

        Parameters
        -----------
        None

        Returns
        --------
        new_self: type(self)
        """
        referenced = self.referenced_vertices()

        inverse = _np.zeros(len(self.vertices), dtype=_settings.INT_DTYPE)
        inverse[referenced] = _np.arange(referenced.sum())

        return self.update_vertices(
            mask=referenced,
            inverse=inverse,
        )

    def update_elements(self, mask: _np.ndarray) -> Edges | Faces | Volumes:
        """Similar to update_vertices, but for elements.

        Parameters
        -----------
        mask: bool or (m,) np.ndarray

        Returns
        --------
        new_self: type(self)
        """
        self.elements = self.elements[mask]

        return self.remove_unreferenced_vertices()

    def update_edges(self, *args, **kwargs):
        """Alias to update_elements."""
        return self.update_elements(*args, **kwargs)

    def dashed(self, spacing: Any | None = None) -> Edges:
        """Turn edges into dashed edges(=lines). Given spacing, it will try to
        chop edges as close to it as possible. Pattern should look:

        ``dashed edges``

        .. code-block:: text

             o--------o    o--------o    o--------o
             |<------>|             |<-->|
                (chop length)         (chop length / 2)

        Parameters
        -----------
        spacing: float
          Default is None and it will use self.bounds_diagonal_norm() / 50

        Returns
        --------
        dashing_edges: Edges
        """
        if self.kind != "edge":
            raise NotImplementedError("dashed is only for edges.")

        if spacing is None:
            # apply "automatic" spacing
            spacing = self.bounds_diagonal_norm() / 50

        v0s = self.vertices[self.edges[:, 0]]
        v1s = self.vertices[self.edges[:, 1]]

        distances = _np.linalg.norm(v0s - v1s, axis=1)
        linspaces = (((distances // (spacing * 1.5)) + 1) * 3).astype(
            _np.int32
        )

        # chop vertices!
        new_vs = []
        for v0, v1, lins in zip(v0s, v1s, linspaces):
            new_vs.append(_np.linspace(v0, v1, lins))

        # we need all chopped vertices.
        # there might be duplicating vertices. you can use merge_vertices
        new_vs = _np.vstack(new_vs)
        # all mid points are explicitly defined, but they aren't required
        # so, rm.
        mask = _np.ones(len(new_vs), dtype=bool)
        mask[1::3] = False
        new_vs = new_vs[mask]

        # prepare edges
        tmp_es = _utils.connec.range_to_edges((0, len(new_vs)), closed=False)
        new_es = tmp_es[::2]

        return Edges(vertices=new_vs, edges=new_es)

    def shrink(
        self, ratio: float = 0.8, map_vertex_data: bool = True
    ) -> Edges | Faces | Volumes:
        """Returns shrunk elements.

        Parameters
        -----------
        ratio: float
          Default is 0.8
        map_vertex_data: bool
          Default is True. Maps all vertex_data.

        Returns
        --------
        s_elements: Elements
          shrunk elements
        """
        elements = self.const_elements
        vs = _np.vstack(self.vertices[elements])
        es = _np.arange(len(vs))

        nodes_per_element = elements.shape[1]
        es = es.reshape(-1, nodes_per_element)

        mids = _np.repeat(self.centers(), nodes_per_element, axis=0)

        vs -= mids
        vs *= ratio
        vs += mids

        s_elements = type(self)(vertices=vs, elements=es)

        if map_vertex_data:
            elements_flat = elements.ravel()
            for key, value in self.vertex_data.items():
                s_elements.vertex_data[key] = value[elements_flat]

            # probably wanna take visualization options too
            s_elements._show_options._options = deepcopy(
                self.show_options._options
            )

        return s_elements

    def to_vertices(self) -> _Vertices:
        """Returns Vertices obj.

        Parameters
        -----------
        None

        Returns
        --------
        vertices: Vertices
        """
        return _Vertices(self.vertices)

    def _get_attr(self, attr: str) -> TrackedArray | _np.ndarray:
        """Internal function to get attribute that maybe property or callable.
        Some properties are replaced by callable in subclasses as it may depend
        on other properties of subclass.

        Parameters
        -----------
        attr: str

        Returns
        --------
        attrib: Any
        """
        attrib = getattr(self, attr)
        return attrib() if callable(attrib) else attrib
