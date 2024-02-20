"""gustaf/gustaf/vertices.py.

Vertices. Base of all "Mesh" geometries.
"""

import copy

import numpy as np

from gustaf import helpers, settings, show, utils
from gustaf._base import GustafBase
from gustaf.helpers.options import Option


class VerticesShowOption(helpers.options.ShowOption):
    """
    Show options for vertices.
    """

    _valid_options = helpers.options.make_valid_options(
        *helpers.options.vedo_common_options,
        Option(
            "vedo",
            "r",
            "Radius of vertices in units of pixels.",
            (float, int),
        ),
        Option(
            "vedo",
            "labels",
            "Places a label/description str at the place of vertices.",
            (np.ndarray, tuple, list),
        ),
        Option(
            "vedo",
            "label_options",
            "Label kwargs to be passed during initialization."
            "Valid keywords are: {scale: float, xrot: float, yrot: float, "
            "zrot: float, ratio: float, precision: int, italic: bool, "
            "font: str, justify: str, c: (str, tuple, list, int), "
            "alpha: float}. "
            "As further hint, justify takes '-' joined combination of "
            "{center, mid, right, left, top, bottom}.",
            (dict,),
        ),
    )

    _helps = "Vertices"

    def _initialize_showable(self):
        """
        Initialize Vertices showable for vedo.

        Parameters
        ----------
        None

        Returns
        -------
        vertices: vedo.Points
        """
        init_options = ("r",)

        vertices = show.vedo.Points(
            self._helpee.const_vertices, **self[init_options]
        )

        labels = self.get("labels", None)
        if labels is not None:
            # check length
            if len(labels) != len(self._helpee.const_vertices):
                raise ValueError(
                    f"number of label contents ({len(labels)}) and "
                    "number of vertices"
                    f"({len(self._helpee.const_vertices)}) does not match."
                )

            # apply options and return labels
            return vertices.labels(
                content=labels,
                on="points",
                **self.get("label_options", {}),
            )

        else:
            # no labels, return Points
            return vertices


class Vertices(GustafBase):
    kind = "vertex"

    __slots__ = (
        "_vertices",
        "_const_vertices",
        "_computed",
        "_show_options",
        "_vertex_data",
    )

    # define frequently used types as dunder variable
    __show_option__ = VerticesShowOption

    def __init__(
        self,
        vertices=None,
    ):
        """Vertices. It has vertices.

        Parameters
        -----------
        vertices: (n, d) np.ndarray

        Returns
        --------
        None
        """
        # call setters
        self.vertices = vertices

        # init helpers
        self._vertex_data = helpers.data.VertexData(self)
        self._computed = helpers.data.ComputedMeshData(self)
        self._show_options = self.__show_option__(self)

    @property
    def vertices(self):
        """Returns vertices.

        Parameters
        -----------
        None

        Returns
        --------
        vertices: (n, d) np.ndarray
        """
        self._logd("returning vertices")
        return self._vertices

    @vertices.setter
    def vertices(self, vs):
        """Vertices setter. This will saved as a tracked array. This tracked
        array is very sensitive and if we do anything with it that may hint an
        inplace operation, it will be marked as modified. This includes copying
        and slicing. If you know you aren't going to modify the array, please
        consider using `const_vertices`. Somewhat c-style hint in naming.

        Parameters
        -----------
        vs: (n, d) np.ndarray

        Returns
        --------
        None
        """
        self._logd("setting vertices")

        # we try not to make copy.
        self._vertices = helpers.data.make_tracked_array(
            vs, settings.FLOAT_DTYPE, copy=False
        )

        # shape check
        if self._vertices.size > 0:
            utils.arr.is_shape(self._vertices, (-1, -1), strict=True)

        # exact same, but not tracked.
        self._const_vertices = self._vertices.view()
        self._const_vertices.flags.writeable = False

        # at each setting, validate vertex_data
        # --> by len mismatch, will clear data
        if hasattr(self, "vertex_data"):
            self.vertex_data._validate_len(raise_=False)

    @property
    def const_vertices(self):
        """Returns non-mutable view of `vertices`. Naming inspired by c/cpp
        sessions.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        self._logd("returning const_vertices")
        return self._const_vertices

    @property
    def vertex_data(self):
        """
        Returns vertex_data manager. Behaves similar to dict() and can be used
        to store values/data associated with each vertex.

        Parameters
        ----------
        None

        Returns
        -------
        vertex_data: VertexData
        """
        self._logd("returning vertex_data")
        return self._vertex_data

    @property
    def show_options(self):
        """
        Returns a show option manager for this object. Behaves similar to
        dict.

        Parameters
        ----------
        None

        Returns
        -------
        show_options: ShowOption
          A derived class that's suitable for current class.
        """
        self._logd("returning show_options")
        return self._show_options

    @property
    def whatami(self):
        """Answers deep philosophical question: "what am i"?

        Parameters
        ----------
        None

        Returns
        --------
        whatami: str
          vertices
        """
        return "vertices"

    @helpers.data.ComputedMeshData.depends_on(["vertices"])
    def unique_vertices(self, tolerance=None, **kwargs):
        """Returns a namedtuple that holds unique vertices info. Unique here
        means "close-enough-within-tolerance".

        Parameters
        -----------
        tolerance: float
          (Optional) Default is settings.TOLERANCE
        recompute: bool
          Only applicable as keyword argument. Force re-computes.

        Returns
        --------
        unique_vertices_info: Unique2DFloats
          namedtuple with `values`, `ids`, `inverse`, `intersection`.
        """
        self._logd("computing unique vertices")
        if tolerance is None:
            tolerance = settings.TOLERANCE

        values, ids, inverse, intersection = utils.arr.close_rows(
            self.const_vertices, tolerance=tolerance, **kwargs
        )

        return helpers.data.Unique2DFloats(
            values,
            ids,
            inverse,
            intersection,
        )

    @helpers.data.ComputedMeshData.depends_on(["vertices"])
    def bounds(self):
        """Returns bounds of the vertices. Bounds means AABB of the geometry.

        Parameters
        -----------
        None

        Returns
        --------
        bounds: (d,) np.ndarray
        """
        self._logd("computing bounds")
        return utils.arr.bounds(self.vertices)

    @helpers.data.ComputedMeshData.depends_on(["vertices"])
    def bounds_diagonal(self):
        """Returns diagonal vector of the bounding box.

        Parameters
        -----------
        None

        Returns
        --------
        bounds_diagonal: (d,) np.ndarray
          same as `bounds[1] - bounds[0]`
        """
        self._logd("computing bounds_diagonal")
        bounds = self.bounds()
        return bounds[1] - bounds[0]

    @helpers.data.ComputedMeshData.depends_on(["vertices"])
    def bounds_diagonal_norm(self):
        """Returns norm of bounds diagonal.

        Parameters
        -----------
        None

        Returns
        --------
        bounds_diagonal_norm: float
        """
        self._logd("computing bounds_diagonal_norm")
        return float(sum(self.bounds_diagonal() ** 2) ** 0.5)

    def update_vertices(self, mask, inverse=None):
        """Update vertices with a mask. In other words, keeps only masked
        vertices. Adapted from `github.com/mikedh/trimesh`. Updates
        connectivity accordingly too.

        Parameters
        -----------
        mask: (n,) bool or int
        inverse: (len(self.vertices),) int

        Returns
        --------
        updated_self: type(self)
        """
        vertices = self.const_vertices.copy()

        # make mask numpy array
        mask = np.asarray(mask)

        if (mask.dtype.name == "bool" and mask.all()) or len(mask) == 0:
            return self

        # create inverse mask if not passed
        check_neg = False
        if inverse is None and self.kind != "vertex":
            inverse = np.full(len(vertices), -11, dtype=settings.INT_DTYPE)
            check_neg = True
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
            elements = self.const_elements.copy()
            elements = inverse[elements.reshape(-1)].reshape(
                (-1, elements.shape[1])
            )
            # remove all the elements that's not part of inverse
            if check_neg:
                elem_mask = (elements > -1).all(axis=1)
                elements = elements[elem_mask]

        # apply mask
        vertices = vertices[mask]

        def update_vertex_data(obj, m, vertex_data):
            """apply mask to vertex data if there's any."""
            new_data = helpers.data.VertexData(obj)

            for key, values in vertex_data.items():
                # should work, since this is called after updating vertices
                new_data[key] = values[m]

            obj._vertex_data = new_data

            return obj

        # make shallow copy of saved vertex data
        v_data = self.vertex_data._saved.copy()

        # update - if number of vertices changes, this will remove all the
        #   length mis-matching data.
        self.vertices = vertices
        if elements is not None:
            self.elements = elements

        update_vertex_data(self, mask, v_data)

        return self

    def select_vertices(self, ranges):
        """Returns vertices inside the given range.

        Parameters
        -----------
        ranges: (d, 2) array-like
          Takes None.

        Returns
        --------
        ids: (n,) np.ndarray
        """
        return utils.arr.select_with_ranges(self.vertices, ranges)

    def remove_vertices(self, ids):
        """Removes vertices with given vertex ids.

        Parameters
        -----------
        ids: (n,) np.ndarray

        Returns
        --------
        new_self: type(self)
        """
        mask = np.ones(len(self.vertices), dtype=bool)
        mask[ids] = False

        return self.update_vertices(mask)

    def merge_vertices(self, tolerance=None, **kwargs):
        """Based on unique vertices, merge vertices if it is mergeable.

        Parameters
        -----------
        tolerance: float
          Default is settings.TOLERANCE

        Returns
        --------
        merged_self: type(self)
        """
        unique_vs = self.unique_vertices(tolerance, **kwargs)

        self._logd("number of vertices")
        self._logd(f"  before merge: {len(self.vertices)}")
        self._logd(f"  after merge: {len(unique_vs.ids)}")

        return self.update_vertices(
            mask=unique_vs.ids,
            inverse=unique_vs.inverse,
        )

    def showable(self, **kwargs):
        """Returns showable object, meaning object of visualization backend.

        Parameters
        -----------
        **kwargs:

        Returns
        --------
        showable: obj
          Obj of `gustaf.settings.VISUALIZATION_BACKEND`
        """
        return show.make_showable(self, **kwargs)

    def show(self, **kwargs):
        """Show current object using visualization backend.

        Parameters
        -----------
        **kwargs:


        Returns
        --------
        None
        """
        return show.show(self, **kwargs)

    def copy(self):
        """Returns deepcopy of self.

        Parameters
        -----------
        None

        Returns
        --------
        self_copy: type(self)
        """
        # all attributes are deepcopy-able
        copied = copy.deepcopy(self)

        # update helpee. otherwise keeps reference to self
        copied._show_options._helpee = copied
        copied._vertex_data._helpee = copied
        copied._computed._helpee = copied

        return copied

    @classmethod
    def concat(cls, *instances):
        """Sequentially put them together to make one object.

        Parameters
        -----------
        *instances: List[type(cls)]
          Allows one iterable object also.

        Returns
        --------
        one_instance: type(cls)
        """

        def is_concatable(inst):
            """Return true, if it is same as type(cls)"""
            return bool(isinstance(inst, cls))

        # If only one instance is given and it is iterable, adjust
        # so that we will just iterate that.
        if (
            len(instances) == 1
            and not isinstance(instances[0], str)
            and hasattr(instances[0], "__iter__")
        ):
            instances = instances[0]

        vertices = []
        has_elem = cls.kind != "vertex"
        if has_elem:
            elements = []

        # check if everything is "concatable".
        for ins in instances:
            if not is_concatable(ins):
                raise TypeError(
                    "Can't concat. One of the instances is not "
                    f"`{cls.__name__}`."
                )

            tmp_ins = ins.copy()

            # make sure each element index starts from 0 & end at len(vertices)
            if has_elem:
                tmp_ins.remove_unreferenced_vertices()

            vertices.append(tmp_ins.vertices)

            if has_elem:
                if len(elements) == 0:
                    elements.append(tmp_ins.elements)
                    e_offset = elements[-1].max() + 1

                else:
                    elements.append(tmp_ins.elements + e_offset)
                    e_offset = elements[-1].max() + 1

        if has_elem:
            return cls(
                vertices=np.vstack(vertices),
                elements=np.vstack(elements),
            )

        else:
            return Vertices(vertices=np.vstack(vertices))

    def __add__(self, to_add):
        """Concat in form of +.

        Parameters
        -----------
        to_add: type(self)

        Returns
        --------
        added: type(self)
        """
        return type(self).concat(self, to_add)
