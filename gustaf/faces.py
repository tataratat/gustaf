"""gustaf/gustaf/faces.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as _np

from gustaf import helpers as _helpers
from gustaf import settings as _settings
from gustaf import show as _show
from gustaf import utils as _utils
from gustaf.edges import Edges as _Edges
from gustaf.helpers.options import Option as _Option

if TYPE_CHECKING:

    from gustaf.helpers.data import TrackedArray, Unique2DIntegers

# special types for face texture option
try:
    import vedo

    vedoPicture = vedo.Picture
    # there are other ways to get here, but this is exact path for our use
    vtkTexture = vedo.vtkclasses.vtkTexture
except ImportError as err:
    vedoPicture = _helpers.raise_if.ModuleImportRaiser("vedo", err)
    vtkTexture = _helpers.raise_if.ModuleImportRaiser("vedo", err)


class FacesShowOption(_helpers.options.ShowOption):
    """
    Show options for vertices.
    """

    _valid_options = _helpers.options.make_valid_options(
        *_helpers.options.vedo_common_options,
        _Option(
            "vedo", "lw", "Width of edges (lines) in pixel units.", (int,)
        ),
        _Option(
            "vedo", "lc", "Color of edges (lines).", (int, str, tuple, list)
        ),
        _Option(
            "vedo",
            "texture",
            "Texture of faces in array, vedo.Picture, vtk.vtkTexture, "
            "or path to an image.",
            (_np.ndarray, tuple, list, str, vedoPicture, vtkTexture),
        ),
    )

    _helps = "Faces"

    def _initialize_showable(self):
        """
        Initializes Faces as vedo.Mesh

        Parameters
        ----------
        None

        Returns
        -------
        faces: vedo.Mesh
        """

        faces = _show.vedo.Mesh(
            [self._helpee.const_vertices, self._helpee.const_faces],
        )

        for option in ["lw", "lc", "texture"]:
            val = self.get(option, False)
            if val:
                getattr(faces, option)(val)

        return faces


class Faces(_Edges):
    kind = "face"

    const_edges = _helpers.raise_if.invalid_inherited_attr(
        "Edges.const_edges",
        __qualname__,
        property_=True,
    )
    update_edges = _helpers.raise_if.invalid_inherited_attr(
        "Edges.update_edges",
        __qualname__,
        property_=False,
    )
    dashed = _helpers.raise_if.invalid_inherited_attr(
        "Edges.dashed",
        __qualname__,
        property_=False,
    )

    __slots__ = (
        "_faces",
        "_const_faces",
        "BC",
    )

    __show_option__ = FacesShowOption
    __boundary_class__ = _Edges

    def __init__(
        self,
        vertices: list[list[float]] | TrackedArray | _np.ndarray = None,
        faces: _np.ndarray | None = None,
        elements: _np.ndarray | None = None,
    ) -> None:
        """Faces. It has vertices and faces. Faces could be triangles or
        quadrilaterals.

        Parameters
        -----------
        vertices: (n, d) np.ndarray
        faces: (n, 3) or (n, 4) np.ndarray
        """
        super().__init__(vertices=vertices)
        if faces is not None:
            self.faces = faces

        elif elements is not None:
            self.faces = elements

        self.BC = {}

    @_helpers.data.ComputedMeshData.depends_on(["elements"])
    def edges(self) -> _np.ndarray:
        """Edges from here aren't main property. So this needs to be computed.

        Parameters
        -----------
        None

        Returns
        --------
        edges: (n, 2) np.ndarray
        """
        self._logd("computing edges")
        faces = self._get_attr("faces")

        return _utils.connec.faces_to_edges(faces)

    @property
    def whatami(self) -> str:
        """Determines whatami.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        return type(self).whatareyou(self)

    @classmethod
    def whatareyou(cls, face_obj: Faces) -> str:
        """classmethod that tells you if the Faces is tri or quad or invalid
        kind.

        Parameters
        -----------
        face_obj: Faces

        Returns
        --------
        whatareyou: str
        """
        if not cls.kind.startswith(face_obj.kind):
            raise TypeError("Given obj is not {cls.__qualname__}")

        if face_obj.faces.shape[1] == 3:
            return "tri"

        elif face_obj.faces.shape[1] == 4:
            return "quad"

        else:
            raise ValueError(
                "Invalid faces connectivity shape. It should be (n, 3) or "
                f"(n, 4), but given: {face_obj.faces.shape}"
            )

    @property
    def faces(self) -> TrackedArray:
        """Returns faces.

        Parameters
        -----------
        None

        Returns
        --------
        faces
        """
        self._logd("returning faces")
        return self._faces

    @faces.setter
    def faces(self, fs: TrackedArray | _np.ndarray) -> None:
        """Faces setter. Similar to vertices, this will be a tracked array.

        Parameters
        -----------
        fs: (n, 2) np.ndarray

        Returns
        --------
        None
        """
        self._logd("setting faces")

        self._faces = _helpers.data.make_tracked_array(
            fs,
            _settings.INT_DTYPE,
            copy=False,
        )
        # shape check
        if fs is not None:
            _utils.arr.is_one_of_shapes(
                fs,
                ((-1, 3), (-1, 4)),
                strict=True,
            )

        # same, but non-writeable view of tracked array
        self._const_faces = self._faces.view()
        self._const_faces.flags.writeable = False

    @property
    def const_faces(self) -> TrackedArray:
        """Returns non-writeable view of faces.

        Parameters
        -----------
        None

        Returns
        --------
        const_faces: (n, 2
        """
        return self._const_faces

    @_helpers.data.ComputedMeshData.depends_on(["elements"])
    def sorted_faces(self) -> _np.ndarray:
        """Similar to edges_sorted but for faces.

        Parameters
        -----------
        None

        Returns
        --------
        sorted_faces: (self.faces.shape) np.ndarray
        """
        faces = self._get_attr("faces")

        return _np.sort(faces, axis=1)

    @_helpers.data.ComputedMeshData.depends_on(["elements"])
    def unique_faces(self) -> Unique2DIntegers:
        """Returns a namedtuple of unique faces info. Similar to unique_edges.

        Parameters
        -----------
        None

        Returns
        --------
        unique_info: Unique2DIntegers
          valid attributes are {values, ids, inverse, counts}
        """
        unique_info = _utils.connec.sorted_unique(
            self.sorted_faces(), sorted_=True
        )

        faces = self._get_attr("faces")

        unique_info.values[:] = faces[unique_info.ids]

        return unique_info

    @_helpers.data.ComputedMeshData.depends_on(["elements"])
    def single_faces(self) -> _np.ndarray:
        """Returns indices of very unique faces: faces that appear only once.
        For well constructed volumes, this can be considered as surfaces.

        Parameters
        -----------
        None

        Returns
        --------
        single_faces: (m,) np.ndarray
        """
        unique_info = self.unique_faces()

        return unique_info.ids[unique_info.counts == 1]

    def update_faces(self, *args, **kwargs):
        """Alias to update_elements."""
        self.update_elements(*args, **kwargs)

    def to_edges(self, unique=True):
        """Returns Edges obj.

        Parameters
        -----------
        unique: bool
          Default is True. If True, only takes unique edges.

        Returns
        --------
        edges: Edges
        """
        return _Edges(
            self.vertices,
            edges=self.unique_edges().values if unique else self.edges(),
        )

    def to_subelements(
        self,
        unique=True,  # noqa ARG002 # used inside the return eval str
    ):
        """Returns current elements represented as its boundary element class.
        For faces, this is equivalent to `to_edges()`.
        For volumes, `to_faces()`.

        Parameters
        ----------
        unique: bool
          Default is True. If True, only takes unique edges.

        Returns
        -------
        subelements: boundary class
        """
        return getattr(
            self, f"to_{self.__boundary_class__.__qualname__.lower()}"
        )(unique)
