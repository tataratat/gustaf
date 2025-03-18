"""gustaf/gustaf/faces.py."""

import numpy as np
from packaging.version import Version

from gustaf import helpers, settings, show, utils
from gustaf.edges import Edges
from gustaf.helpers.options import Option

# special types for face texture option
try:
    import vedo

    if Version(vedo.__version__) < Version("2025.5.3"):
        vedoPicture = vedo.Picture
    else:
        vedoPicture = vedo.Image
    # there are other ways to get here, but this is exact path for our use
    vtkTexture = vedo.vtkclasses.vtkTexture
except ImportError as err:
    vedoPicture = helpers.raise_if.ModuleImportRaiser("vedo", err)
    vtkTexture = helpers.raise_if.ModuleImportRaiser("vedo", err)


class FacesShowOption(helpers.options.ShowOption):
    """
    Show options for vertices.
    """

    _valid_options = helpers.options.make_valid_options(
        *helpers.options.vedo_common_options,
        Option("vedo", "lw", "Width of edges (lines) in pixel units.", (int,)),
        Option(
            "vedo", "lc", "Color of edges (lines).", (int, str, tuple, list)
        ),
        Option(
            "vedo",
            "texture",
            "Texture of faces in array, vedo.Picture/Image, vtk.vtkTexture, "
            "or path to an image.",
            (np.ndarray, tuple, list, str, vedoPicture, vtkTexture),
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

        faces = show.vedo.Mesh(
            [self._helpee.const_vertices, self._helpee.const_faces],
        )

        for option in ["lw", "lc", "texture"]:
            val = self.get(option, False)
            if val:
                getattr(faces, option)(val)

        return faces


class Faces(Edges):
    kind = "face"

    const_edges = helpers.raise_if.invalid_inherited_attr(
        "Edges.const_edges",
        __qualname__,
        property_=True,
    )
    update_edges = helpers.raise_if.invalid_inherited_attr(
        "Edges.update_edges",
        __qualname__,
        property_=False,
    )
    dashed = helpers.raise_if.invalid_inherited_attr(
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
    __boundary_class__ = Edges

    def __init__(
        self,
        vertices=None,
        faces=None,
        elements=None,
    ):
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

    @helpers.data.ComputedMeshData.depends_on(["elements"])
    def edges(self):
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

        return utils.connec.faces_to_edges(faces)

    @property
    def whatami(self):
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
    def whatareyou(cls, face_obj):
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
    def faces(self):
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
    def faces(self, fs):
        """Faces setter. Similar to vertices, this will be a tracked array.

        Parameters
        -----------
        fs: (n, 2) np.ndarray

        Returns
        --------
        None
        """
        self._logd("setting faces")

        self._faces = helpers.data.make_tracked_array(
            fs,
            settings.INT_DTYPE,
            copy=False,
        )
        # shape check
        if fs is not None:
            utils.arr.is_one_of_shapes(
                fs,
                ((-1, 3), (-1, 4)),
                strict=True,
            )

        # same, but non-writeable view of tracked array
        self._const_faces = self._faces.view()
        self._const_faces.flags.writeable = False

    @property
    def const_faces(self):
        """Returns non-writeable view of faces.

        Parameters
        -----------
        None

        Returns
        --------
        const_faces: (n, 2
        """
        return self._const_faces

    @helpers.data.ComputedMeshData.depends_on(["elements"])
    def sorted_faces(self):
        """Similar to edges_sorted but for faces.

        Parameters
        -----------
        None

        Returns
        --------
        sorted_faces: (self.faces.shape) np.ndarray
        """
        faces = self._get_attr("faces")

        return np.sort(faces, axis=1)

    @helpers.data.ComputedMeshData.depends_on(["elements"])
    def unique_faces(self):
        """Returns a namedtuple of unique faces info. Similar to unique_edges.

        Parameters
        -----------
        None

        Returns
        --------
        unique_info: Unique2DIntegers
          valid attributes are {values, ids, inverse, counts}
        """
        unique_info = utils.connec.sorted_unique(
            self.sorted_faces(), sorted_=True
        )

        faces = self._get_attr("faces")

        unique_info.values[:] = faces[unique_info.ids]

        return unique_info

    @helpers.data.ComputedMeshData.depends_on(["elements"])
    def single_faces(self):
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
        return Edges(
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
