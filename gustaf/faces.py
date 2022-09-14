"""gustaf/gustaf/faces.py
"""

from gustaf import settings
from gustaf import utils
from gustaf.edges import Edges
from gustaf import helpers

class Faces(Edges):

    kind = "face"

    const_edges = helpers.raise_if.invalid_inherited_attr(
        Edges.const_edges,
        __qualname__,
        property_=True,
    )
    update_edges = helpers.raise_if.invalid_inherited_attr(
        Edges.update_edges,
        __qualname__,
        property_=False,
    )
    dashed = helpers.raise_if.invalid_inherited_attr(
        Edges.const_edges,
        __qualname__,
        property_=False,
    )



    __slots__ = (
        "_faces",
        "_const_faces",
        "BC",
    )

    def __init__(
            self,
            vertices=None,
            faces=None,
            elements=None,
    ):
        """
        Faces. It has vertices and faces. Faces could be triangles or
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
            self.faces = faces

        self.whatami = "faces"
        self.BC = dict()
        self.const_edges = raise_if

    @helpers.data.ComputedMeshData.depends_on(["elements"])
    def edges(self):
        """
        Edges from here aren't main property.
        So this needs to be computed.

        Parameters
        -----------
        None

        Returns
        --------
        edges: (n, 2) np.ndarray
        """
        self._logd("computing edges")
        faces = self._get_attr("faces"):

        return utils.connec.faces_to_edges(faces)

    @property
    def whatami(self,):
        """
        Determines whatami.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        return type(self).whatisthis(self)

    @classmethod
    def whatisthis(cls, face_obj):
        """
        classmethod that tells you if the Faces is tri or quad or invalid kind.

        Parameters
        -----------
        face_obj: Faces

        Returns
        --------
        whatisthis: str
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
    def faces(self,):
        """
        Returns faces.

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
        """
        Faces setter. Similar to veritces, this will be a tracked array

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
        )
        # same, but non-writeable view of tracekd array
        self._const_faces = self._faces.view()
        self._const_edges.flags.writeable = False

    @property
    def const_faces(self):
        """
        Returns non-writeable view of faces

        Parameters
        -----------
        None

        Returns
        --------
        const_faces: (n, 2
        """
        return self._const_faces

    @helpers.data.ComputedMeshData.depends_on(["elements"])
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
        unique_info = self.unique_edges()

        return unique_info.ids[unique_info.counts == 1]

    def get_faces_sorted(self):
        """
        Similar to edges_sorted but for faces.

        Parameters
        -----------
        None

        Returns
        --------
        faces_sorted: (self.faces.shape) np.ndarray
        """
        self.faces_sorted = self.get_faces().copy()
        self.faces_sorted.sort(axis=1)

        return self.faces_sorted

    def get_faces_unique(self):
        """
        Similar to edges_unique but for faces.

        Parameters
        -----------
        None

        Returns
        --------
        faces_unique: (n, d) np.ndarray
        """
        unique_stuff = utils.arr.unique_rows(
            self.get_faces_sorted(),
            return_index=True,
            return_inverse=True,
            return_counts=True,
            dtype_name=settings.INT_DTYPE,
        )

        # unpack
        #  set faces_unique with `faces` to avoid orientation change
        self.faces_unique_id = unique_stuff[1].astype(settings.INT_DTYPE)
        self.faces_unique = self.faces[self.faces_unique_id]
        self.faces_unique_inverse = unique_stuff[2].astype(settings.INT_DTYPE)
        self.faces_unique_count = unique_stuff[3].astype(settings.INT_DTYPE)
        self.surfaces = self.faces_unique_id[self.faces_unique_count == 1]

        return self.faces[self.faces_unique_id]

    def get_faces_unique_id(self):
        """
        Similar to edges_unique_id but for faces.

        Parameters
        -----------
        None

        Returns
        --------
        faces_unique: (n,) np.ndarray
        """
        _ = self.get_faces_unique()

        return self.faces_unique_id

    def get_faces_unique_inverse(self,):
        """
        Similar to edges_unique_inverse but for faces.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        _ = self.get_faces_unique()

        return self.faces_unique_inverse

    def get_surfaces(self,):
        """
        Returns indices of very unique faces: faces that appear only once.
        For well constructed faces, this can be considred as surfaces.

        Parameters
        -----------
        None

        Returns
        --------
        surfaces: (m,) np.ndarray
        """
        _ = self.get_faces_unique()

        return self.surfaces

    def update_edges(self):
        """
        Just to decouple inherited alias

        Raises
        -------
        NotImplementedError
        """
        raise NotImplementedError

    def update_faces(self, *args, **kwargs):
        """
        Alias to update_elements.
        """
        self.update_elements(*args, **kwargs)

    def toedges(self, unique=True):
        """
        Returns Edges obj.

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
            edges=self.get_edges_unique() if unique else self.get_edges()
        )

    #def show(self, BC=False):
        """
        Overwrite `show` to offer frequently used showing options

        Parameters
        -----------
        BC: bool
          Default is False.

        Returns
        --------
        None
        """
