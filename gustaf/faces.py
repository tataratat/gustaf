"""gustaf/gustaf/faces.py
"""

from gustaf import settings
from gustaf import utils
from gustaf.edges import Edges

class Faces(Edges):

    kind = "face"

    __slots__ = [
        "faces",
        "faces_sorted",
        "faces_unique",
        "faces_unique_id",
        "faces_unique_inverse",
        "faces_unique_count",
        "surfaces",
        "BC",
    ]

    def __init__(
            self,
            vertices=None,
            faces=None,
            elements=None,
            process=False,
    ):
        if vertices is not None:
            self.vertices = utils.arr.make_c_contiguous(
                vertices,
                settings.FLOAT_DTYPE
            )

        if faces is not None:
            self.faces = utils.arr.make_c_contiguous(faces, settings.INT_DTYPE)
        elif elements is not None:
            self.faces = utils.arr.make_c_contiguous(
                elements,
                settings.INT_DTYPE
            )

        self.whatami = "faces"
        self.vis_dict = dict()
        self.vertexdata = dict()
        self.BC = dict()

        self.process(process)

    def process(
            self,
            edges=True,
            faces_sorted=True,
            faces_unique=True,
            faces_unique_id=True,
            faces_unique_inverse=True,
            outlines=True,
            force_process=True,
            **kwargs,
    ):
        pass

    def get_whatami(self,):
        """
        Determines whatami.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        if self.faces.shape[1] == 3:
            self.whatami = "tri"
        elif self.faces.shape[1] == 4:
            self.whatami = "quad"
        else:
            raise ValueError(
                "I have invalid faces array shape. It should be (n, 3) or "
                f"(n, 4), but I have: {self.faces.shape}"
            )

        return self.whatami

    def get_faces(self,):
        """
        Generates edges based on faces and returns.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        if self.kind == "face":
            self.faces = utils.arr.make_c_contiguous(
                self.faces,
                settings.INT_DTYPE,
            )

            return self.faces

        if hasattr(self, "volumes"):
            whatami = self.get_whatami()
            if whatami.startswith("tet"):
                self.faces = utils.connec.tet_to_tri(self.volumes)
            elif whatami.startswith("hexa"):
                self.faces = utils.connec.hexa_to_quad(self.volumes)

            return self.faces

        # Shouldn't reach this point, but in case it dooes
        self._logd("cannot compute/return faces.")
        return None

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
        self.faces_unique = unique_stuff[0]
        self.faces_unique_id = unique_stuff[1].astype(settings.INT_DTYPE)
        self.faces_unique_inverse = unique_stuff[2].astype(settings.INT_DTYPE)
        self.faces_unique_count = unique_stuff[3].astype(settings.INT_DTYPE)
        self.surfaces = self.faces_unique_id[self.faces_unique_count == 1]

        return self.faces_unique

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
