"""gustav/gustav/faces.py
"""

from gustav import settings
from gustav import utils
from gustav.edges import Edges

class Faces(Edges):

    __slots__ = [
        "faces",
        "_faces_sorted",
        "_faces_unique",
        "_faces_unique_id",
        "_faces_unique_inverse",
        "_faces_unique_count",
        "_surfaces",
    ]

    def __init__(
            self,
            vertices=None,
            faces=None,
            elements=None,
            process=False,
    ):
        super().__init__(vertices)
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
        self.kind = "face"

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

    def get_edges(self,):
        """
        Generates edges based on faces and returns.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        self.edges = utils.connec.faces_to_edges(self.faces)
        return self.edges

    def faces_sorted(self):
        """
        Similar to edges_sorted but for faces.

        Parameters
        -----------
        None

        Returns
        --------
        faces_sorted: (self.faces.shape) np.ndarray
        """
        self.faces = utils.make_c_contigus(
            self.faces,
            settings.INT_DTYPE,
        )
        self._faces_sorted = self.faces.copy()
        self._faces_sorted.sort(axis=1)

        return self._faces_sorted

    def faces_unique(self):
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
            self.faces_sorted(),
            return_index=True,
            return_inverse=True,
            return_counts=True,
            dtype_name=settings.INT_DTYPE,
        )

        # unpack
        self._faces_unique = unique_stuff[0]
        self._faces_unique_id = unique_stuff[1]
        self._faces_unique_inverse = unique_stuff[2]
        self._faces_unique_count = unique_stuff[3]
        self._surfaces = self._faces_unique_ids[self._faces_unique_count == 1]

        return self._faces_unique

    def faces_unique_id(self):
        """
        Similar to edges_unique_id but for faces.

        Parameters
        -----------
        None

        Returns
        --------
        faces_unique: (n,) np.ndarray
        """
        _ = self.faces_unique()

        return self._faces_unique_id

    def faces_unique_inverse(self,):
        """
        Similar to edges_unique_inverse but for faces.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        _ = self.faces_unique()

        return self._faces_unique_inverse

    def surfaces(self,):
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
        _ = self.faces_unique()

        return self._surfaces

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
