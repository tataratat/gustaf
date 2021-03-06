"""gustaf/gustaf/volumes.py
"""

import numpy as np

from gustaf import utils
from gustaf import settings
from gustaf.faces import Faces

class Volumes(Faces):

    kind = "volume"

    __slots__ = [
        "volumes",
        "volumes_sorted",
        "volumes_unique",
        "volumes_unique_id",
        "volumes_unique_inverse",
        "volumes_unique_count",
    ]

    def __init__(
            self,
            vertices=None,
            volumes=None,
            elements=None,
    ):

        self.whatami = "volumes"
        self.vis_dict = dict()
        self.vertexdata = dict()

        if vertices is not None:
            self.vertices = utils.arr.make_c_contiguous(
                vertices,
                settings.FLOAT_DTYPE,
            )

        if volumes is not None:
            self.volumes = utils.arr.make_c_contiguous(
                volumes,
                settings.INT_DTYPE,
            )

        elif elements is not None:
            self.volumes = utils.arr.make_c_contiguous(
                elements,
                settings.INT_DTYPE,
            )

    def process(
            self,
            faces=True,
            force_process=True,
    ):
        pass

    def get_whatami(self):
        """
        Determines whatami.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        if self.volumes.shape[1] == 4:
            self.whatami = "tet"
        elif self.volumes.shape[1] == 8:
            self.whatami = "hexa"
        else:
            raise ValueError(
                "I have invalid volumes array shape. It should be (n, 4) or "
                f"(n, 8), but I have: {self.faces.shape}"
            )

        return self.whatami

    def update_faces(self):
        """
        """
        raise NotImplementedError

    def update_volumes(self, *args, **kwargs):
        """
        Alias to update_elements.
        """
        self.update_elements(*args, **kwargs)

    def get_volumes_sorted(self):
        """
        Sort volumes along axis=1.

        Parameters
        -----------
        None

        Returns
        --------
        volumes_sorted: (volumes.shape) np.ndarray
        """
        self.volumes_sorted = self.volumes.copy()
        self.volumes_sorted.sort(axis=1)

        return self.volumes_sorted

    def get_volumes_unique(self):
        """
        Returns unique volumes.

        Parameters
        -----------
        None

        Returns
        --------
        volumes_unique: (n, 4) or (n, 8) np.ndarray
        """
        unique_stuff = utils.arr.unique_rows(
            self.get_volumes_sorted(),
            return_index=True,
            return_inverse=True,
            return_counts=True,
            dtype_name=settings.INT_DTYPE,
        )

        # unpack
        #  set volumes_unique with `volumes` to avoid orientation change
        self.volumes_unique_id = unique_stuff[1].astype(settings.INT_DTYPE)
        self.volumes_unique = self.volumes[self.volumes_unique_id]
        self.volumes_unique_inverse = unique_stuff[2].astype(
            settings.INT_DTYPE
        )
        self.volumes_unique_count = unique_stuff[2].astype(settings.INT_DTYPE)

        return self.volumes_unique

    def get_volumes_unique_id(self):
        """
        Similar to faces_unique_id but for volumes.

        Parameters
        -----------
        None

        Returns
        --------
        volumes_unique: (n,) np.ndarray
        """
        _ = self.get_volumes_unique()

        return self.volumes_unique_id

    def get_volumes_unique_inverse(self,):
        """
        Similar to faces_unique_inverse but for volumes.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        _ = self.get_volumes_unique()

        return self.volumes_unique_inverse

    def tofaces(self, unique=True):
        """
        Returns Faces obj.

        Parameters
        -----------
        unique: bool
          Default is True. If True, only takes unique faces.

        Returns
        --------
        faces: Faces
        """
        return Faces(
            self.vertices,
            faces=self.get_faces_unique() if unique else self.get_faces()
        )
