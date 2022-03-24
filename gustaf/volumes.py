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

        #if volumes is not None or elements is not None:
        #    if self.volumes.shape[1] == 4:
        #        self.whatami = "tet"
        #    elif self.volumes.shape[1] == 8:
        #        self.whatami = "hexa"

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

    def shrink(self, ratio=.8, map_vertexdata=True):
        """
        Returns shrinked faces.

        Parameters
        -----------
        ratio: float
          Default is 0.8
        map_vertexdata: bool
          Default is True. Maps all vertexdata using `shrinked_data_mapping`.

        Returns
        --------
        s_faces: Faces
          shrinked faces
        """
        vs = np.vstack(self.vertices[self.get_faces()])
        fs = np.arange(len(vs))

        whatami = self.get_whatami()
        reshape = 3 if whatami.startswith("tet") else 4
        fs = fs.reshape(-1, reshape)

        repeats = 4 if whatami.startswith("tet") else 6
        mids = np.repeat(self.get_centers(), repeats * reshape, axis=0)

        vs -= mids
        vs *= ratio
        vs += mids

        s_faces = Faces(vs, fs)

        if map_vertexdata:
            faces_flat = self.faces.ravel()
            for key, value in self.vertexdata.items():
                s_faces.vertexdata[key] = value[faces_flat]

            # probably wanna take visulation options too
            s_faces.vis_dict = self.vis_dict

        return s_faces

    def shrinked_data_mapping(self):
        """
        Provides data mapping to transfer vertexdata to shrinked faces.

        Parameters
        -----------
        None

        Returns
        --------
        shrinked_data_mapping: (n, m) np.ndarray
        """
        return self.get_faces().ravel()
