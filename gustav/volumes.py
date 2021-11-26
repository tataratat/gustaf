"""gustav/gustav/volumes.py
"""

from gustav import utils
from gustav import settings
from gustav.faces import Faces

class Volumes(Faces):

    __slots__ = [
        "volumes",
    ]

    def __init__(
            self,
            vertices,
            volumes,
            elements,
    ):

        self.whatami = "volumes"
        self.kind = "volume"

        if vertices is not None:
            self.vertices = utils.make_c_contiguous(
                vertices,
                settings.FLOAT_DTYPE,
            )

        if volumes is not None:
            self.volumes = utils.make_c_contiguous(
                volumes,
                settings.INT_DTYPE,
            )

        elif elements is not None:
            self.volumes = utils.make_c_contiguous(
                elements,
                settings.INT_DTYPE,
            )

        if volumes is not None or elements is not None:
            if self.volumes.shape[1] == 4:
                self.whatami = "tet"
            elif self.volumes.shape[1] == 8:
                self.whatami = "hexa"

    def process(
            self,
            faces=True,
            force_process=True,
    ):
        pass

    def faces(self,):
        """
        Generates faces based on volumes and returns.

        Parameters
        -----------
        None

        Returns
        --------
        faces: (n, d) np.ndarray
        """
        #self.faces = utils.connec.volumes_to_faces(self.volumes)
        if self.volumes.shape[1] == 4:
            self.whatami = "tet"
            self.faces = utils.connec.tet_to_tri(self.volumes)

        elif self.volumes.shape[1] == 8:
            self.whatami = "hexa"
            self.faces = utils.connec.hexa_to_quad(self.volumes)

        return self.faces
