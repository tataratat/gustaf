"""gustav/gustav/volumes.py
"""

from gustav import utils
from gustav import settings
from gustav.faces import Faces

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
                + "(n, 8), but I have: " + self.faces.shape
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
