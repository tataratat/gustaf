"""gustaf/gustaf/volumes.py."""

import numpy as np

from gustaf import utils
from gustaf import helpers
from gustaf import settings
from gustaf.faces import Faces


class Volumes(Faces):

    kind = "volume"

    const_faces = helpers.raise_if.invalid_inherited_attr(
            Faces.const_faces,
            __qualname__,
            property_=True,
    )
    update_faces = helpers.raise_if.invalid_inherited_attr(
            Faces.update_edges,
            __qualname__,
            property_=False,
    )

    __slots__ = (
            "_volumes",
            "_const_volumes",
    )

    def __init__(
            self,
            vertices=None,
            volumes=None,
            elements=None,
    ):
        """Volumes. It has vertices and volumes. Volumes could be tetrahedrons
        or hexahedrons.

        Parameters
        -----------
        vertices: (n, d) np.ndarray
        volumes: (n, 4) or (n, 8) np.ndarray
        """
        super().__init__(vertices=vertices)
        if volumes is not None:
            self.volumes = volumes
        elif elements is not None:
            self.volumes = elements

    @helpers.data.ComputedMeshData.depends_on(["elements"])
    def faces(self):
        """Faces here aren't main property. So this needs to be computed.

        Parameters
        -----------
        None

        Returns
        --------
        faces: (n, 3) or (n, 4) np.ndarray
        """
        whatami = self.whatami
        faces = None
        if whatami.startswith("tet"):
            faces = utils.connec.tet_to_tri(self.volumes)
        elif whatami.startswith("hexa"):
            faces = utils.connec.hexa_to_quad(self.volumes)

        return faces

    @classmethod
    def whatareyou(cls, volume_obj):
        """overwrites Faces.whatareyou to tell you is this volume is tet or
        hexa.

        Parameters
        -----------
        volume_obj: Volumes

        Returns
        --------
        whatareyou: str
        """
        if not cls.kind.startswith(volume_obj.kind):
            raise TypeError("Given obj is not {cls.__qualname__}")

        if volume_obj.volumes.shape[1] == 4:
            return "tet"

        elif volume_obj.volumes.shape[1] == 8:
            return "hexa"

        else:
            raise ValueError(
                    "Invalid volumes connectivity shape. It should be (n, 4) "
                    f"or (n, 8), but given: {volume_obj.volumes.shape}"
            )

    @property
    def volumes(self):
        """Returns volumes.

        Parameters
        -----------
        None

        Returns
        --------
        volumes: (n, 4) or (n, 8) np.ndarray
        """
        return self._volumes

    @volumes.setter
    def volumes(self, vols):
        """volumes setter. Similar to vertices, this will be a tracked array.

        Parameters
        -----------
        vols: (n, 4) or (n, 8) np.ndarray

        Returns
        --------
        None
        """
        if vols is not None:
            utils.arr.is_one_of_shapes(
                    vols,
                    ((-1, 4), (-1, 8)),
                    strict=True,
            )

        self._volumes = helpers.data.make_tracked_array(
                vols,
                settings.INT_DTYPE,
        )
        # same, but non-writeable view of tracked array
        self._const_volumes = self._volumes.view()
        self._const_volumes.flags.writeable = False

    @property
    def const_volumes(self):
        """Returns non-writeable view of volumes.

        Parameters
        -----------
        None

        Returns
        --------
        const_volumes: (n, 4) or (n, 8) np.ndarray
        """
        return self._const_volumes

    @helpers.data.ComputedMeshData.depends_on(["elements"])
    def sorted_volumes(self):
        """Sort volumes along axis=1.

        Parameters
        -----------
        None

        Returns
        --------
        volumes_sorted: (volumes.shape) np.ndarray
        """
        volumes = self._get_attr("volumes")

        return np.sort(volumes, axis=1)

    @helpers.data.ComputedMeshData.depends_on(["elements"])
    def unique_volumes(self):
        """Returns a namedtuple of unique volumes info. Similar to
        unique_edges.

        Parameters
        -----------
        None

        Returns
        --------
        unique_info: Unique2DIntegers
          valid attribut4es are {values, ids, inverse, counts}
        """
        unique_info = utils.connec.sorted_unique(
                self.sorted_volumes(),
                sorted_=True,
        )

        volumes = self._get_attr("volumes")

        unique_info.values[:] = volumes[unique_info.ids]

        return unique_info

    def update_volumes(self, *args, **kwargs):
        """Alias to update_elements."""
        self.update_elements(*args, **kwargs)

    def tofaces(self, unique=True):
        """Returns Faces obj.

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
                faces=self.unique_faces().values if unique else self.faces()
        )
