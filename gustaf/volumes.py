"""gustaf/gustaf/volumes.py."""

import numpy as np

from gustaf import helpers, settings, show, utils
from gustaf.faces import Faces
from gustaf.helpers.options import Option


class VolumesShowOption(helpers.options.ShowOption):
    """
    Show options for vertices.
    """

    _valid_options = helpers.options.make_valid_options(
        *helpers.options.vedo_common_options,
        Option("vedo", "lw", "Width of edges (lines) in pixel units.", (int,)),
        Option(
            "vedo", "lc", "Color of edges (lines).", (int, str, tuple, list)
        ),
    )

    _helps = "Volumes"

    def _initialize_showable(self):
        """
        Initialize volumes as vedo.UGrid or visually equivalent vedo.Mesh

        Parameters
        ----------
        None

        Returns
        -------
        volumes: vedo.UGrid or vedo.Mesh
        """
        # without a data to plot on the surface, return vedo.UGrid
        # if vertex_ids is on, we will go with mesh
        if (
            self.get("data", None) is None
            and not self.get("vertex_ids", False)
            and not self.get("arrow_data", False)
            and not show.is_ipython
        ):
            from vtk import VTK_HEXAHEDRON as herr_hexa
            from vtk import VTK_TETRA as frau_tetra

            to_vtktype = {"tet": frau_tetra, "hexa": herr_hexa}
            grid_type = to_vtktype[self._helpee.whatami]
            u_grid = show.vedoUGrid(
                [
                    self._helpee.const_vertices,
                    self._helpee.const_volumes,
                    [grid_type] * len(self._helpee.const_volumes),
                ]
            )

            for option in ["lw", "lc"]:
                val = self.get(option, False)
                if val:
                    getattr(u_grid, option)(val)

            return u_grid.c("hotpink")

        # to show data, let's use Faces. This will plot all the elements
        # as well as invisible ones. This will at least try to avoid
        # duplicating faces.  If you wanna see inside faces, try
        # as_shrunk_faces = volumes.to_faces(unique=False).shrink(.8)
        faces = self._helpee.to_faces(unique=True)
        self.copy_valid_options(faces.show_options)

        return faces.show_options._initialize_showable()


class Volumes(Faces):
    kind = "volume"

    const_faces = helpers.raise_if.invalid_inherited_attr(
        "Faces.const_faces",
        __qualname__,
        property_=True,
    )
    update_faces = helpers.raise_if.invalid_inherited_attr(
        "Faces.update_edges",
        __qualname__,
        property_=False,
    )

    __slots__ = (
        "_volumes",
        "_const_volumes",
    )

    __show_option__ = VolumesShowOption
    __boundary_class__ = Faces

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
        self._volumes = helpers.data.make_tracked_array(
            vols,
            settings.INT_DTYPE,
            copy=False,
        )
        if vols is not None:
            utils.arr.is_one_of_shapes(
                vols,
                ((-1, 4), (-1, 8)),
                strict=True,
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
          valid attributes are {values, ids, inverse, counts}
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

    def to_faces(self, unique=True):
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
            faces=self.unique_faces().values if unique else self.faces(),
        )
