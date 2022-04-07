"""gustaf/gustaf/vertices.py

Vertices. Base of all "Mesh" geometries.
"""

import copy

import numpy as np

from gustaf import settings
from gustaf import utils
from gustaf import show
from gustaf._base import GustavBase

class Vertices(GustavBase):

    kind = "vertex"

    __slots__ = [
        "whatami",
        "vertices",
        "vertices_unique",
        "vertices_unique_id",
        "vertices_unique_inverse",
        "vertices_overlapping",
        "bounds",
        "centers",
        "vis_dict",
        "vertexdata",
    ]

    def __init__(
            self,
            vertices=None,
    ):
        """
        Vertices. It has vertices.

        Parameters
        -----------
        vertices: (n, d) np.ndarray

        Returns
        --------
        None
        """
        if vertices is not None:
            self.vertices = utils.arr.make_c_contiguous(
                vertices,
                settings.FLOAT_DTYPE
            )
        self.whatami = "vertices"
        self.vis_dict = dict()
        self.vertexdata = dict()

    def process(
            self,
            vetices_unique=False,
            vertices_unique_id=False,
            vertices_unique_inverse=False,
            bounds=False,
            centers=False,
            everything=False,
    ):
        """
        Returns unique vertices.

        Parameters
        -----------
        None

        Returns
        --------
        """
        self.vertices = utils.make_c_contiguous(
            self.vertices,
            settings.FLOAT_DTYPE
        )

        if (
            vertices_unqiue
            or vertices_unique_id
            or vertices_unique_inverse
            or everything
        ):
            self.vertices_unique()

        if bounbds or everything:
            self.bounds()

        if centers or everything:
            self.centers()

    def elements(self, elements=None):
        """
        Returns current elements.
        Elements mean different things for different classes:
          Vertices -> vertices
          Edges -> edges
          Faces -> faces
          Volumes -> volumes
        Could be understood as connectivity.
        Inplace updates only.

        Parameters
        -----------
        elements: (n, d) np.ndarray
          int. Only updates if it is not None.

        Returns
        --------
        elements: (n, d) np.ndarray
          int. iff elements=None
        """
        if hasattr(self, "volumes"):
            if elements is None:
                return self.volumes
            else:
                self.volumes = elements

        elif hasattr(self, "faces"):
            if elements is None:
                return self.faces
            else:
                self.faces = elements

        elif hasattr(self, "edges"):
            if elements is None:
                return self.edges
            else:
                self.edges = elements

        elif hasattr(self, "vertices"):
            return np.arange(
                (self.vertices.shape[0], 1),
                dtype=settings.INT_DTYPE
            )

        else:
            return None

    def get_vertices_unique(
            self,
            tolerance=settings.TOLERANCE,
            referenced_only=True,
            return_referenced=False,
            workers=1,
    ):
        """
        Finds unique vertices using KDTree from scipy.
        TODO: use jjcpp unique

        Parameters
        -----------
        tolerance: float
          Default is settings.TOLERANCE.
        referenced_only: bool
          Only search for unique for referenced vertices. Default is True.
        return_referenced: bool
          Default is False.
        workers: int
          n_jobs for parallel processing. Default is 1.
          -1 uses all processes.

        Returns
        --------
        vertices_unique: (n, d) np.ndarray
          float.
        referenced: (len(self.vertices),) np.ndarray
          bool.
        """
        from scipy.spatial import cKDTree as KDTree

        # Get referenced vertices if it is not vertices and desired
        # to avoid unnecessary computation
        referenced = np.empty(len(self.vertices), dtype=bool)
        if self.kind != "vertex" and referenced_only:
            referenced[self.elements()] = True

        else:
            referenced[:] = True

        # Build kdtree
        kdt = KDTree(self.vertices[referenced])

        # Ball point query, taking tolerance as radius
        neighbors = kdt.query_ball_point(
            self.vertices[referenced],
            tolerance,
            #workers=workers,
            #return_sorted=True # new in 1.6, but default is True, so pass.
        )

        # inverse based on original vertices.
        o_inverse = np.array(
            [n[0] for n in neighbors],
            dtype=settings.INT_DTYPE,
        )

        # unique of o_inverse, and inverse based on that
        (_, uniq_id, inv) = np.unique(
            o_inverse,
            return_index=True,
            return_inverse=True,
        )

        # Save 
        self.vertices_unique = self.vertices[uniq_id]
        self.vertices_unique_id = uniq_id
        self.vertices_unique_inverse = inv
        self.vertices_overlapping = neighbors#.tolist() # array of lists.

        if not return_referenced:
            return self.vertices_unique

        else:
            return self.vertices_unique, referenced

    def get_vertices_unique_id(
            self,
            tolerance=settings.TOLERANCE,
            referenced_only=True,
            return_referenced=False,
            workers=1,
    ):
        """
        Returns ids of unique vertices.

        Parameters
        -----------
        tolerance: float
          Default is settings.TOLERANCE.
        referenced_only: bool
          Only search for unique for referenced vertices. Default is True.
        return_referenced: bool
          Default is False.
        workers: int
          n_jobs for parallel processing. Default is 1.
          -1 uses all processes.

        Returns
        --------
        vertices_unique_id: (n,) np.ndarray
          int
        referenced: (len(self.vertices),) np.ndarray
          bool. iff return_referenced==True
        """
        # last_item_is_ref maybe np.ndarray or tuple
        # tuple, iff return_referenced==True
        last_item_is_ref = self.get_vertices_unique(
                tolerance=tolerance,
                referenced_only=referenced_only,
                return_referenced=return_referenced,
                workers=workers,
        )

        if return_referenced:
            return self.vertices_unique_id, last_item_is_ref[-1]

        else:
            return self.vertices_unique_id

    def get_vertices_unique_inverse(
            self,
            tolerance=settings.TOLERANCE,
            referenced_only=True,
            return_referenced=False,
            workers=1,
    ):
        """
        Returns ids that can ber used to reconstruct vertices with unique
        vertices.

        Parameters
        -----------
        tolerance: float
          Default is settings.TOLERANCE.
        referenced_only: bool
          Only search for unique for referenced vertices. Default is True.
        return_referenced: bool
          Default is False.
        workers: int
          n_jobs for parallel processing. Default is 1.
          -1 uses all processes.

        Returns
        --------
        vertices_unique_id: (n,) np.ndarray
          int
        referenced: (len(self.vertices),) np.ndarray
          bool. iff return_referenced==True
        """
        # last_item_is_ref maybe np.ndarray or tuple
        # tuple, iff return_referenced==True
        last_item_is_ref = self.get_vertices_unique(
                tolerance=tolerance,
                referenced_only=referenced_only,
                return_referenced=return_referenced,
                workers=workers,
        )

        if return_referenced:
            return self.vertices_unique_inverse, last_item_is_ref[-1]

        else:
            return self.vertices_unique_inverse

    def get_vertices_overlapping(
            self,
            tolerance=settings.TOLERANCE,
            referenced_only=True,
            return_referenced=False,
            workers=1,
    ):
        """
        Returns list of ids that overlapps with current vertices.
        Includes itself.

        Parameters
        -----------
        tolerance: float
          Default is settings.TOLERANCE.
        referenced_only: bool
          Only search for unique for referenced vertices. Default is True.
        return_referenced: bool
          Default is False.
        workers: int
          n_jobs for parallel processing. Default is 1.
          -1 uses all processes.

        Returns
        --------
        self.vertices_overlapping: (len(self.vertices)) np.ndarray
          list 
        """
        last_item_is_ref = self.get_vertices_unique(
                tolerance=tolerance,
                referenced_only=referenced_only,
                return_referenced=return_referenced,
                workers=workers,
        )

        if return_referenced:
            return self.vertices_overlapping, last_item_is_ref[-1]

        else:
            return self.vertices_overlapping

    def get_bounds(self):
        """
        Returns bounds of the vertices.

        Parameters
        -----------
        None

        Returns
        --------
        bounds: (d,) np.ndarray
        """
        self.bounds = utils.arr.bounds(self.vertices)

        return self.bounds

    def get_centers(self):
        """
        Center of elements.

        Parameters
        -----------
        None

        Returns
        --------
        centers: (n_elements, d) np.ndarray
        """
        elements = self.elements()
        self.centers = self.vertices[elements].mean(axis=1)

        return self.centers

    def update_vertices(self, mask, inverse=None, inplace=True):
        """
        Update vertices with a mask.
        In other words, keeps only masked vertices.
        Adapted from `github.com/mikedh/trimesh`.

        Parameters
        -----------
        mask: (n,) bool or int
        inverse: (len(self.vertices),) int
        inplace: bool

        Returns
        --------
        updated_self: type(self)
          iff inplace==Ture
        """
        vertices = self.vertices.copy()

        # make mask numpy array
        mask = np.asarray(mask)

        if (
            (mask.dtype.name == "bool" and mask.all())
            or len(mask) == 0
        ):
            # Nothing to do if inplace
            if inplace:
                return None

            # if not inplace, it is same as copy
            else:
                return self.copy()

        # create inverse mask if not passed
        if inverse is None:
            inverse = np.zeros(len(vertices), dtype=settings.INT_DTYPE)
            if mask.dtype.kind == "b":
                inverse[mask] = np.arange(mask.sum())
            elif mask.dtype.kind == "i":
                inverse[mask] = np.arange(len(mask))
            else:
                inverse = None

        # re-index elements from inverse
        # TODO: Here could be a good place to preserve BCs.
        elements = None
        if inverse is not None and self.kind != "vertex":
            elements = self.elements().copy()
            elements = inverse[elements.reshape(-1)].reshape(
                (-1, elements.shape[1])
            )

        # apply mask
        vertices = vertices[mask]

        def update_vertexdata(obj, m, vertex_data=None):
            """
            apply mask to vertex data if there's any.
            """
            newdata = dict()
            if vertex_data is None:
                vertex_data = obj.vertexdata

            for key, values in vertex_data.items():
                newdata[key] = values[m]

            obj.vertexdata = newdata

            return obj


        # update
        if inplace:
            self.vertices = vertices
            if elements is not None:
                self.elements(elements)

            update_vertexdata(self, mask)

            return None

        else:
            if elements is None:
                updated = type(self)(vertices=vertices)
                update_vertexdata(updated, mask, self.vertexdata)

                return updated

            else:
                updated = type(self)(vertices=vertices, elements=elements)
                update_vertexdata(updated, mask, self.vertexdata)

                return updated

    def select_vertices(self, ranges):
        """
        Returns vertices inside the given range.

        Parameters
        -----------
        ranges: (d, 2) array-like
          Takes None.

        Returns
        --------
        ids: (n,) np.ndarray
        """
        return utils.arr.select_with_ranges(self.vertices, ranges)

    def remove_vertices(self, ids, inplace=True):
        """
        Removes vertices with given vertex ids.

        Parameters
        -----------
        ids: (n,) np.ndarray
        inplace: bool

        Returns
        --------
        new_self: type(self)
          iff inplace=True.
        """
        mask = np.ones(len(self.vertices), dtype=bool)
        mask[ids] = False

        return self.update_vertices(mask, inplace=inplace)

    def referenced_vertices(self,):
        """
        Returns mask of referenced vertices.

        Parameters
        -----------
        None

        Returns
        --------
        referenced: (n,) np.ndarray
        """
        referenced = np.zeros(len(self.vertices), dtype=bool)
        referenced[self.elements()] = True

        return referenced

    def remove_unreferenced_vertices(self, inplace=True):
        """
        Remove unreferenced vertices.
        Adapted from `github.com/mikedh/trimesh`

        Parameters
        -----------
        inplace: bool

        Returns
        --------
        new_self: type(self)
          iff inplace=True.
        """
        if self.kind == "vertex":
            return self

        referenced = self.referenced_vertices()

        inverse = np.zeros(len(self.vertices), dtype=settings.INT_DTYPE)
        inverse[referenced] = np.arange(referenced.sum())

        return self.update_vertices(
            mask=referenced,
            inverse=inverse,
            inplace=inplace,
        )

    def merge_vertices(
            self,
            tolerance=settings.TOLERANCE,
            referenced_only=True,
            inplace=True
    ):
        """
        Based on vertices unique, merge vertices if it is mergeable.

        Parameters
        -----------
        tolerance: float
          Default is settings.TOLERANCE
        referenced_only: bool
          Default is True.
        inplace: bool
          Default is True.

        Returns
        --------
        merged_self: type(self)
          iff inplace==True
        """
        inv, referenced = self.get_vertices_unique_inverse(
            tolerance=tolerance,
            referenced_only=referenced_only,
            return_referenced=True,
            workers=1,
        )

        # inverse mask for update
        inverse = np.zeros(len(self.vertices), dtype=settings.INT_DTYPE)
        inverse[referenced] = inv
        # Turn bool mask into int mask and extract only required
        mask = np.nonzero(referenced)[0][self.vertices_unique_id]

        self._logd("number of vertices")
        self._logd(f"  before merge: {len(self.vertices)}")
        self._logd(f"  after merge: {len(mask)}")

        return self.update_vertices(
            mask=mask,
            inverse=inverse,
            inplace=inplace
        )

    def showable(self, **kwargs):
        """
        Returns showable object, meaning object of visualization backend.

        Parameters
        -----------
        **kwargs:

        Returns
        --------
        showable: obj
          Obj of `gustaf.settings.VISUALIZATION_BACKEND`
        """
        return show.make_showable(self, **kwargs)

    def show(self, **kwargs):
        """
        Show current object using visualization backend.

        Parameters
        -----------
        **kwargs:


        Returns
        --------
        None          
        """
        show.show(self, **kwargs)

    def copy(self):
        """
        Returns deepcopy of self.

        Parameters
        -----------
        None

        Returns
        --------
        selfcopy: type(self)
        """
        # all attributes are deepcopy-able
        return copy.deepcopy(self)


    @classmethod
    def concat(cls, *instances):
        """
        Sequentially put them together to make one object.

        Parameters
        -----------
        *instances: *type(cls)
          Allows one iterable object also.

        Returns
        --------
        one_instance: type(cls)
        """
        def is_concatable(inst):
            """
            Return true, if it is same as type(cls)
            """
            if cls.__name__.startswith(inst.__class__.__name__):
                return True
            else:
                return False

        # If only one instance is given and it is iterable, adjust
        # so that we will just iterate that.
        if (
            len(instances) == 1
            and not isinstance(instances[0], str)
            and hasattr(instances[0], "__iter__")
        ):
            instances = instances[0]

        vertices = []
        haselem = cls.kind != "vertex"
        if haselem:
            elements = []

        # check if everything is "concatable".
        for ins in instances:
            if not is_concatable(ins):
                raise TypeError(
                    "Can't concat. One of the instances is not "
                    f"`{cls.__name__}`."
                )

            # make sure each element index starts from 0 & end at len(vertices)
            tmp_ins = ins.remove_unreferenced_vertices(inplace=False)

            vertices.append(
                tmp_ins.vertices.copy()
            )

            if haselem:
                if len(elements) == 0:
                    elements.append(
                        tmp_ins.elements().copy()
                    )
                    e_offset = elements[-1].max() + 1

                else:
                    elements.append(
                        tmp_ins.elements().copy() + e_offset
                    )
                    e_offset = elements[-1].max() + 1

        if haselem:
            return cls(
                vertices=np.vstack(vertices),
                elements=np.vstack(elements),
            )

        else:
            return Vertices(
                vertices=np.vstack(vertices),
            )

    def __add__(self, to_add):
        """
        Concat in form of +.

        Parameters
        -----------
        to_add: type(self)

        Returns
        --------
        added: type(self)
        """
        return type(self).concat(self, to_add)
