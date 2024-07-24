"""gustaf/gustaf/helpers/data.py.

Helps helpee to manage data. Some useful data structures.
"""

from collections import namedtuple
from functools import wraps

import numpy as np

from gustaf.helpers._base import HelperBase


class TrackedArray(np.ndarray):
    """numpy array object that keeps mirroring inplace changes to the source.
    Meant to help control_points.
    """

    __slots__ = (
        "_super_arr",
        "_modified",
    )

    def __array_finalize__(self, obj):
        """Sets default flags for any arrays that maybe generated based on
        physical space array. For more information,
        see https://numpy.org/doc/stable/user/basics.subclassing.html"""
        self._super_arr = None
        self._modified = True

        # for arrays created based on this subclass
        if isinstance(obj, type(self)):
            # this is copy. nothing to worry here
            if self.base is None:
                return None

            # first child array
            if self.base is obj:
                # make sure this is not a recursively born child
                # for example, `arr[[1,2]][:,2]`
                # we should have set _super_arr to True
                # if we made this array using `make_tracked_array`
                if obj._super_arr is True:
                    self._super_arr = obj
                return None

            # multi generation child array
            if obj._super_arr is not None and self.base is obj.base:
                self._super_arr = obj._super_arr
                return None

            return None

    @property
    def modified(self):
        """
        Modified flag getter
        """
        # have super arr and self is not super_arr,
        if self._super_arr is not None and self._super_arr is not True:
            return self._super_arr._modified

        return self._modified

    @modified.setter
    def modified(self, m):
        if self._super_arr is not None and self._super_arr is not True:
            self._super_arr._modified = m
        else:
            self._modified = m

    def copy(self, *args, **kwargs):
        """copy creates regular numpy array"""
        return np.array(self, *args, copy=True, **kwargs)

    def view(self, *args, **kwargs):
        """Set writeable flags to False for the view."""
        v = super(self.__class__, self).view(*args, **kwargs)
        v.flags.writeable = False
        return v

    def __iadd__(self, *args, **kwargs):
        sr = super(self.__class__, self).__iadd__(*args, **kwargs)
        self.modified = True
        return sr

    def __isub__(self, *args, **kwargs):
        sr = super(self.__class__, self).__isub__(*args, **kwargs)
        self.modified = True
        return sr

    def __imul__(self, *args, **kwargs):
        sr = super(self.__class__, self).__imul__(*args, **kwargs)
        self.modified = True
        return sr

    def __idiv__(self, *args, **kwargs):
        sr = super(self.__class__, self).__idiv__(*args, **kwargs)
        self.modified = True
        return sr

    def __itruediv__(self, *args, **kwargs):
        sr = super(self.__class__, self).__itruediv__(*args, **kwargs)
        self.modified = True
        return sr

    def __imatmul__(self, *args, **kwargs):
        sr = super(self.__class__, self).__imatmul__(*args, **kwargs)
        self.modified = True
        return sr

    def __ipow__(self, *args, **kwargs):
        sr = super(self.__class__, self).__ipow__(*args, **kwargs)
        self.modified = True
        return sr

    def __imod__(self, *args, **kwargs):
        sr = super(self.__class__, self).__imod__(*args, **kwargs)
        self.modified = True
        return sr

    def __ifloordiv__(self, *args, **kwargs):
        sr = super(self.__class__, self).__ifloordiv__(*args, **kwargs)
        self.modified = True
        return sr

    def __ilshift__(self, *args, **kwargs):
        sr = super(self.__class__, self).__ilshift__(*args, **kwargs)
        self.modified = True
        return sr

    def __irshift__(self, *args, **kwargs):
        sr = super(self.__class__, self).__irshift__(*args, **kwargs)
        self.modified = True
        return sr

    def __iand__(self, *args, **kwargs):
        sr = super(self.__class__, self).__iand__(*args, **kwargs)
        self.modified = True
        return sr

    def __ixor__(self, *args, **kwargs):
        sr = super(self.__class__, self).__ixor__(*args, **kwargs)
        self.modified = True
        return sr

    def __ior__(self, *args, **kwargs):
        sr = super(self.__class__, self).__ior__(*args, **kwargs)
        self.modified = True
        return sr

    def __setitem__(self, key, value):
        # set first. invalid setting will cause error
        sr = super(self.__class__, self).__setitem__(key, value)
        self.modified = True
        return sr


def make_tracked_array(array, dtype=None, copy=True):
    """Motivated by nice implementations of `trimesh` (see LICENSE.txt).
    `https://github.com/mikedh/trimesh/blob/main/trimesh/caching.py`.

    Factory-like wrapper function for TrackedArray.
    If you want to use TrackedArray, it is recommended to use this function.

    Parameters
    ------------
    array: array- like object
      To be turned into a TrackedArray
    dtype: np.dtype
      Which dtype to use for the array
    copy: bool
      Default is True. copy if True.

    Returns
    ------------
    tracked : TrackedArray
      Contains input array data
    """
    # if someone passed us None, just create an empty array
    if array is None:
        array = []

    if copy:
        array = np.array(array, dtype=dtype)
    else:
        array = np.asanyarray(array, dtype=dtype)

    tracked = array.view(TrackedArray)

    # this marks original array
    tracked._super_arr = True

    return tracked


class DataHolder(HelperBase):
    __slots__ = ("_saved",)

    def __init__(self, helpee):
        """Base class for any data holder. Behaves similar to dict.

        Parameters
        -----------
        helpee: object
          GustafBase objects would probably make the most sense here.
        """
        self._helpee = helpee
        self._saved = {}

    def __setitem__(self, key, value):
        """Raise Error to disable direct value setting.

        Parameters
        -----------
        key: str
        value: object
        """
        raise NotImplementedError(
            "Sorry, you can't set items directly for "
            f"{type(self).__qualname__}"
        )

    def __getitem__(self, key):
        """Returns stored item if the key exists.

        Parameters
        -----------
        key: str

        Returns
        --------
        value: object
        """
        if key in self._saved:
            return self._saved[key]

        else:
            raise KeyError(f"`{key}` is not stored for {type(self._helpee)}")

    def __contains__(self, key):
        """Returns if saved data contains the given key.

        Parameters
        ----------
        key: str

        Returns
        -------
        value
        """
        return key in self._saved

    def __len__(self):
        """
        Returns number of items.

        Parameters
        ----------
        None

        Returns
        -------
        len: int
        """
        return len(self._saved)

    def pop(self, key, default=None):
        """
        Applied pop() to saved data

        Parameters
        ----------
        key: str
        default: object

        Returns
        -------
        value: object
        """
        return self._saved.pop(key, default)

    def clear(self):
        """
        Clears saved data by reassigning new dict
        """
        self._saved = {}

    def get(self, key, default_values=None):
        """Returns stored item if the key exists. Else, given default value. If
        the key exist, default value always exists, since it is initialized
        that way.

        Parameters
        -----------
        key: str
        default_values: object

        Returns
        --------
        value: object
        """
        if key in self._saved:
            return self._saved[key]
        else:
            return default_values

    def keys(self):
        """Returns keys of data holding dict.

        Returns
        --------
        keys: dict_keys
        """
        return self._saved.keys()

    def values(self):
        """Returns values of data holding dict.

        Returns
        --------
        values: dict_values
        """
        return self._saved.values()

    def items(self):
        """Returns items of data holding dict.

        Returns
        --------
        values: dict_values
        """
        return self._saved.items()

    def update(self, **kwargs):
        """
        Updates given kwargs using __setitem__.

        Parameters
        ----------
        **kwargs: kwargs

        Returns
        -------
        None
        """
        self._saved.update(**kwargs)


class ComputedData(DataHolder):
    _depends = None
    _inv_depends = None

    __slots__ = ()

    def __init__(self, helpee, **_kwargs):
        """Stores last computed values.

        Keys are expected to be the same as helpee's function that computes the
         value.

        Parameters
        -----------
        helpee: GustafBase
        """
        super().__init__(helpee)

    @classmethod
    def depends_on(cls, var_names, make_property=False):
        """Decorator as classmethod.

        checks if the key should be computed. Two cases, where the answer is
        yes:

        1. there's modification on arrays that the key depend on.
            ->erases all other
        2. is corresponding value None?

        Supports multi-dependency

        Parameters
        -----------
        var_name: list
        make_property:
        """

        def inner(func):
            # following are done once while modules are loaded
            # just subclass this class to make a special helper
            # for each helpee class.
            assert isinstance(var_names, list), "var_names should be a list"
            # initialize property
            # _depends is dict(str: list)
            if cls._depends is None:
                cls._depends = {}
            if cls._depends.get(func.__name__, None) is None:
                cls._depends[func.__name__] = []
            # add dependency info
            cls._depends[func.__name__].extend(var_names)

            # _inv_depends is dict(str: list)
            if cls._inv_depends is None:
                cls._inv_depends = {}
            # add inverse dependency
            for vn in var_names:
                if cls._inv_depends.get(vn, None) is None:
                    cls._inv_depends[vn] = []

                cls._inv_depends[vn].append(func.__name__)

            @wraps(func)
            def compute_or_return_saved(*args, **kwargs):
                """Check if the key should be computed,"""
                # extract some related info
                self = args[0]  # the helpee itself

                # explicitly settable kwargs.
                # unless recompute flag is set False,
                # it will always recompute and save them
                # if you call the same function without kwargs
                # the last one with kwargs will be returned
                recompute = False
                if kwargs:
                    recompute = kwargs.get("recompute", True)

                # computed arrays are called _computed.
                # loop over dependencies and check if they are modified
                for dependee_str in cls._depends[func.__name__]:
                    dependee = getattr(self, dependee_str)
                    # is modified?
                    if dependee._modified:
                        for inv in cls._inv_depends[dependee_str]:
                            self._computed._saved[inv] = None

                # is saved / want to recompute?
                # recompute is added for computed values that accepts params.
                saved = self._computed._saved.get(func.__name__, None)
                if saved is not None and not recompute:
                    return saved

                # we've reached this point because we have to compute this
                computed = func(*args, **kwargs)
                if isinstance(computed, np.ndarray):
                    computed.flags.writeable = False  # configurable?
                self._computed._saved[func.__name__] = computed

                # so, all fresh. we can press NOT-modified  button
                for dependee_str in cls._depends[func.__name__]:
                    dependee = getattr(self, dependee_str)
                    dependee._modified = False

                return computed

            if make_property:
                return property(compute_or_return_saved)
            else:
                return compute_or_return_saved

        return inner


class VertexData(DataHolder):
    """
    Minimal manager for vertex data. Checks input array size, transforms
    data on request. __setitem__ and __getitem__ will perform length checks.
    key(), values(), items(), and get() will return whatever is currently
    stored.

    gustaf supports two kinds of data representation: scalar-data with cmap
    and vector-data with arrows.
    """

    __slots__ = ()

    def __init__(self, helpee):
        """Checks if helpee has vertices as attr beforehand.

        Parameters
        ----------
        helpee: Vertices
          Vertices and its derived classes.
        """
        if not hasattr(helpee, "vertices"):
            raise AttributeError("Helpee does not have `vertices`.")

        super().__init__(helpee)

    def _validate_len(self, value=None, raise_=True):
        """Checks if given value is a valid vertex_data based of its length.

        If raise_, throws error, else, deletes all incompatible values.
        Only checks len(). If array has (1, len) shape, this will still return
        False.

        Parameters
        ----------
        value: array-like
          Default is None. If None, checks all existing values.
        raise_: bool
          Default is True, If True, raises in case of incompatibility.

        Returns
        -------
        validity: bool
          If raise_ is False.
        """
        valid = True
        helpee_len = len(self._helpee.vertices)
        if value is not None:
            if len(value) != helpee_len:
                valid = False

            if raise_ and not valid:
                raise ValueError(
                    f"Expected ({helpee_len}) length data, "
                    f"Given ({len(value)})"
                )

            return valid

        # here, check all saved values.
        to_pop = []
        for key, d_value in self._saved.items():
            if len(d_value) != helpee_len:
                valid = False

            if not valid:
                if raise_:
                    raise ValueError(
                        f"`{key}`-data len ({len(d_value)}) doesn't match "
                        f"expected len ({helpee_len})"
                    )
                else:
                    self._logd(
                        f"`{key}`-data len ({len(d_value)}) doesn't match "
                        f"expected len ({helpee_len}). Deleting `{key}`."
                    )
                # pop invalid data
                to_pop.append(key)
                to_pop.append(key + "__norm")

        # pop if needed
        for tp in to_pop:
            self._saved.pop(tp, None)

        return valid

    def __setitem__(self, key, value):
        """
        Performs len() based check before storing vertex_data.

        Parameters
        ----------
        key: str
        value: object

        Returns
        -------
        None
        """
        self._validate_len(value, raise_=True)

        # we are here because this is valid
        self._saved[key] = make_tracked_array(value, copy=False).reshape(
            len(self._helpee.vertices), -1
        )

        # if "data" or "arrow_data" is empty in show_options, we want to
        # set this data to show. We will always set this as "data".
        show_options = getattr(self._helpee, "show_options", None)
        if show_options is not None:
            if "data" in show_options or "arrow_data" in show_options:
                return None
            show_options["data"] = key

    def __getitem__(self, key):
        """
        Validates data length before returning item.

        Parameters
        ----------
        key: str

        Returns
        -------
        data: array-like
        """
        value = super().__getitem__(key)  # raises KeyError
        valid = self._validate_len(value, raise_=False)
        if valid:
            return value
        else:
            raise KeyError(
                "Either requested data is not stored or deleted due to "
                "changes in number of vertices."
            )

    def as_scalar(self, key, default=None):
        """
        Returns scalar version of requested data. If it is already a scalar,
        will return as is. Else, will return a norm. using `np.linalg.norm()`.

        Parameters
        ----------
        key: str
        default: object

        Returns
        -------
        data_as_scalar: (n, 1) np.ndarray
        """
        if key not in self.keys():
            return default

        # interpret scalar as norm
        # save the norm once it is called.
        if "__norm" not in key:
            norm_key = key + "__norm"
        else:
            norm_key = key
            key = key.replace("__norm", "")

        if norm_key in self.keys():
            saved = self[norm_key]  # performs len check
            # return if original is not modified
            if not self[key]._modified:  # check if original data is modified
                return saved
            else:
                self._saved.pop(norm_key)

        # we are here because we have to compute norm. let's save norm
        value = self[key]
        if value.shape[1] == 1:
            value_norm = value
        else:
            value_norm = np.linalg.norm(value, axis=1).reshape(-1, 1)

        # save norm
        self[norm_key] = value_norm
        # considered not modified
        self[key]._modified = False

        return value_norm

    def as_arrow(self, key, default=None, raise_=True):
        """
        Returns an array as is, only if it is showable as arrow.

        Parameters
        ----------
        key: str
        default: object
        raise_: bool

        Returns
        -------
        data: (n, d) np.ndarray
        """
        if key not in self.keys():
            return default

        value = self[key]
        if value.shape[1] == 1:
            self._logd(f"as_arrow() requested data ({key}) is 1D data.")
            if raise_:
                raise ValueError(
                    f"`{key}`-data is 1D and cannot be represented as arrows."
                )

        return value


Unique2DFloats = namedtuple(
    "Unique2DFloats", ["values", "ids", "inverse", "intersection"]
)
Unique2DFloats.__doc__ = """
namedtuple to hold unique information of float type arrays.
Note that for float types, "close enough" might be a better name than unique.
This way, all tracked arrays, as long as they are 2D, have a dot separated
syntax to access unique info. For example, `mesh.unique_vertices.ids`.
"""
Unique2DFloats.values.__doc__ = """`(n, d) np.ndarray`
    Field number 0"""
Unique2DFloats.ids.__doc__ = """`(n, d) np.ndarray`
    Field number 1"""
Unique2DFloats.inverse.__doc__ = """`(n, d) np.ndarray`
    Field number 2"""
Unique2DFloats.intersection.__doc__ = """`(m) list of list`
  given original array's index, returns overlapping arrays, including itself.
  Field number 3
"""

Unique2DIntegers = namedtuple(
    "Unique2DIntegers", ["values", "ids", "inverse", "counts"]
)
Unique2DIntegers.__doc__ = """
namedtuple to hold unique information of integer type arrays.
Similar approach to Unique2DFloats.
"""

Unique2DIntegers.values.__doc__ = """`(n, d) np.ndarray`
    Field number 0"""
Unique2DIntegers.ids.__doc__ = """`(n) np.ndarray`
    Field number 1"""
Unique2DIntegers.inverse.__doc__ = """`(m) np.ndarray`
    Field number 2"""
Unique2DIntegers.counts.__doc__ = """`(n) np.ndarray`
    Field number 3"""


class ComputedMeshData(ComputedData):
    """A class to hold computed-mesh-data.

    Subclassed to keep its own dependency info.
    """

    pass
