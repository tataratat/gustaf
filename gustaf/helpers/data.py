"""gustaf/gustaf/helpers/saved.py

Stores latest computed values saved values.
"""

import abc
from functools import wraps

import numpy as np

class _TrackedArray(np.ndarray):
    """
    Taken from nice implementations of `trimesh` (see LICENSE.txt).
    `https://github.com/mikedh/trimesh/blob/main/trimesh/caching.py`.
    Minor adaption, since we don't have hashing functionalities.
    """
    def __array_finalize__(self, obj):
        """
        Sets a modified flag on every TrackedArray
        This flag will be set on every change as well as
        during copies and certain types of slicing.
        """
        self._modified = True
        if isinstance(obj, type(self)):
            obj._modified = True

    @property
    def mutable(self):
        return self.flags['WRITEABLE']

    @mutable.setter
    def mutable(self, value):
        self.flags.writeable = value

    def __iadd__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__iadd__(*args,
                                                    **kwargs)

    def __isub__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__isub__(*args,
                                                    **kwargs)

    def __imul__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__imul__(*args,
                                                    **kwargs)

    def __idiv__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__idiv__(*args,
                                                    **kwargs)

    def __itruediv__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__itruediv__(*args,
                                                        **kwargs)

    def __imatmul__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__imatmul__(*args,
                                                       **kwargs)

    def __ipow__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__ipow__(*args, **kwargs)

    def __imod__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__imod__(*args, **kwargs)

    def __ifloordiv__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__ifloordiv__(*args,
                                                         **kwargs)

    def __ilshift__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__ilshift__(*args,
                                                       **kwargs)

    def __irshift__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__irshift__(*args,
                                                       **kwargs)

    def __iand__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__iand__(*args,
                                                    **kwargs)

    def __ixor__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__ixor__(*args,
                                                    **kwargs)

    def __ior__(self, *args, **kwargs):
        self._modified = True
        return super(self.__class__, self).__ior__(*args,
                                                   **kwargs)

    def __setitem__(self, *args, **kwargs):
        self._modified = True
        super(self.__class__, self).__setitem__(*args,
                                                **kwargs)

    def __setslice__(self, *args, **kwargs):
        self._modified = True
        super(self.__class__, self).__setslice__(*args,
                                                 **kwargs)


def TrackedArray(array, dtype=None):
    """
    Taken from nice implementations of `trimesh` (see LICENSE.txt).
    `https://github.com/mikedh/trimesh/blob/main/trimesh/caching.py`.

    Properly subclass a numpy ndarray to track changes.
    Avoids some pitfalls of subclassing by forcing contiguous
    arrays and does a view into a TrackedArray.

    Factory-like wrapper function for _TrackedArray.

    Parameters
    ------------
    array : array- like object
      To be turned into a TrackedArray
    dtype : np.dtype
      Which dtype to use for the array

    Returns
    ------------
    tracked : TrackedArray
      Contains input array data
    """
    # if someone passed us None, just create an empty array
    if array is None:
        array = []
    # make sure it is contiguous then view it as our subclass
    tracked = np.ascontiguousarray(
        array, dtype=dtype).view(_TrackedArray)
    # should always be contiguous here
    assert tracked.flags['C_CONTIGUOUS']

    return tracked


class DataHolder(abc.ABC):
    __slots__ = ["_helpee", "_saved",]

    def __init__(self, helpee):
        """
        Base class for any data holder.
        Behaves similar to dict.

        Attributes
        -----------
        None

        Parameters
        -----------
        helpee: object
          GustafBase objects would probably make the most sense here.
        """
        self._helpee = helpee
        self._saved = dict()

    def __setitem__(self, key, value):
        """
        Raise Error to disable direct value setting.

        Parameters
        -----------
        key: str
        value: object

        Returns
        --------
        None
        """
        raise NotImplementedError(
            "Sorry, you can't set items directly for "
            f"{type(self).__qualname__}"
        )

    def __getitem__(self, key):
        """
        Returns stored item if the key exists.

        Parameters
        -----------
        key: str

        Returns
        --------
        value: object
        """
        if key in self._saved.keys():
            return self._saved[key]

        else:
            raise KeyError(
                f"`{key}` is not stored for {type(self._helpee)}"
            )

    def get(self, key, default_values=None):
        """
        Returns stored item if the key exists.
        Else, given default value.
        If the key exist, default value always exists, since it is
        initialized that way.

        Parameters
        -----------
        key: str
        default_values: object

        Returns
        --------
        value: np.ndarray
        """
        if key in self._saved.keys():
            return self._saved[key]
        else:
            return default_values

    def keys(self):
        """
        Returns keys of data holding dict.

        Parameters
        -----------
        None

        Returns
        --------
        keys: dict_keys
        """
        return self._saved.keys()

    def values(self):
        """
        Returns values of data holding dict.

        Parameters
        -----------
        None

        Returns
        --------
        values: dict_values
        """
        return self._saved.values()

    def items(self):
        """
        Returns items of data holding dict.

        Parameters
        -----------
        None

        Returns
        --------
        values: dict_values
        """
        return self._saved.items()


class ComputedArrays(DataHolder):
    _depends = None
    _inv_depends = None

    ___slots___ = []

    def __init__(self, helpee, **kwrags):
        """
        Stores last computed values.
        Keys are expected to be the same as helpee's function that computes
        the value.

        Attributes
        -----------
        None

        Parameters
        -----------
        helpee: GustafBase
        kwrags: **kwrags
          keys and str of attributes, on which this array depends
        """ 
        super().__init__(helpee)

    @classmethod
    def depends_on(cls, var_name):
        """
        Decorator as classmethod.

        checks if the key should be computed. Three cases, where the answer is
        yes:
        1. there's modification on arrays that the key depend on.
          -> erases all other
        2. corresponding value is None
        3. ?i just can't remember?

        Parameters
        -----------
        key: str
        """
        def inner(func):
            # followings are done once while modules are loaded
            # just subclass this class to make a special helper
            # for each helpee class.

            # initialize property
            if cls._depends is None:
                cls._depends = dict()

            cls._depends[func.__name__] = var_name

            if cls._inv_depends is None:
                cls._inv_depends = dict()
            if cls._inv_depends.get(var_name, None) is None:
                cls._inv_depends[var_name] = list()

            cls._inv_depends[var_name].append(func.__name__)

            @wraps(func)
            def compute_or_return_saved(*args, **kwargs):
                """
                Check if the key should be computed,
                """
                self = args[0] # the helpee itself
                # computed arrays are called _computed.
                dependee = getattr(self, cls._depends[func.__name__])
                if dependee.is_modified:
                    for invd in cls._inv_depends[cls._depends[func.__name]]:
                        self._computed._saved[invd] = None

                saved = self._computed._saved.get(func.__name__, None)
                if saved is not None:
                    return saved

                # we've reached this point because we have to compute this
                computed = func(*args, **kwargs)
                computed.flags.writable = False # configurable?
                self._computed._saved[func.__name__] = computed

                return computed


class MeshComputedArrays(ComputedArrays): pass
