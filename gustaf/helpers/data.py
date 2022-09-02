"""gustaf/gustaf/helpers/saved.py

Stores latest computed values saved values.
"""

import abc

import numpy as np

class TrackedArray(np.ndarray):
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


def to_tracked_array(array, dtype=None):
    """
    Taken from nice implementations of `trimesh` (see LICENSE.txt).
    `https://github.com/mikedh/trimesh/blob/main/trimesh/caching.py`.

    Properly subclass a numpy ndarray to track changes.
    Avoids some pitfalls of subclassing by forcing contiguous
    arrays and does a view into a TrackedArray.

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
        array, dtype=dtype).view(TrackedArray)
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

    @abc.abstractmethod
    def _save(self, key, value):
        """
        Do what __setitem__ would do, but only meant for internal use.
        """
        pass


class LastSavedArrays(DataHolder):

    ___slots___ = ["_valid_keys"]

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
        self._valid_keys = tuple(kwargs.keys())

        # check if helpee has same named methods that's callable
        for key, values in kwargs.items():
            mem_func = getattr(self._helpee, k, None)
            # None init.
            self._saved[key] = None
            if not callable(mem_func):
                raise AttributeError(
                    f"{type(self._helpee)} is expected to have a {f}() method,"
                    " but it doesn't."
                )

    def __setitem__(self, key, value):
        """
        Stores values if the key exists.

        Parameters
        -----------
        key: str
        value: np.ndarray

        Returns
        --------
        None
        """
        if key in self._valid_keys:
            return self._saved[key] = value

        else:
            raise KeyError(
                f"`{key}` cannot be stored for {type(self._helpee)}"
            )

    def 
