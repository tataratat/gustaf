"""gustaf/gustaf/_base.py.

Useful base class for gustaf. Plus some useful decorators.
"""
from gustaf.utils import log


class GustafBase:
    """Base class for gustaf, where logging is nicely wrapped, and some useful
    methods are defined as classmethods..

    Since attributes are predefined with __slots__, we can pre define
    all the properties that could have been saved.
    Adding `get_` in front of such properties are names for functions that
    freshly compute and save the properties and their byproducts.

    Other more complex operations will be a separate function.

    TODO: maybe add explicit `use_saved` switch to avoid recomputing
    """

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        """
        Add logger shortcut.
        """
        cls._logi = log.prepended_log("<" + cls.__qualname__ + ">", log.info)
        cls._logd = log.prepended_log("<" + cls.__qualname__ + ">", log.debug)
        cls._logw = log.prepended_log(
            "<" + cls.__qualname__ + ">", log.warning
        )
        return super().__new__(cls)
