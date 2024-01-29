"""gustaf/gustaf/_base.py.

Useful base class for gustaf. Plus some useful decorators.
"""

from gustaf.utils import log


class GustafBase:
    """Base class for gustaf, where logging is nicely wrapped, and some useful
    methods are defined as classmethods..
    """

    __slots__ = ("__weakref__",)

    def __init_subclass__(cls, *args, **kwargs):
        """
        Add logger shortcut.
        """
        super().__init_subclass__(*args, **kwargs)
        cls._logi = log.prepended_log("<" + cls.__qualname__ + ">", log.info)
        cls._logd = log.prepended_log("<" + cls.__qualname__ + ">", log.debug)
        cls._logw = log.prepended_log(
            "<" + cls.__qualname__ + ">", log.warning
        )
