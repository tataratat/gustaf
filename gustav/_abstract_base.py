"""gustav/gustav/mesh.py

Mesh.
"""

import abc
import logging
import os

import numpy as np

from gustav import utils

class AB(abc.ABC):
    """
    Abstract (but not really) base class for gustav, where logging is nicely
    wrapped.

    All the classes are expected to have __slots__.
    In this version, the idea of setter, getter and cached is disregarded.

    If some values are going to be used more than once, save it by yourself.

    Since attributes are predefined with __slots__, we can pre define
    all the properties that could have been cached.
    Although leading underscore indicates internal use, feel free to grab it
    it you know that you have no inplace changes.
    One magic call `process` will make all these values available.

    Other more complex operations will be a separate function.
    """

    __slots__ = [
        "whatami",
        "kind",
    ]

    def _logd(self, *log):
        """
        Debug logger wrapper for Mesh.

        Parameters
        -----------
        *log: *str

        Returns
        --------
        None
        """
        utils.log._debug(type(self).__qualname__, "-", *log)

    def _logi(self, *log):
        """
        Info logger wrapper for Mesh.

        Parameters
        -----------
        *log: *str

        Returns
        --------
        None
        """
        utils.log._info(type(self).__qualname__, "-", *log)

    def _logw(self, *log):
        """
        Warning logger wrapper for Mesh.

        Parameters
        -----------
        *log: *str

        Returns
        --------
        None
        """
        utils.log._warning(type(self).__qualname__, "-", *log)
