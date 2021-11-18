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
    Abstract (but not really) base class for gustav, where logging and property
    / cache calls are nicely wrapped. If a class inherits this `AB`, one must
    declare instance variable `_properties` and `_cached`.

    Purpose of this base functions is to excessively log every action,
    with a hope that it is easier to debug.

    If you don't find any use of `_cached`, don't declare/use.
    If you set `_properties = self.__dict__`. It will just act as normal
    attribute.
    """

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

    def _get_property(self, key):
        """
        Checks if property is defined with given key.

        Parameters
        -----------
        key: str

        Returns
        --------
        property: obj or None
        """
        return utils._dict._get_property(
            self._properties,
            key,
            type(self).__qualname__,
        )

    def _update_property(self, key, value):
        """
        Updates property with given value.

        Parameters
        -----------
        key: str
        value: obj

        Returns
        --------
        None
        """
        utils._dict._update_property(
            self._properties,
            key,
            value,
            type(self).__qualname__,
        )

    def _get_cached(self, key):
        """
        Checks if obj is cached with given key.

        Parameters
        -----------
        key: str

        Returns
        --------
        cached_property: obj or None
        """
        return utils._dict._get_cached(
            self._cached,
            key,
            type(self).__qualname__,
        )

    def _update_cached(self, key, value):
        """
        Updates cached dict with given key and value.

        Parameters
        -----------
        key: str
        value: obj

        Returns
        --------
        None
        """
        utils._dict._update_cached(
            self._cached,
            key,
            value,
            type(self).__qualname__,
        )

    def _clear_cached(self):
        """
        Removes cached data by newly assigning dict.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        self._logd(
            "Clearing cached data:",
            f"{str(self._cached.keys())[10:-1]}"
        )
        self._cached.clear()
