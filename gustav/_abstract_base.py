"""gustav/gustav/mesh.py

Mesh.
"""

import abc
import logging
import os

import numpy as np

from gustav import utils

class AB(abc.ABC):

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

    def _delete_cached(self):
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
            "Removing cached data:",
            f"{str(self._cached.keys())[10:-1]}"
        )
