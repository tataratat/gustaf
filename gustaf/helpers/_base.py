"""gustaf/gustaf/helpers/_base.py

Base class for helper
"""

from typing import Any as _Any
from weakref import ref

from gustaf._base import GustafBase as _GustafBase


class HelperBase(_GustafBase):
    """
    Minimal base layer for helper classes to avoid cyclic referencing.
    Instead of saving a pure reference to helpee object, this will create a
    weakref instead. This class defines `_helpee` setters and getters, so that
    existing code can remain untouched.
    """

    __slots__ = ("_helpee_weak_ref",)

    @property
    def _helpee(self) -> _Any:
        """Returns dereferenced weak ref. Setter will create and save weakref
        of a helpee."""
        return self._helpee_weak_ref()

    @_helpee.setter
    def _helpee(self, helpee: _Any) -> None:
        self._helpee_weak_ref = ref(helpee)
