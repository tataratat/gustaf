"""gustaf/gustaf/helpers/raise_if.py.

Collection of wrapper functions/classes that raises Error with certain
behavior
"""

from typing import Any, Optional


def invalid_inherited_attr(attr_name, qualname, property_=False):
    """Returns a function that would behave the same as given function, but
    would raise AttributeError. This needs to be defined in class level.

    Parameters
    ----------
    func: (function)
        _description_
    qualname: (class)
        _description_
    property_: (bool, optional)
        is this function a property?. Defaults to False.

    Returns
    -------
    raiser: function
        behaves same as func if `property_` is correctly defined
    """

    def raiser():
        raise AttributeError(
            f"{attr_name} is not supported from {qualname} "
            "and its subclasses thereof."
        )

    if property_:
        return property(raiser)

    else:
        return raiser


class ModuleImportRaiser:
    """Mock imports optional modules if they are not installed.

    Class used to have better import error handling in the case that a
    package package is not installed. This is necessary due to that some
    packages are not a dependency of `gustaf`, but some parts require
    them to function. Examples are `splinepy` and `vedo`.
    """

    def __init__(self, lib_name: str, error_message: Optional[str] = None):
        original_message = ""
        if error_message is not None:
            original_message = f"\nOriginal error message - {error_message}"
        self._message = str(
            f"Cannot load {lib_name} package, on which requested "
            "functionality depends. "
            "Please refer to the installation instructions "
            "[tataratat.github.io/gustaf] for more information."
            f"{original_message}"
        )

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        """Is called when the object is called by object().

        Will notify the user, that the functionality is not accessible
        and how to proceed to access the functionality.
        """
        raise ImportError(self._message)

    def __getattr__(self, __name: str) -> Any:
        """Is called when any attribute of the object is accessed by
        object.attr.

        Will notify the user, that the functionality is not accessible
        and how to proceed to access the functionality.
        """
        if __name == "_ModuleImportRaiser__message":
            return object.__getattr__(self, __name[-8:])
        else:
            raise ImportError(self._message)

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        Is called when any attribute of the object is set by object.attr = new.
        Will notify the user, that the functionality is not accessible and how
        to proceed to access the functionality.
        """
        if __name == "_message":
            object.__setattr__(self, __name, __value)
        else:
            raise ImportError(self._message)

    def __getitem__(self, key):
        """Is called when the object is subscripted object[x].

        Will notify the user, that the functionality is not accessible
        and how to proceed to access the functionality.
        """
        raise ImportError(self._message)
