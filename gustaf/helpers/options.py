"""gustaf/gustaf/helpers/options.py

Classes to help organize options.
"""
from gustaf import settings

class Option:
    """
    Minimal Class to hold each options. Mainly to replace nested dict.

    Attributes
    -----------
    description: str
      Description about the option
    allowed_types: tuple
      Acceptable types
    """
    __slots__ = ("backend", "key", "description", "allowed_types")

    def __init__(self, backend, key, description, allowed_types):
        self.backend = backend
        self.key = key
        self.description = description
        self.allowed_types = allowed_types

    def __repr__(self):
        specific = "\n".join([
                self.key,
                "=" * len(self.key),
                "backend: " + self.backend,
                "description:",
                "  " + str(self.description),
                "allowed_types:",
                "  " + str(self.allowed_types),
                super().__repr__(),
        ])


# predefine recurring options
vedo_color_option = Option(
        "vedo",
        "c",
        "Color in {rgb, RGB, str of (hex, name), int}",
        (str, tuple, list, int),
)


vedo_alpha_option = Option(
        "vedo",
        "alpha",
        "Transparency in range [0, 1].",
        (float,)
)


def make_valid_options(*options):
    """
    Forms valid options. Should run only once during module loading.

    Parameters
    ----------
    *options: Option

    Returns
    -------
    valid_options: dict()
    """
    valid_options = dict()
    for opt in options:
        if not isinstance(opt, Option):
            raise TypeError("Please use `Option` to define options.")

        if not valid_options.get(opt.backend, False):
            valid_options[opt.backend] = dict()

        valid_options[opt.backend][opt.key] = opt

    return valid_options


class ShowOption:
    """
    Behaves similar to dict, but will only accept a set of options that's
    applicable to the helpee class. Intented use is to create a
    subclass that would define valid options for helpee.
    Valid options should take a form of:
        { backend_name : {valid_options : (description, allowed_types)}}
    """
    vedo = dict()
    __slots__ = ("_helpee", "_options", "_backend")

    _valid_options = dict()

    def __init__(self, helpee):
        """
        Parameters
        ----------
        helpee: object
        """
        self._helpee = helpee
        self._options = dict()
        self._backend = settings.VISUALIZATION_BACKEND

    def __setitem__(self, key, value):
        if key in self._valid_options.keys():
            if isinstance(value, _valid_options[key]):
                pass
        elif key.startswith("backend"):
            pass
        else:
            pass

    def __getitem__(self, key):
        pass

    def keys(self):
        pass

    def values(self):
        pass

    def items(self):
        pass

    def clear(self):
        """
        Clears all the options.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass
