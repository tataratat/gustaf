"""gustaf/gustaf/helpers/options.py

Classes to help organize options.
"""
from gustaf import settings


class Option:
    """
    Minimal Class to hold each options. Mainly to replace nested dict.
    """
    __slots__ = (
            "backend",
            "key",
            "description",
            "allowed_types",
            "is_init_param",
    )

    def __init__(self, backend, key, description, allowed_types, is_init_param):
        self.backend = backend
        self.key = key
        self.description = description
        self.allowed_types = allowed_types
        # for summarizing init options
        self.is_init_param = is_init_param
 

    def __repr__(self):
        specific = "\n".join(
                [
                        "", self.key, "=" * len(self.key),
                        "backend: " + self.backend, "description:",
                        "  " + str(self.description), "allowed_types:",
                        "  " + str(self.allowed_types),
                        super().__repr__(), ""
                ]
        )
        return specific


# predefine recurring options
vedo_common_options = (
        Option(
                "vedo", "c", "Color in {rgb, RGB, str of (hex, name), int}",
                (str, tuple, list, int), True
        ),
        Option("vedo", "alpha", "Transparency in range [0, 1].", (float, int), True),
        Option(
                "vedo", "dataname", "Name of vertexdata to show. "
                "Object must have vertexdata with the same name.", (str, ), False
        ), Option("vedo", "cmap", "Colormap for vertexdata plots.", (str, ), False),
        Option("vedo", "vmin", "Minimum value for cmap", (float, int), False),
        Option("vedo", "vmax", "Maximum value for cmap", (float, int), False),
        Option(
                "vedo", "cmapalpha", "Colormap Transparency in range [0, 1].",
                (float, int), False
        ),
        Option(
                "vedo", "scalarbar",
                "Scalarbar describing cmap. At least an empty dict or "
                "dict with following items are accepted: "
                "{title: str, pos: tuple, title_yoffset: int, font_size: int, "
                "nlabels: int, c: str, horizontal: bool, use_alpha: bool, "
                "label_format: str}", (dict, ), False
        ),
        Option(
                "vedo", "extra",
                "Additional kwargs to be applied during showable "
                "initialization. For example ones that're not provided by gustaf.",
                (dict, ), True
        )
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
    Options should be described by Option object.
    """
    __slots__ = ("_helpee", "_options", "_backend")

    _valid_options = dict()

    _helps = None

    def __init__(self, helpee):
        """
        Parameters
        ----------
        helpee: object
        """
        self._helpee = helpee  # maybe won't save
        if not type(helpee).__qualname__.startswith(self._helps):
            raise TypeError(
                    f"This show option is for {self._helps}. "
                    f"Given helpee is {type(helpee)}."
            )
        self._options = dict()
        self._backend = settings.VISUALIZATION_BACKEND

        # initialize backend specific option holder
        self._options[self._backend] = dict()

    def __setitem__(self, key, value):
        """
        Sets option after checking its validity.

        Parameters
        ----------
        key: str
        value: object

        Returns
        -------
        None
        """
        if key in self._valid_options[self._backend].keys():
            # valid type check
            if not isinstance(
                    value,
                    self._valid_options[self._backend][key].allowed_types
            ):
                raise TypeError(
                        f"{type(value)} is invalid value type for '{key}'. "
                        f"Details for '{key}':\n"
                        f"{self._valid_options[self._backend][key]}"
                )

            # types are valid. let's add
            self._options[self._backend][key] = value

        elif key.startswith("backend"):
            # special keyword.
            if not isinstance(value, str):
                raise TypeError(
                        f"Invalid backend info ({value}). Must be a str"
                )
            self._backend = value
            if not self._options.get(self._backend, False):
                self._options[self._backend] = dict()

        else:
            pass

    def __getitem__(self, key):
        """
        """
        return self._options[self._backend][key]

    def get(self, key, default):
        return self._options[self._backend].get(key, default)

    def update(self, **kwargs):
        """
        Calls __setitem__ iteratively for validity check.

        Parameters
        ----------
        **kwargs: kwargs

        Returns
        -------
        None
        """
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def keys(self):
        """
        Registered option keys.

        Parameters
        ----------
        None

        Returns
        -------
        keys: dict_keys
        """
        return self._options.keys()

    def values(self):
        """
        Registered option values.

        Parameters
        ----------
        None

        Returns
        -------
        keys: dict_values
        """
        return self._options.values()

    def items(self):
        """
        Registered option items.

        Parameters
        ----------
        None

        Returns
        -------
        items: dict_items
        """
        return self._options.items()

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
        self._options.clear()
        # put back default backend option dict
        self._options[self._backend] = dict()

    def split_options(self):
        """
        Returns 2 dicts of options key and values only required for init and
        that requires further processing

        Parameters
        ----------
        None

        Returns
        -------
        init_options: dict
        after_init_options: dict
        """
        init_options = dict()
        after_init_options = dict()
        for key, value in self._options[self._backend].items():
            if value.is_init_param:
                init_options[key] = value
            else:
                after_init_options[key] = value
