"""gustaf/gustaf/helpers/options.py

Classes to help organize options.
"""
from copy import deepcopy

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
    )

    def __init__(self, backend, key, description, allowed_types):
        self.backend = backend
        self.key = key
        self.description = description
        self.allowed_types = allowed_types

    def __repr__(self):
        specific = "\n".join(
            [
                "",
                self.key,
                "=" * len(self.key),
                "backend: " + self.backend,
                "description:",
                "  " + str(self.description),
                "allowed_types:",
                "  " + str(self.allowed_types),
                super().__repr__(),
                "",
            ]
        )
        return specific


# predefine recurring options
vedo_common_options = (
    Option(
        "vedo",
        "c",
        "Color in {rgb, RGB, str of (hex, name), int}",
        (str, tuple, list, int),
    ),
    Option("vedo", "alpha", "Transparency in range [0, 1].", (float, int)),
    Option(
        "vedo",
        "data_name",
        "Name of vertex_data to show. "
        "Object must have vertex_data with the same name.",
        (str,),
    ),
    Option("vedo", "vertex_ids", "Show ids of vertices", (bool,)),
    Option("vedo", "element_ids", "Show ids of elements", (bool,)),
    Option(
        "vedo",
        "lighting",
        "Lighting options {'default', 'metallic', 'plastic', 'shiny', "
        "'glossy', 'ambient', 'off'}",
        (str,),
    ),
    Option("vedo", "cmap", "Colormap for vertex_data plots.", (str,)),
    Option("vedo", "vmin", "Minimum value for cmap", (float, int)),
    Option("vedo", "vmax", "Maximum value for cmap", (float, int)),
    Option(
        "vedo",
        "cmap_alpha",
        "Colormap Transparency in range [0, 1].",
        (float, int),
    ),
    Option(
        "vedo",
        "scalarbar",
        "Scalarbar describing cmap. At least an empty dict or "
        "dict with following items are accepted: "
        "{title: str, pos: tuple, title_yoffset: int, font_size: int, "
        "nlabels: int, c: str, horizontal: bool, use_alpha: bool, "
        "label_format: str}. Setting bool will add a default scalarbar",
        (bool, dict),
    ),
    Option(
        "vedo",
        "scalarbar3d",
        "3D scalarbar describing cmap. At least an empty dict or "
        "dict with following items are accepted: "
        "{title: str, pos: tuple, size: list, title_font: str, "
        "title_xoffset: float, title_yoffset: float, title_size: float, "
        "title_rotation: float, nlabels: int, label_font:str, "
        "label_size: float, label_offset: float, label_rotation: int, "
        "label_format: str, draw_box: bool, above_text: str, below_text: str, "
        "nan_text: str, categories: list}",
        (bool, dict),
    ),
    Option(
        "vedo",
        "arrow_data",
        "Name of vertex_data to plot as arrow. Corresponding data should be "
        "at least 2D. If you want more control over arrows, consider creating "
        "edges using gus.create.edges.from_data().",
        (str,),
    ),
    Option(
        "vedo",
        "arrow_data_scale",
        "Scaling factor for arrow data.",
        (float, int),
    ),
    Option(
        "vedo",
        "arrow_data_color",
        "Color for arrow data. Can be either cmap name or color. For "
        "cmap, colors are based on the size of the arrows.",
        (str, tuple, list, int),
    ),
    Option(
        "vedo",
        "axes",
        "Configure a specific axes with options. Expect dict(), but setting "
        "True will set a default axes. For full options, see "
        "https://vedo.embl.es/autodocs/content/vedo/addons.html"
        "#vedo.addons.Axes",
        (bool, dict),
    ),
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
    applicable to the helpee class. Intended use is to create a
    subclass that would define valid options for helpee.
    Options should be described by Option object.
    Helps all the way up to initializing backend showables up to their backend
    specific common routines. ShowOption and ShowManager in a sense.
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
        self._helpee = helpee
        if not type(helpee).__qualname__.startswith(self._helps):
            raise TypeError(
                f"This show option is for {self._helps}. "
                f"Given helpee is {type(helpee)}."
            )
        self._options = dict()
        self._backend = settings.VISUALIZATION_BACKEND

        # initialize backend specific option holder
        self._options[self._backend] = dict()

    def __repr__(self):
        """
        Friendly info prints for show option.

        Parameters
        ----------
        None

        Returns
        -------
        description: str
        """
        valid_and_current = list()
        for vo in self._valid_options[self._backend].values():
            valid = str(vo)
            current = ""
            if vo.key in self.keys():
                current = "current option: " + str(self[vo.key])
            valid_and_current.append(valid + current)

        header = [
            f"ShowOption for {self._helps}",
            f"Selected Backend: {self._backend}",
        ]
        return "\n".join(header + valid_and_current)

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
                value, self._valid_options[self._backend][key].allowed_types
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
            raise ValueError(f"{key} is an invalid option for {self._helps}.")

    def __getitem__(self, key):
        """
        operator[]

        Parameters
        ----------
        key: str or iterable

        Returns
        -------
        items: object or dict
        """
        if isinstance(key, str):
            return self._options[self._backend][key]
        elif hasattr(key, "__iter__"):
            items = dict()
            for k in key:
                if k in self._options[self._backend]:
                    items[k] = self._options[self._backend][k]
            return items
        else:
            raise TypeError(f"Invalid key type for {type(self)}")

    def get(self, key, default):
        """
        Gets value from key and default. Similar to dict.get()

        Parameters
        ----------
        key: stir
        default: object

        Returns
        -------
        values: object
        """
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

    def valid_keys(self, backend=None):
        """
        Returns valid keys. Can directly specify backend. If not, returns
        valid_keys for currently selected backend.

        Parameters
        ----------
        backend: str

        Returns
        -------
        valid_keys: dict_keys
        """
        backend = self._backend if backend is None else backend
        return self._valid_options[backend].keys()

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
        return self._options[self._backend].keys()

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
        return self._options[self._backend].values()

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
        return self._options[self._backend].items()

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

    def pop(self, *args, **kwargs):
        """
        Calls pop() on current backend options

        Parameters
        ----------
        None

        Returns
        -------
        value: object
        """
        self._options[self._backend].pop(*args, **kwargs)

    def copy_valid_options(self, copy_to, keys=None):
        """
        Copies valid option to other show_option. Simply iterates and tries.

        Parameters
        ----------
        copy_to: ShowOption
        keys: tuple or list
          Can specify keys

        Returns
        -------
        None
        """
        if not isinstance(copy_to, ShowOption):
            raise TypeError("copy_to should be a ShowOption")
        valid_keys = copy_to.valid_keys()

        if keys is not None:
            items = self[keys].items()
        else:
            items = self.items()

        for key, value in items:
            if key in valid_keys:
                copy_to[key] = deepcopy(value)  # is deepcopy necessary?

    def _initialize_showable(self):
        """
        Creates basic showable all the way up to backend common procedures.

        Parameters
        ----------
        None

        Returns
        -------
        showable: object
        """
        return eval(f"self._initialize_{self._backend}_showable()")
