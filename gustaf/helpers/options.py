"""gustaf/gustaf/helpers/options.py

Classes to help organize options.
"""

from copy import deepcopy

from gustaf.helpers._base import HelperBase
from gustaf.helpers.raise_if import ModuleImportRaiser


class Option:
    """
    Minimal Class to hold each options. Mainly to replace nested dict.

    Parameters
    ----------
    backends: set
      set of strings.
    key: str
    description: str
    allowed_types: set
      set of types
    default: one of allwed_types
      Optional. Default is None
    """

    __slots__ = (
        "backends",
        "key",
        "description",
        "allowed_types",
        "default",
    )

    def __init__(
        self, backends, key, description, allowed_types, default=None
    ):
        """
        Check types
        """
        if isinstance(backends, str):
            self.backends = {backends}
        elif getattr(backends, "__iter__", False):
            self.backends = set(backends)
        else:
            raise TypeError("Invalid backends type")

        if isinstance(key, str):
            self.key = key
        else:
            raise TypeError("Invalid key type")

        if isinstance(description, str):
            self.description = description
        else:
            raise TypeError("Invalid description type")

        if isinstance(allowed_types, (tuple, list, set)):
            self.allowed_types = tuple(allowed_types)
        else:
            raise TypeError("Invalid allowed_types type")

        if default is None or type(default) in self.allowed_types:
            self.default = default
        else:
            raise TypeError(f"{type(default)} is invalid default type")

    def __repr__(self):
        specific = "\n".join(
            [
                "",
                self.key,
                "=" * len(self.key),
                f"backends: {self.backends}",
                "description:",
                f"  {self.description}",
                "allowed_types:",
                f"  {self.allowed_types}",
                super().__repr__(),
                "",
            ]
        )
        return specific


class SetDefault:
    """
    Default setter object. Can use as argument in `make_valid_options`
    """

    __slots__ = ("key", "default")

    def __init__(self, key, default):
        self.key = key
        self.default = default


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
        "data",
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
        "cmap_n_colors",
        "Set the number of available colors",
        (int,),
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
        "title_rotation: float, nlabels: int, label_font: str, "
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
        "arrow_data_to_origin",
        "Points arrow data to geometric origin. By default, arrows point away "
        "from origin. "
        "When enabled, arrows are shifted backwards by their own magnitudes. "
        "Works in conjunction with arrow_data and arrow_data_scale options.",
        (bool,),
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
    valid_options = {}
    for opt in options:
        if isinstance(opt, Option):
            # copy option object to avoid overwriting defaults
            # only exception is if this option is a backend object
            # and wrapped by ModuleImportRaiser
            allowed_types = []
            for at in opt.allowed_types:
                if isinstance(at, ModuleImportRaiser):
                    continue
                allowed_types.append(at)
            opt.allowed_types = tuple(allowed_types)

            valid_options[opt.key] = deepcopy(opt)
        elif isinstance(opt, SetDefault):
            # overwrite default of existing option.
            if opt.key not in valid_options:
                raise KeyError("Given key is not in valid_option.")

            if type(opt.default) not in valid_options[opt.key].allowed_types:
                raise TypeError("Invalid default type")
            valid_options[opt.key].default = opt.default
        else:
            raise TypeError("Please use `Option` to define options.")

    return valid_options


class ShowOption(HelperBase):
    """
    Behaves similar to dict, but will only accept a set of options that's
    applicable to the helpee class. Intended use is to create a
    subclass that would define valid options for helpee.
    Options should be described by Option object.
    Helps all the way up to initializing backend showables up to their backend
    specific common routines. ShowOption and ShowManager in a sense.
    """

    __slots__ = ("_options",)

    _valid_options = {}

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
        self._options = {}

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
        valid_and_current = []
        for vo in self._valid_options.values():
            valid = str(vo)
            current = ""
            if vo.key in self.keys():
                current = "current option: " + str(self[vo.key])
            valid_and_current.append(valid + current)

        header = [
            f"ShowOption for {self._helps}",
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
        if key in self._valid_options:
            # valid type check
            if not isinstance(value, self._valid_options[key].allowed_types):
                raise TypeError(
                    f"{type(value)} is invalid value type for '{key}'. "
                    f"Details for '{key}':\n"
                    f"{self._valid_options[key]}"
                )

            # types are valid. let's add
            self._options[key] = value

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
            return self._options[key]
        elif hasattr(key, "__iter__"):
            items = {}
            for k in key:
                if k in self._options:
                    items[k] = self._options[k]
            return items
        else:
            raise TypeError(f"Invalid key type for {type(self)}")

    def __contains__(self, key):
        """Returns if current option contains given key.

        Parameters
        ----------
        key: str

        Returns
        -------
        contains: bool
        """
        return key in self._options

    def __call__(self, **kwargs):
        """A short-cut to update(), but it returns helpee object.
        This is mainly to support one-line/inline visualizations.

        Parameters
        ----------
        kwargs: **kwargs

        Returns
        -------
        helpee: Any

        Example
        -------
        .. code-block:: python

            gus.show(
                mesh1.show_options(c="red"),
                mesh2.show_options(c="green"),
                mesh3.show_options(c="blue"),
            )
        """
        self.update(**kwargs)

        return self._helpee

    def get(self, key, default=None):
        """
        Gets value from key and default. Similar to dict.get(),
        but this is always safe, as it will always return None

        Parameters
        ----------
        key: stir
        default: object

        Returns
        -------
        values: object
        """
        if default is not None:
            return self._options.get(key, default)

        # overwrite default with valid option's
        default = getattr(self._valid_options.get(key, None), "default", None)

        return self._options.get(key, default)

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

    def valid_keys(self):
        """
        Returns valid keys.

        Parameters
        ----------
        None

        Returns
        -------
        valid_keys: dict_keys
        """
        return self._valid_options.keys()

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

    def pop(self, *args, **kwargs):
        """
        Calls pop() on current options

        Parameters
        ----------
        None

        Returns
        -------
        value: object
        """
        self._options.pop(*args, **kwargs)

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

        items = self[keys].items() if keys is not None else self.items()

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
        raise NotImplementedError("Derived class must implement this method")
