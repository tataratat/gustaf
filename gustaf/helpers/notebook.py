"""gustaf/gustaf/helpers/notebook.py.

Enables the plotting in ipynb with k3d.

There are no import guards since they are in the place where this module is
imported. This should be enough since I do not think that this module
will/should be used outside this place.
"""

import importlib

import numpy as np
import vedo
from IPython.display import display
from ipywidgets import GridspecLayout

from gustaf._base import GustafBase


def get_shape(N, x, y):
    """Taken verbatim from vedo plotter:show function.

    Args:
        N (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    nx = int(np.sqrt(int(N * y / x) + 1))
    ny = int(np.sqrt(int(N * x / y) + 1))
    lm = [
        (nx, ny),
        (nx, ny + 1),
        (nx - 1, ny),
        (nx + 1, ny),
        (nx, ny - 1),
        (nx - 1, ny + 1),
        (nx + 1, ny - 1),
        (nx + 1, ny + 1),
        (nx - 1, ny - 1),
    ]
    ind, minl = 0, 1000
    for i, m in enumerate(lm):
        l_something = m[0] * m[1]
        if N <= l_something < minl:
            ind = i
            minl = l_something
    return lm[ind]


class K3DPlotterN(GridspecLayout, GustafBase):
    """Helper to plot in notebooks with k3d.

    Sets up K3D plotter especially if multiple plots are to be shown in the
    notebook.
    """

    def __init__(self, N, size, background=0xFFFFFF):
        """Setups the plotter with potentially multiple plots.

        Parameters
        ----------
        N: int
            Number of plots to be shown.
        size: (str, list)
            See vedo.Plotter for details.
        background: ((color, str), optional)
          See vedo.Plotter for details. Defaults to 0xFFFFFF.
        """
        if importlib.util.find_spec("k3d") is None:
            raise ImportError(
                "k3d is not installed. Please install it with `pip "
                "install k3d`."
            )
        self.N = N
        self.x, self.y = get_shape(N, *(2160, 1440))
        self.shape = (self.x, self.y)
        super().__init__(self.x, self.y)
        self.vedo_plotters = []
        for _ in range(N):
            self.vedo_plotters.append(
                vedo.Plotter(
                    size=size,
                    bg=background,
                )
            )

    def _at_get_location(self, N):
        """Gets the grid coordinate of the wanted renderer.

        Parameters
        ----------
        N: int
            Render id.

        Returns:
            (int, int): Grid Coordinate of the wanted renderer.
        """
        if (self.x * self.y) < N:
            return (self.x - 1, self.y - 1)
        return (N // (self.y), N % self.y)

    def show(
        self, list_of_showables, at, interactive, camera, axes, *args, **kwargs
    ):
        """Add the showables to the renderer at the given location.

        Parameters
        -----------
        list_of_showables: Any
        at: int
            Render id.
        interactive: bool
            See vedo.Plotter.show for details.
        camera: Any
            See vedo.Plotter.show for details.
        axes: bool
            Add axes to the plot. Will also cast int to bool.
        """
        if len(args) != 0 or len(kwargs) != 0:
            self._logd(f"*args ({args}) and **kwargs ({kwargs}) ignored")

        # this converts vedo plotter to k3d plot.
        # after this, vedo plotter has no relevance to what you see
        self[self._at_get_location(at)] = self.vedo_plotters[at].show(
            list_of_showables,
            interactive=interactive,
            camera=camera,
            axes=axes,
        )

    def display(self, close=True):
        """Display the plotter.

        This is needed in case the plotter is the last thing in a cell. In that
        case the IPython will try to call this function to display this.
        """
        display(self)

        # we add option to close here, as we set default_autoclose = False
        if close:
            self.clear()
            self.close()

    def clear(self, *args, **kwargs):
        """Clear the plotters."""
        for v_plotter in self.vedo_plotters:
            v_plotter.clear(*args, deep=True, **kwargs)

    def close(self):
        """Closes all vedo.Plotters"""
        for v_plotter in self.vedo_plotters:
            v_plotter.close()
