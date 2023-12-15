import numpy as np
import vedo
from IPython.display import display
from ipywidgets import GridspecLayout


def get_shape(N, x, y):
    """Taken verbatim from vedo plotter class.

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


class K3DPlotterN(GridspecLayout):
    def __init__(self, N, size, background=0xFFFFFF):
        self.N = N
        self.x, self.y = get_shape(N, *(2160, 1440))
        self.shape = (self.x, self.y)
        super().__init__(self.x, self.y)
        self.renderers = []
        for _ in range(N):
            self.renderers.append(
                vedo.Plotter(
                    size=size,
                    bg=background,
                )
            )

    def _at_get_location(self, N):
        if self.x * self.y < N:
            return (self.x - 1, self.y - 1)
        return (N // (self.y + 1), N % self.y)

    def show(self, list_of_showables, at, interactive, camera):
        self[self._at_get_location(at)] = self.renderers[at].show(
            list_of_showables,
            interactive=interactive,
            camera=camera,
            # offscreen=offscreen,
        )

    def display(self):
        display(self)

    def clear(*args, **kwargs):
        pass
