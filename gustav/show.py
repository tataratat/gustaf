"""gustav/gustav/show.py

Everything related to show/visualization.
"""

import numpy as np

from gustav import settings
from gustav import utils
from gustav._base import GustavBase


def show(*gusobj, **kwargs):
    """
    Shows using appropriate backend.

    Parameters
    -----------
    *gusobj: gustav objects 

    Returns
    --------
    None
    """
    showables = [make_showable(g, **kwargs) for g in gusobj]
    vis_b = settings.VISUALIZATION_BACKEND

    if vis_b.startswith("vedo"):
        show_vedo(showables)
    elif vis_b.startswith("trimesh"):
        pass
    elif vis_b.startswith("matplotlib"):
        pass
    else:
        raise NotImplementedError


def show_vedo(*args):
    """
    `vedo.show` wrapper.
    Each args represent one section of window. In other words len(args) == N,
    where N corresponds to the parameter for vedo.show().

    Parameters
    -----------
    *args: *list or *dict or gustav_obj or vedo_obj
    """
    import vedo

    # vedo parameter
    N = len(args)

    # get plotter
    plt = vedo.Plotter(N=N, sharecam=False)

    # loop and plot
    for i, arg in enumerate(args):
        # form valid input type.
        if isinstance(arg, dict):
            showlist = list(arg.values())
        elif isinstance(arg, list):
            showlist = arg
        else:
            #raise TypeError("For vedo_show, only list or dict is valid input")
            utils.log.debug(
                "one of args for show_vedo is neither `dict` nor",
                "`list`. Putting it naively into a list."
            )
            showlist = [arg]

        # quickcheck if the list is gustav or non-gustav
        # if gustav, make it vedo-showable.
        # if there's spline, we need to pop the element and
        # extend showables to the list.
        to_pop = []
        to_extend = []
        for j, sl in enumerate(showlist):
            if isinstance(sl, GustavBase):
                tmp_showable = sl.showable(backend="vedo")
                # splines return dict
                # - maybe it is time to do some typing..
                if isinstance(tmp_showable, dict):
                    # mark to pop later
                    to_pop.append(j)
                    # add to extend later
                    to_extend.append(list(tmp_showable.values()))

                else:
                    # replace gustavobj with vedo_obj.
                    showlist[j] = tmp_showable

        # extend and pop
        if len(to_pop) == len(to_extend) != 0:
            for te in to_extend:
                showlist.extend(te)

            # pop bigger indices first
            to_pop.sort()
            for tp in to_pop[::-1]:
                showlist.pop(tp)

        # set interactive to true at last element
        if int(i + 1) == len(args):
            plt.show(showlist, at=i, interactive=True)
        else:
            plt.show(showlist, at=i, interactive=False)

    plt.close()

def _vedo_showable(obj, **kwargs):
    """
    Generates a vedo obj based on `kind` attribute from given obj.

    Parameters
    -----------
    obj: gustav obj

    Returns
    --------
    vedo_obj: vedo obj
    """
    import vedo

    utils.log.debug("making vedo-showable obj")
    if obj.kind == "vertex":
        return vedo.Points(obj.vertices, **obj.vis_dict, **kwargs)

    elif obj.kind == "edge":
        return vedo.Lines(obj.vertices[obj.edges], **obj.vis_dict, **kwargs)

    elif obj.kind == "face":
        return vedo.Mesh([obj.vertices, obj.faces], **obj.vis_dict, **kwargs)

    elif obj.kind == "volume":
        from vtk import VTK_TETRA as frau_tetra
        from vtk import VTK_HEXAHEDRON as herr_hexa

        whatami = obj.get_whatami()
        if whatami.startswith("tet"):
            grid_type = frau_tetra
        elif whatami.startswith("hexa"):
            grid_type = herr_hexa
        else:
            return None # get_whatami should've rasied error.. 

        vol = vedo.UGrid(
            [
                obj.vertices,
                obj.volumes,
                np.repeat([grid_type], len(obj.volumes)),
            ]
        )

        if kwargs.get("shrink", True):
            vol = vol.tomesh(shrink=.8)

        vol.color(kwargs.get("c", "hotpink"))

        return vol


def _trimesh_showable(obj):
    """
    """
    pass


def _matplotlib_showable(obj):
    """
    """
    pass


def make_showable(obj, backend=settings.VISUALIZATION_BACKEND, **kwargs):
    """
    Since gustav does not natively support visualization, one of the following
    library is used to visualize gustav (visualizable) objects:
    (1) vedo -> Fast, offers a lot of features
    (2) trimesh -> Fast, compatible with old opengl
    (3) matplotlib -> Slow, offers vector graphics

    This determines showing types using `whatami`.

    Parameters
    -----------
    obj: gustav-objects
    backend: str
      (Optional) Default is `gustav.settings.VISUALIZATION_BACKEND`.
      Options are: "vedo" | "trimesh" | "matplotlib"

    Returns
    --------
    showalbe_objs: list
      List of showable objects. 
    """
    if backend.startswith("vedo"):
        return _vedo_showable(obj, **kwargs)
    elif backend.startswith("trimesh"):
        return _trimesh_showable(obj, **kwargs)
    elif backend.startswith("matplotlib"):
        return _matplotlib_showable(obj, **kwargs)
    else:
        raise NotImplementedError
