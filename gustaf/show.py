"""gustaf/gustaf/show.py

Everything related to show/visualization.
"""
from math import prod

import numpy as np

from gustaf import settings
from gustaf import utils
from gustaf._base import GustavBase


def show(*gusobj, **kwargs):
    """
    Shows using appropriate backend.

    Parameters
    -----------
    *gusobj: gustaf objects 

    Returns
    --------
    None
    """
    showables = [make_showable(g, **kwargs) for g in gusobj]
    vis_b = settings.VISUALIZATION_BACKEND

    if vis_b.startswith("vedo"):
        return show_vedo(showables, **kwargs)
    elif vis_b.startswith("trimesh"):
        pass
    elif vis_b.startswith("matplotlib"):
        pass
    else:
        raise NotImplementedError


def show_vedo(*args, **kwargs,):
    """
    `vedo.show` wrapper.
    Each args represent one section of window. In other words len(args) == N,
    where N corresponds to the parameter for vedo.show().

    Parameters
    -----------
    *args: *list or *dict or gustaf_obj or vedo_obj
    """
    import vedo

    # vedo plotter parameter
    N = len(args)
    offs = kwargs.get("offscreen", False)
    interac = kwargs.get("interactive", True)
    plt = kwargs.get("vedoplot", None)
    skip_clear = kwargs.get("skip_clear", False)
    close = kwargs.get("close", None)

    def clear_vedoplotter(plotter, numrenderers, skipcl=skip_clear):
        """enough said."""
        # for whatever reason it is desired
        if skipcl:
            return None

        for i in range(numrenderers):
            plotter.clear(at=i)

        return None

    # get plotter
    if plt is None:
        plt = vedo.Plotter(N=N, sharecam=False, offscreen=offs)

    else:
        # check if plt has enough Ns
        trueN = prod(plt.shape)
        clear_vedoplotter(plt, trueN) # always clear. 
        if trueN != N:
            utils.log.warning(
                "Number of args exceed given vedo.Plotter's capacity.",
                "Assigning a new one",
            )
            if close: # only if it is explicitly stated
                plt.close() # Hope that this truely releases..
            # assign a new one
            plt = vedo.Plotter(N=N, sharecam=False, offscreen=offs)


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

        # quickcheck if the list is gustaf or non-gustaf
        # if gustaf, make it vedo-showable.
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
                    # replace gustafobj with vedo_obj.
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
            plt.show(
                showlist,
                at=i,
                interactive=interac,
                #offscreen=offs,
            )

        else:
            plt.show(
                showlist,
                at=i,
                interactive=False,
                #offscreen=offs,
            )

    if interac:
        # only way to ensure memory is released
        clear_vedoplotter(plt, prod(plt.shape))

        if close or close is None: # explicitly given or None.
            # It seems to leak some memory, but here it goes.
            plt.close() # if i close it, this cannot be reused...
            return None

    return plt

def _vedo_showable(obj, **kwargs):
    """
    Generates a vedo obj based on `kind` attribute from given obj.

    Parameters
    -----------
    obj: gustaf obj

    Returns
    --------
    vedo_obj: vedo obj
    """
    import vedo

    # parse from vis_dict
    basic_options = dict(
        c = obj.vis_dict.get("c", None),
        r = obj.vis_dict.get("r", None),
        lw = obj.vis_dict.get("lw", None),
        alpha = obj.vis_dict.get("alpha", None),
        #shrink = obj.vis_dict.get("shrink", None),
        cmap = obj.vis_dict.get("cmap", None),
        dataname = obj.vis_dict.get("dataname", None)
    )
    # loop once more to extract basics from kwargs
    # done after vis_dict, so that this overpowers
    keys = list(kwargs.keys()) # to pop dict during loop
    for key in keys:
        if key in basic_options.keys():
            basic_options[key] = kwargs[key]
            kwargs.pop(key)

    utils.log.debug("making vedo-showable obj")
    local_options = dict()

    if obj.kind == "vertex":
        for key in ["c", "r", "alpha"]:
            value = basic_options[key]
            if value is not None:
                local_options.update({key : value})

        vobj = vedo.Points(
            obj.vertices,
            **local_options,
            **kwargs
        )

    elif obj.kind == "edge":
        for key in ["c", "lw", "alpha"]:
            value = basic_options[key]
            if value is not None:
                local_options.update({key : value})

        vobj = vedo.Lines(
            obj.vertices[obj.edges],
            **local_options,
            **kwargs
        )

    elif obj.kind == "face":
        for key in ["c", "alpha"]:
            value = basic_options[key]
            if value is not None:
                local_options.update({key : value})

        vobj = vedo.Mesh(
            [obj.vertices, obj.faces],
            **local_options,
            **kwargs
        )

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

        ## shrink should be done at gustaf level, since it is available now!
        # shrink? save here to allocate values accordingly
        #shrink = basic_options["shrink"]
        #if shrink is None:
        #    shrink = kwargs.get("shrink", True) # maybe False?

        #if shrink:
        #    vobj = vobj.tomesh(shrink=.8)
        #    if basic_options["c"] is None:
        #        vobj.color("hotpink")
        #    else:
        #        vobj.color(basic_options["c"])

        if basic_options["dataname"]:
            from gustaf.faces import Faces

            # UGrid would be politically correct,
            # but currently it can't show field
            # so, extract only surface mesh
            surf_ids = obj.get_surfaces() # gets faces too
            sfaces = Faces(obj.vertices, obj.faces[surf_ids])
            sfaces.remove_unreferenced_vertices(inplace=True)

            vobj = sfaces.showable(backend="vedo") # recursive alert
        
        else:
            vobj = vedo.UGrid(
                [
                    obj.vertices,
                    obj.volumes,
                    np.repeat([grid_type], len(obj.volumes)),
                ]
            )

            if basic_options["c"] is None:
                basic_options["c"] = "hotpink"

            vobj.color(basic_options["c"])
            vobj.alpha(basic_options["alpha"])

    # this sets vedo v2021.0.6+ requirement
    dname = basic_options["dataname"]
    if dname is not None:
        # transfer data
        vobj.pointdata[dname] = obj.vertexdata[dname]

        # default cmap is jet.
        if basic_options["cmap"] is None:
            basic_options["cmap"] = "jet"

        # register cmap and data
        vobj.cmap(basic_options["cmap"], dname)


    return vobj


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
    Since gustaf does not natively support visualization, one of the following
    library is used to visualize gustaf (visualizable) objects:
    (1) vedo -> Fast, offers a lot of features
    (2) trimesh -> Fast, compatible with old opengl
    (3) matplotlib -> Slow, offers vector graphics

    This determines showing types using `whatami`.

    Parameters
    -----------
    obj: gustaf-objects
    backend: str
      (Optional) Default is `gustaf.settings.VISUALIZATION_BACKEND`.
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
