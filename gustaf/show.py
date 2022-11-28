"""gustaf/gustaf/show.py.

Everything related to show/visualization.
"""
import numpy as np

from gustaf import settings
from gustaf import utils
from gustaf._base import GustafBase

# @linux it raises error if vedo is imported inside the function.
try:
    import vedo
except ImportError as err:
    # overwrites the vedo module with an object which will throw an error
    # as soon as it is used the first time. This means that any non vedo
    # functionality works as before, but as soon as vedo is used a
    # comprehensive exception will be raised which is understandable in
    # contrast to the possible errors previously possible
    from gustaf.helpers.raise_if import ModuleImportRaiser
    vedo = ModuleImportRaiser("vedo", err)


def show(*gusobj, **kwargs):
    """Shows using appropriate backend.

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


def show_vedo(
        *args,
        **kwargs,
):
    """`vedo.show` wrapper. Each args represent one section of window. In other
    words len(args) == N, where N corresponds to the parameter for vedo.show().

    Parameters
    -----------
    *args: Union[List[Union[gustaf_obj, vedo_obj]], Dict[str, Any]]]
    """
    # vedo plotter parameter
    N = len(args)
    offs = kwargs.get("offscreen", False)
    interac = kwargs.get("interactive", True)
    plt = kwargs.get("vedoplot", None)
    skip_clear = kwargs.get("skip_clear", False)
    close = kwargs.get("close", None)
    size = kwargs.get("size", "auto")
    cam = kwargs.get("cam", None)
    title = kwargs.get("title", "gustaf")
    return_show_list = kwargs.get("return_showable_list", False)

    def clear_vedoplotter(plotter, numrenderers, skipcl=skip_clear):
        """enough said."""
        # for whatever reason it is desired
        if skipcl:
            return None

        for i in range(numrenderers):
            plotter.clear(at=i)

        return None

    def cam_tuple_to_list(dictcam):
        """if entity is tuple, turns it into list."""
        if dictcam is None:
            return None

        for key, value in dictcam.items():
            if isinstance(value, tuple):
                dictcam[key] = list(value)

        return dictcam

    # get plotter
    if plt is None:
        plt = vedo.Plotter(
                N=N, sharecam=False, offscreen=offs, size=size, title=title
        )

    else:
        # check if plt has enough Ns
        trueN = np.prod(plt.shape)
        clear_vedoplotter(plt, trueN)  # always clear.
        if trueN != N:
            utils.log.warning(
                    "Number of args exceed given vedo.Plotter's capacity.",
                    "Assigning a new one",
            )
            title = plt.title
            if close:  # only if it is explicitly stated
                plt.close()  # Hope that this truely releases..
            # assign a new one
            plt = vedo.Plotter(
                    N=N,
                    sharecam=False,
                    offscreen=offs,
                    size=size,
                    title=title
            )

    # loop and plot
    for i, arg in enumerate(args):
        # form valid input type.
        if isinstance(arg, dict):
            showlist = list(arg.values())
        elif isinstance(arg, list):
            showlist = arg.copy()
        else:
            # raise TypeError(
            #     "For vedo_show, only list or dict is valid input")
            utils.log.debug(
                    "one of args for show_vedo is neither `dict` nor",
                    "`list`. Putting it naively into a list."
            )
            showlist = [arg]

        # quickcheck if the list is gustaf or non-gustaf
        # if gustaf, make it vedo-showable.
        # if there's spline, we need to pop the element and
        # extend showables to the list.
        # A showlist is a list to be plotted into a single subframe of the
        # plot
        list_of_showables = []
        for sl in showlist:
            if not isinstance(sl, list):
                sl = [sl]
            for k, item in enumerate(sl):
                if isinstance(item, GustafBase):
                    tmp_showable = item.showable(backend="vedo", **kwargs)
                    # splines return dict
                    # - maybe it is time to do some typing..
                    if isinstance(tmp_showable, dict):
                        # add to extend later
                        list_of_showables.extend(list(tmp_showable.values()))

                    else:
                        # replace gustafobj with vedo_obj.
                        list_of_showables.append(tmp_showable)
                else:
                    list_of_showables.extend(sl)

        # set interactive to true at last element
        if int(i + 1) == len(args):
            plt.show(
                    list_of_showables,
                    at=i,
                    interactive=interac,
                    camera=cam_tuple_to_list(cam),
                    # offscreen=offs,
            )

        else:
            plt.show(
                    list_of_showables,
                    at=i,
                    interactive=False,
                    camera=cam_tuple_to_list(cam),
                    # offscreen=offs,
            )

    if interac and not offs:
        # only way to ensure memory is released
        clear_vedoplotter(plt, np.prod(plt.shape))

        if close or close is None:  # explicitly given or None.
            # It seems to leak some memory, but here it goes.
            plt.close()  # if i close it, this cannot be reused...
            plt = None

    if return_show_list:
        return (plt, list_of_showables)
    else:
        return plt


def _vedo_showable(obj, **kwargs):
    """Generates a vedo obj based on `kind` attribute from given obj.

    Parameters
    -----------
    obj: gustaf obj

    Returns
    --------
    vedo_obj: vedo obj
    """
    # parse from vis_dict
    # NOTE: maybe we can make a helper class to organize this nicely
    basic_options = dict(
            c=obj.vis_dict.get("c", None),
            r=obj.vis_dict.get("r", None),
            lw=obj.vis_dict.get("lw", None),
            alpha=obj.vis_dict.get("alpha", None),
            cmap=obj.vis_dict.get("cmap", None),
            # > followings are cmap options
            vmin=obj.vis_dict.get("vmin", None),
            vmax=obj.vis_dict.get("vmax", None),
            cmapalpha=obj.vis_dict.get("cmapalpha", 1),
            # > takes scalarbar options as dict.
            scalarbar=obj.vis_dict.get("scalarbar", None),
            dataname=obj.vis_dict.get("dataname", None),
            # <
            arrows=obj.vis_dict.get("arrows", None),  # only for edges
            # >only for edges internally treated same as `lw`,
            # but higher priority
            thickness=obj.vis_dict.get("thickness", None),
            title=obj.vis_dict.get("title", "gustaf")
    )
    # loop once more to extract basics from kwargs
    # done after vis_dict, so that this overpowers
    keys = list(kwargs.keys())  # to pop dict during loop
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
                local_options.update({key: value})

        vobj = vedo.Points(
                obj.vertices,
                **local_options,
        )

    elif obj.kind == "edge":
        for key in ["c", "lw", "alpha"]:
            value = basic_options[key]
            if value is not None:
                local_options.update({key: value})

        # edges can be arrows if vis_dict["arrows"] is set True
        if not basic_options["arrows"]:
            vobj = vedo.Lines(
                    obj.vertices[obj.edges],
                    **local_options,
            )

        else:
            if basic_options.get("thickness", False):
                local_options.update({"thickness": basic_options["thickness"]})

            # turn lw into thickness if there's no thickness
            elif local_options.get("lw", False):
                thickness = local_options.pop("lw")
                local_options.update({"thickness": thickness})

            # `s` is another param for arrows
            local_options.update({"s": obj.vis_dict.get("s", None)})

            vobj = vedo.Arrows(
                    obj.vertices[obj.edges],
                    **local_options,
            )

    elif obj.kind == "face":
        for key in ["c", "alpha"]:
            value = basic_options[key]
            if value is not None:
                local_options.update({key: value})

        vobj = vedo.Mesh(
                [obj.vertices, obj.faces],
                **local_options,
        )

    elif obj.kind == "volume":
        from vtk import VTK_TETRA as frau_tetra
        from vtk import VTK_HEXAHEDRON as herr_hexa

        whatami = obj.whatami
        if whatami.startswith("tet"):
            grid_type = frau_tetra
        elif whatami.startswith("hexa"):
            grid_type = herr_hexa
        else:
            return None  # get_whatami should've rasied error..

        if basic_options["dataname"]:
            from gustaf.faces import Faces

            # UGrid would be politically correct,
            # but currently it can't show field
            # so, extract only surface mesh
            surf_ids = obj.single_faces()  # gets faces too
            sfaces = Faces(obj.vertices, obj.faces()[surf_ids])
            sfaces.remove_unreferenced_vertices()

            vobj = sfaces.showable(backend="vedo")  # recursive alert

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
        vobj.cmap(
                basic_options["cmap"],
                input_array=dname,
                on="points",  # hardcoded since yet, we don't have cell field
                vmin=basic_options["vmin"],
                vmax=basic_options["vmax"],
                alpha=basic_options["cmapalpha"],
        )

        # scalarbar?
        scalarbar_dict = basic_options["scalarbar"]
        if scalarbar_dict is not None:
            # if horizontal==True, size doesnt really matter
            vobj.addScalarBar(**scalarbar_dict)

    return vobj


def _trimesh_showable(obj):
    """"""
    pass


def _matplotlib_showable(obj):
    """"""
    pass


def make_showable(obj, backend=settings.VISUALIZATION_BACKEND, **kwargs):
    """Since gustaf does not natively support visualization, one of the
    following library is used to visualize gustaf (visualizable) objects: (1)
    vedo -> Fast, offers a lot of features (2) trimesh -> Fast, compatible with
    old opengl (3) matplotlib -> Slow, offers vector graphics.

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


# possibly relocate
def interpolate_vedo_dictcam(cameras, resolutions, spline_degree=1):
    """Interpolate between vedo dict cameras.

    Parameters
    ------------
    cameras: list or tuple
    resolutions: int
    spline_degree: int
      if > 1 and splinepy is available and there are more than two cameras,
      we interpolate all the entries using spline.

    Returns
    --------
    interpolated_cams: list
    """
    try:
        import splinepy
        spp = True

    except ImportError:
        spp = False

    # quick type check loop
    camkeys = ["pos", "focalPoint", "viewup", "distance", "clippingRange"]
    for cam in cameras:
        if not isinstance(cam, dict):
            raise TypeError("Only `dict` description of vedo cam is allowed.")
        else:
            for key in camkeys:
                if cam[key] is None:
                    raise ValueError(
                            f"One of the camera does not contain `{key}` info"
                    )

    interpolated_cams = []
    total_cams = int(resolutions) * (len(cameras) - 1)

    if spp and spline_degree > 1 and len(cameras) > 2:
        if spline_degree > len(cameras):
            raise ValueError(
                    "Not enough camera to interpolate with "
                    f"spline degree {spline_degree}"
            )

        ps = []
        fs = []
        vs = []
        ds = []
        cs = []
        for cam in cameras:
            ps.append(list(cam[camkeys[0]]))
            fs.append(list(cam[camkeys[1]]))
            vs.append(list(cam[camkeys[2]]))
            ds.append([float(cam[camkeys[3]])])
            cs.append(list(cam[camkeys[4]]))

        interpolated = dict()
        for i, prop in enumerate([ps, fs, vs, ds, cs]):
            ispline = splinepy.BSpline()
            ispline.interpolate_curve(
                    query_points=prop,
                    degree=spline_degree,
                    save_query=False,
            )
            interpolated[camkeys[i]] = ispline.sample([total_cams])

        for i in range(total_cams):
            interpolated_cams.append(
                    {
                            camkeys[0]: interpolated[camkeys[0]][i].tolist(),
                            camkeys[1]: interpolated[camkeys[1]][i].tolist(),
                            camkeys[2]: interpolated[camkeys[2]][i].tolist(),
                            camkeys[3]:
                            interpolated[camkeys[3]][i][0],  # float?
                            camkeys[4]: interpolated[camkeys[4]][i].tolist(),
                    }
            )

    else:
        i = 0
        for startcam, endcam in zip(cameras[:-1], cameras[1:]):
            if i == 0:
                interpolated = [
                        np.linspace(
                                startcam[ckeys],
                                endcam[ckeys],
                                resolutions,
                        ).tolist() for ckeys in camkeys
                ]

            else:
                interpolated = [
                        np.linspace(
                                startcam[ckeys],
                                endcam[ckeys],
                                int(resolutions + 1),
                        )[1:].tolist() for ckeys in camkeys
                ]

            i += 1

            for j in range(resolutions):
                interpolated_cams.append(
                        {
                                camkeys[0]: interpolated[0][j],
                                camkeys[1]: interpolated[1][j],
                                camkeys[2]: interpolated[2][j],
                                camkeys[3]: interpolated[3][j],  # float?
                                camkeys[4]: interpolated[4][j],
                        }
                )

    return interpolated_cams
