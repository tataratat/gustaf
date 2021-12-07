"""gustav/gustav/show.py

Everything related to show/visualization.
"""

import numpy as np

from gustav import settings
from gustav import utils


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
        import vedo
        vedo.show(showables).close()
    elif vis_b.startswith("trimesh"):
        pass
    elif vis_b.startswith("matplotlib"):
        pass
    else:
        raise NotImplementedError


def show_vedo(vedo_lists):
    """
    Thin vedo show wrapper to nicely 
    """
    pass


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

    utils.log._debug("making vedo-showable obj")
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
