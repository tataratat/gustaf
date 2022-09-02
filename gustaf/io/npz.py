"""gustaf/gustaf/io/npz.py

io functions for numpy.savez.
"""

import numpy as np

from gustaf import utils
from gustaf.vertices import Vertices
from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.volumes import Volumes

_stored_arrays = {
        Vertices: ('vertices',),
        Edges: ('vertices', 'edges'),
        Faces: ('vertices', 'faces'),
        Volumes: ('vertices', 'volumes')
        }

_stored_dicts = {
        Vertices: (),
        Edges: (),
        Faces: ('BC',),
        Volumes: ('BC',)
        }

def load(
        fname,
        force_type=None,
        check=False
):
    """
    Load an npz file.

    The type is determined from the 'kind' array. If force_type is used, this
    type must match what is found in 'kind'.

    Arrays, such as 'vertices', are read from the file as they are. Dictionaries
    are created from certain array groups that start with the same prefix, such
    as 'vertex_groups_'.

    Parameters
    -----------
    fname : str
    force_type : type or None
        (Default: None)
    check : bool
        (Default: False) Check consistency of connectivity data.

    Returns
    -------
    mesh : Volumes, Faces, Edges, or Vertices
    """
    # read a dictionary from the file
    with np.load(fname, allow_pickle=False) as data:
        # find kind and set mesh class type
        mesh_kind_array = data.get("kind", None)
        if mesh_kind_array is not None and len(mesh_kind_array) == 1:
            mesh_kind = mesh_kind_array[0]
        else:
            raise RuntimeError("NPZ file is missing correct 'kind' array. "
                    "Found " + ", ".join(data) + ".")

        # prepare kind => type dict (e.g. 'volume': Volumes)
        kind_to_type = {type_.kind: type_ for type_ in _stored_arrays.keys()}
        try:
            mesh_type = kind_to_type[mesh_kind]
        except KeyError:
            raise RuntimeError(f"Kind '{mesh_kind}' doesn't match any known "
                    + "types: " + ", ".join(kind_to_type) + ".")

        if force_type is not None and force_type != mesh_type:
            raise RuntimeError(f"Requested {force_type} but got {mesh_type}.")

        # loop over all array entries and assign correctly
        # (we're importing into intermediate variables so we can make sure all
        # arrays are imported before the dicts.)
        read_arrays = {}
        read_dicts = {}
        for name, content in data.items():
            # check for array
            if name in _stored_arrays[mesh_type]:
                read_arrays[name] = content
            elif name == "kind":
                continue
            else:
                # read dictionary entry name as list to avoid error if
                # underscore is missing
                dict_name, *dict_entry_name = name.split("_", 1)
                # check for dict
                if (dict_name in _stored_dicts[mesh_type]
                        and len(dict_entry_name) > 0):
                    if dict_name not in read_dicts:
                        read_dicts[dict_name] = {}
                    read_dicts[dict_name][dict_entry_name[0]] = content
                else:
                    utils.log.warning("Encountered unknown array "
                            f"'{name}' while reading NPZ file.")

        # check if everything's there
        missing_arrays = set(_stored_arrays[mesh_type]).difference(
                set(read_arrays))
        if missing_arrays:
            raise RuntimeError("Missing arrays in NPZ file: "
                    + ", ".join(list(missing_arrays)))

        # we are not checking for dictionaries, because they may be empty on
        # purpose!

        # create mesh
        mesh = mesh_type(**read_arrays)

        # run checks?
        if check:
            if not np.all(np.less(mesh.elements().flatten(),
                    mesh.vertices.shape[0])):
                raise RuntimeError("Connectivity array is referencing "
                    "invalid vertex indices.")

        # import dictionaries
        for dict_name, dict_content in read_dicts.items():
            # we try to update existing dictionaries since they might be of a
            # subclass type
            if hasattr(mesh, dict_name):
                getattr(mesh, dict_name).update(dict_content)
            else:
                setattr(mesh, dict_name, dict_content)

        return mesh


def export(mesh, fname, compressed = False):
    """
    Export with numpy.savez.

    Parameters
    -----------
    mesh : Volumes, Faces, Edges, or Vertices
    fname : str
    compressed : bool
        (Default: False) Use 'savez_compressed'.

    Returns
    --------
    None
    """
    if type(mesh) not in (Volumes, Faces, Edges, Vertices):
        raise TypeError(f"Can't export type {type(mesh)}.")

    # export arrays
    data = {
            array_name: getattr(mesh, array_name)
            for array_name in _stored_arrays[type(mesh)]
            }

    # add kind
    data["kind"] = np.array([type(mesh).kind])

    # export dictionaries
    for dict_name in _stored_dicts[type(mesh)]:
        if hasattr(mesh, dict_name):
            data.update({
                f'{dict_name}_{dict_entry_name}': array
                for dict_entry_name, array in getattr(mesh, dict_name).items()
                })

    # write to file
    (np.savez if not compressed else np.savez_compressed)(fname, **data)

