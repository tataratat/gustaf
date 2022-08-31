"""gustaf/gustaf/io/npz.py

io functions for numpy.savez.
"""

#import os
#import struct

import numpy as np

from gustaf.vertices import Vertices
from gustaf.edges import Edges
from gustaf.faces import Faces
from gustaf.volumes import Volumes
#from gustaf.io.ioutils import abs_fname, check_and_makedirs
#from gustaf.utils import log

# tuple instead of dict for correct iteration order
_types_and_arrays = (
        (Vertices, ('vertices')),
        (Edges, ('edges', 'vertices')),
        (Faces, ('faces', 'vertices')),
        (Volumes, ('volumes', 'vertices')),
        )

def load(
        fname,
        force_type = None,
        check = False
):
    """
    Load an npz file.

    This will look for a connectivity array named either 'volumes', 'faces', or
    'edges' and determine the class type based on that. If none of these are
    found, only vertices are read.

    A specific mesh type can be forced with, e.g., force_type=Volumes.

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
        # determine mesh class type
        mesh_type = None
        if force_type is not None:
            assert force_type in (Volumes, Faces, Edges, Vertices)
            mesh_type = force_type
        else:
            for type_, array_names in _types_and_arrays:
                if array_names[0] in data:
                    mesh_type = type_
        assert mesh_type is not None, \
                "Could not determine mesh type."

        # create mesh
        mesh = mesh_type(**{
            array_name: data[array_name]
            for array_name in dict(_types_and_arrays)[mesh_type]
            })

        # run checks?
        if check:
            assert np.all(np.less(mesh.elements().flatten(),
                mesh.vertices.shape[0])), \
                        "Connectivity array is referencing invalid " \
                        "vertex indices."

        # also import BC array
        if mesh_type in (Faces, Volumes):
            # look for 'BC_*' arrays and import as '*'
            mesh.BC = {
                    boundary_name[0]: array
                    for (prefix, *boundary_name), array in
                    ((array_name.split('_', 1), array)
                        for array_name, array in data.items())
                    if prefix == "BC"
                    }

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
    assert type(mesh) in (Volumes, Faces, Edges, Vertices)

    # create export dict
    data = {
            array_name: getattr(mesh, array_name)
            for array_name in dict(_types_and_arrays)[type(mesh)]
            }

    # also export boundary condition dict
    if type(mesh) in (Volumes, Faces) and hasattr(mesh, 'BC'):
        data.update((
            (f'BC_{boundary_name}', array)
            for boundary_name, array in mesh.BC.items()
            ))

    # write to file
    (np.savez if not compressed else np.savez_compressed)(fname, **data)

