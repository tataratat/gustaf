"""gustaf/gustaf/io/nutils.py
io functions for nutils.
"""

import os
import struct

import numpy as np

from gustaf.vertices import Vertices
from gustaf.faces import Faces
from gustaf.volumes import Volumes
from gustaf.io.ioutils import abs_fname, check_and_makedirs
from gustaf.utils import log
from gustaf.io import mixd

def load(fname):
    """
    nutils load.
    Loads a nutils (np.savez) file and returns a Gustaf Mesh.

    Parameters
    -----------
    fname: str
      The npz file needs the following keys: nodes, cnodes, coords, tags, btags, ptags.
    """
    npzfile = np.load(fname, allow_pickle=True)
    nodes = npzfile['nodes']
    cnodes = npzfile['cnodes']
    coords = npzfile['coords']
    tags = npzfile['tags'].item()
    btags = npzfile['btags'].item()
    ptags = npzfile['ptags'].item()

    vertices = coords

    # connec
    simplex = True
    connec = None
    volume = True  #adaption nec

    try:
        connec = nodes
    except:
        log.debug("Error")

    # reshape connec
    if connec is not None:
        ncol = int(3) if simplex and not volume else int(4)
        ncol = int(8) if ncol == int(4) and volume and not simplex else ncol

        connec = connec.reshape(-1, ncol)

        mesh = Volumes(vertices, connec) if volume else Faces(vertices, connec)

    mesh.BC = btags
    return mesh

def export(fname, mesh):
    """
    Export in Nutils format. Files are saved as np.savez().
    Supports triangle,and tetrahedron Meshes.

    Parameters
    -----------
    mesh: Faces or Volumes
    fname: str

    Returns
    --------
    None
    """

    dic = to_nutils_simplex(mesh)

    # prepare export location
    fname = abs_fname(fname)
    check_and_makedirs(fname)

    np.savez(fname, **dic)



def to_nutils_simplex(mesh):
    """
    Converts a Gustaf_Mesh to a Dictionary, which can be interpreted by nutils.mesh.simplex(**to_nutils_simplex(mesh)). Only work for Triangles and Tetrahedrons!

    Parameters
    -----------
    mesh: Faces or Volumes

    Returns
    --------
    dic_to_nutils: dict
    """

    dic_to_nutils = dict()

    vertices = mesh.get_vertices_unique()
    faces = mesh.get_faces()			
    whatami = mesh.get_whatami()

    if whatami.startswith("tri"):
        dimension = 2
        permutation = [1,2,0]
        elements = faces
    elif whatami.startswith("tet"):
        dimension = 3
        permutation = [2,3,1,0]
        volumes = mesh.volumes				
        elements = volumes     
    else:
        raise TypeError('Only Triangle and Tetrahedrons are accepted.') 
    
    #Sort the Node IDs for each Element. In 2D, element = face. In 3D, element = volume.
    elements_sorted = np.zeros(elements.shape)
    sort_array = np.zeros(elements.shape)
    
    for index, row in enumerate(elements, start = 0):
        elements_sorted[index] = np.sort(row)	
        sort_array[index] = np.argsort(row)
    elements_sorted = elements_sorted.astype(int)	
    sort_array = sort_array.astype(int)

    #Let`s get the Boundaries
    bcs = dict()
    bcs_in = mixd.as_mrng(dimension + 1,mesh)	
    bcs_in = np.ndarray.reshape(bcs_in,(int(len(bcs_in)/(dimension + 1)),(dimension + 1)))

    bound_id = np.unique(bcs_in)
    bound_id = bound_id[bound_id > 0]

    #Reorder the mrng according to nutils permutation --> important: does not work if mien file is rotated.
    bcs_in[:,:] = bcs_in[:,permutation]	#swap collumns
        
    #Let's reorder the mrng file with the sort_array
    bcs_sorted = np.zeros(bcs_in.shape)	
    for index, sorts in enumerate(sort_array, start = 0):
        bcs_sorted[index] = bcs_in[index][sorts]
    bcs_sorted = bcs_sorted.astype(int)
        
    for bi in bound_id:
        temp = []
        for elid, el in enumerate(bcs_sorted, start = 0):
            for index, edge in enumerate(el, start = 0):
                if bi == edge:
                    temp.append([elid,index])
        bcs.update(
            {str(bi)		:	np.array(temp)}
        )

    dic_to_nutils.update(   
        {   'nodes'     :   elements_sorted,
            'cnodes'    :   elements_sorted,
            'coords'    :   vertices,
            'tags'      :   {},
            'btags'     :   bcs	,
            'ptags'     :   {}
        }   
    )

    return dic_to_nutils

