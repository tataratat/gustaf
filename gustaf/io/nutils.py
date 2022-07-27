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
    npzfile = np.load(fname, allow_pickle=True)
    nodes = npzfile['nodes']
    cnodes = npzfile['cnodes']
    coords = npzfile['coords']
    tags = npzfile['tags'].item()
    btags = npzfile['btags'].item()
    ptags = npzfile['ptags'].item()



    print(btags)

#    return mesh

def export(fname, mesh):
    dic = to_nutils_simplex(mesh)

    # prepare export location
    fname = abs_fname(fname)
    check_and_makedirs(fname)

    np.savez(fname, nodes = dic['nodes'], cnodes = dic['cnodes'], coords = dic['coords'], tags = dic['tags'] , btags = dic['btags'], ptags = dic['ptags'])


def to_nutils_simplex(mesh):
    dic_to_nutils = dict()
    dimension = 2
    permutation = [1,2,0]

    vertices = mesh.get_vertices_unique()
    faces = mesh.get_faces()			
    elements = faces
    whatami = mesh.get_whatami()
    
    if whatami.startswith("tet"):
        dimension = 3
        permutation = [2,3,1,0]
        volumes = mesh.volumes				
        elements = volumes
    
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
    bcs_in = np.ndarray.reshape(bcs_in,(int(len(bcs_in)/(dimension + 1)),(dimension + 1)))			#this is the mrng file

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

