"""gustaf/gustaf/io/hmascii.py

Read a mesh and boundaries from a HyperMesh .hmascii file.
Output is not supported.

This implementation requires a certain structure of the model. It only imports
volume elements of a single type, which is either tetrahedra or hexahedra. They
all must be in a single component (default name: 'volume').

Boundaries can be imported as vertex groups, which can be converted to face
groups. To enable the creation of a vertex group, a component must be created in
HyperMesh. This component must contain surface elements of the current
subelement type. They must use the same node IDs as the volume elements and not
just share their coordinates.

An initial surface component can be created in HyperMesh using:
tools -> faces
"""

import numpy as np

from gustaf.volumes import Volumes
from gustaf.utils import log

class HMLine:
    def __init__(self, line):
        """
        Parse a line from an HMASCII file.

        Parameters
        -----------
        line: str

        Returns
        --------
        None
        """
        self.name = ''
        self.values = ''

        if line[0] == '*':
            parts = line[1:].split('(')
            self.name = parts[0]
            if len(parts) > 1:
                self.values = [value.strip('"') for value in
                        parts[1][:-2].split(',')]

class HMElementType:
    def __init__(self, number_of_nodes, subelement = ''):
        """
        Store HyperMesh element type information.

        Parameters
        -----------
        number_of_nodes: int
        subelement: str
            This does not need to be specified for element types that only occur
            as boundary elements.

        Returns
        --------
        None
        """
        self.number_of_nodes = int(number_of_nodes)
        self.subelement = str(subelement)

class HMComponent:
    element_types = {
            'tetra4': HMElementType(4, 'tria3'),
            'hexa8': HMElementType(8, 'quad4'),
            'tria3': HMElementType(3),
            'quad4': HMElementType(4)
            }

    def __init__(self, line):
        """
        Create a component from an HMASCII line.

        Parameters
        -----------
        line: HMLine

        Returns
        --------
        None
        """
        self.name = str(line.values[1])
        self.elements = dict() # element type: list()

    def add_element(self, line):
        """
        Parse an element line from an HMASCII file.

        Parameters
        -----------
        line: HMLine

        Returns
        --------
        None
        """
        element_type = line.name
        if element_type not in self.elements:
            self.elements[element_type] = list()
        self.elements[element_type].append([int(node) for node in
            line.values[2:2+self.element_types[element_type].number_of_nodes]])

    def __repr__(self):
        return str(vars(self))

class HMModel:
    def __init__(self, filename):
        """
        Create a representation of a HyperMesh model from the HMASCII file
        specified by `filename`.

        Parameters
        -----------
        filename: str

        Returns
        --------
        None
        """
        # read file into arrays
        self.node_ids = dict() # HM node id: internal node id
        self.node_coordinates = list()
        self.components = list() # HMComponent objects

        current_component = None

        with open(filename, 'r') as hm_file:
            for line_string in hm_file:
                line = HMLine(line_string)

                # read node
                if line.name == "node":
                    self.node_ids[int(line.values[0])] =\
                        len(self.node_coordinates)
                    self.node_coordinates.append([float(coord) for coord in
                        line.values[1:4]])

                # read component
                elif line.name == "component":
                    current_component = HMComponent(line)
                    self.components.append(current_component)

                # read element
                elif line.name in HMComponent.element_types:
                    if not current_component:
                        raise RuntimeError('Encountered element before first '\
                                'component.')
                    current_component.add_element(line)

    def get_component_by_name(self, component_name):
        """
        Find the component specified by `component_name`.

        Parameters
        -----------
        component_name: str

        Returns
        --------
        component: HMComponent
        """
        for component in self.components:
            if component.name == component_name:
                return component
        raise ValueError(f"Can't find requested component: {component_name}")

    def __repr__(self):
        return str(vars(self))

def load(
        fname,
        element_type = '',
        create_face_groups = True,
        main_component_name = 'volume'
):
    """
    mixd load.
    To avoid reading minf, all the crucial info can be given as params.
    Default input will try to import `mxyz`, `mien`, `mrng` from current
    location and assumes mesh is 2D triangle.

    Parameters
    -----------
    fname: str
        The filename ending in '.hmascii'.
    element_type: str
        The volume element type to be imported. If this is not specified, the
        most common element type in the main component is chosen.
    create_vertex_groups: bool
        If True (default), the boundary components will be read into vertex
        groups.
    main_component_name: str
        The name of the component that contains the volume elements (default:
        'volume').
    """
    hm_model = HMModel(fname)

    # find main component
    main_component = hm_model.get_component_by_name(main_component_name)

    # determine import element type
    if not element_type:
        # find dominant element type
        if not len(main_component.elements):
            raise RuntimeError(f"Component '{main_component.name}' does "\
                    "not contain any elements of known type.")
        element_type = sorted([(len(element_list), element_type)
            for element_type, element_list in
            main_component.elements.items()])[-1][1]

    # are there additional element types?
    if len(main_component.elements) > 1:
        log.warning(f"Component '{main_component.name}' contains "\
                f"more than one element type. Ignoring all except "\
                f"'{element_type}'.")

    # create unique element list
    hm_volume_elements_nonunique = np.array(
            main_component.elements[element_type], dtype=np.int32)
    hm_volume_elements_sorted = np.sort(hm_volume_elements_nonunique, axis=1)
    hm_volume_elements_unique_indices = np.unique(hm_volume_elements_sorted,
            return_index=True, axis=0)[1]
    # sorting the unique indices isn't necessary, but it might maintain a more
    # contiguous element order
    volumes = np.squeeze(hm_volume_elements_nonunique[
            hm_volume_elements_unique_indices.sort()])

    # create minimal vertex array
    hm_node_indices = np.unique(volumes)
    number_of_vertices = len(hm_node_indices)
    if number_of_vertices == len(hm_model.node_coordinates):
        vertices = np.array(hm_model.node_coordinates, dtype=np.float64)
        node_perm = hm_model.node_ids
    else:
        # filter nodes
        vertices = np.zeros((number_of_vertices, 3), dtype=np.float64)
        node_perm = dict()
        for vertex_id, hm_node_id in enumerate(hm_node_indices):
            vertices[vertex_id, :] = hm_model.node_coordinates[
                    hm_model.node_ids[hm_node_id]]
            node_perm[hm_node_id] = vertex_id

    # finalize volumes array
    for hm_node_ids in volumes:
        for volume_node_index, hm_node_id in enumerate(hm_node_ids):
            hm_node_ids[volume_node_index] = node_perm[hm_node_id]

    mesh = Volumes(vertices=vertices, volumes=volumes)

    if create_face_groups:
        # determine subelement type
        subelement_type = HMComponent.element_types[element_type].subelement
        # get all faces in volume
        faces = mesh.get_faces()
        # transform to 1D tuple array
        tuple_dtype = ",".join(["i"] * faces.shape[1])
        faces_tuples = np.sort(faces).view(dtype=tuple_dtype).copy()

        # go through boundary components to import face groups
        for component in hm_model.components:
            # check if subelement type is contained
            if subelement_type in component.elements:
                group_faces = np.array(component.elements[subelement_type],
                        dtype=np.int32)
                for hm_vertex_ids in group_faces:
                    for index, hm_vertex_id in enumerate(hm_vertex_ids):
                        hm_vertex_ids[index] = node_perm[hm_vertex_id]
                group_faces_sorted = np.sort(group_faces)
                group_faces_tuples =\
                        group_faces_sorted.view(dtype=tuple_dtype)
                face_indices = np.intersect1d(faces_tuples, group_faces_tuples,
                        return_indices=True)[1]
                mesh.face_groups[component.name] = face_indices
            elif component != main_component:
                log.warning(f"Component '{component.name}' does not "\
                        f"contain any elements of type '{subelement_type}'.")

    return mesh

