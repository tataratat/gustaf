"""gustaf/gustaf/io/hmascii.py

Read a mesh and boundaries from a HyperMesh .hmascii file.
Output is not supported.

This implementation requires a certain structure of the model. It only imports
volume elements of a single type, which is either tetrahedra or hexahedra.
They all must be in a single component (default name: 'volume').
Boundaries can be imported as vertex groups, which can be converted to face
groups. To enable the creation of a vertex group, a component must be created
in HyperMesh. This component must contain surface elements of the current
subelement type. They must use the same node IDs as the volume elements and
not just share their coordinates.
An initial surface component can be created in HyperMesh using:
tools -> faces
"""

import logging
import numpy as np

from gustaf.faces import Faces
from gustaf.volumes import Volumes
from gustaf.edges import Edges


class HMLine:
    """Parse a line from an HMASCII file.
    """

    __slots__ = ["name", "values"]

    def __init__(self, line):
        """
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
                self.values = [
                        value.strip('"') for value in parts[1][:-2].split(',')
                ]


class HMElementType:
    """Store HyperMesh element type information.
    """

    __slots__ = ["number_of_nodes", "subelement", "MeshType"]

    def __init__(self, number_of_nodes, MeshType, subelement=''):
        """
        Parameters
        -----------
        number_of_nodes: int
        subelement: str
        MeshType: Mesh

        Returns
        --------
        None
        """
        self.number_of_nodes = int(number_of_nodes)
        self.MeshType = MeshType
        self.subelement = str(subelement)


class HMComponent:
    """Create a component from an HMASCII line.
    """
    __slots__ = ["name", "elements"]

    element_types = {
            'tetra4': HMElementType(4, Volumes, 'tria3'),
            'hexa8': HMElementType(8, Volumes, 'quad4'),
            'tria3': HMElementType(3, Faces, 'plotel'),
            'quad4': HMElementType(4, Faces, 'plotel'),
            'plotel': HMElementType(2, Edges),
    }
    element_type_preference = ('hexa8', 'tetra4', 'quad4', 'tria3')

    def __init__(self, line):
        """
        Parameters
        -----------
        line: HMLine

        Returns
        --------
        None
        """
        self.name = str(line.values[1])
        self.elements = dict()  # element type: list()

    def add_element(self, line):
        """Parse an element line from an HMASCII file.

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
        self.elements[element_type].append(
                [
                        int(node) for node in
                        line.values[2:2 + self.element_types[element_type].
                                    number_of_nodes]
                ]
        )


class HMModel:
    """Create a representation of a HyperMesh model from the HMASCII file
    specified by `filename`.

    Parameters
    -----------
    node_ids: dict
    node_coordinates: list
    components: list

    Returns
    --------
    None
    """

    __slots__ = ["node_ids", "node_coordinates", "components"]

    def __init__(self, filename):
        """
        Parameters
        -----------
        filename: str

        Returns
        --------
        None
        """
        # read file into arrays
        self.node_ids = dict()  # HM node id: internal node id
        self.node_coordinates = list()
        self.components = list()  # HMComponent objects

        current_component = None

        with open(filename, 'r') as hm_file:
            for line_string in hm_file:
                line = HMLine(line_string)

                # read node
                if line.name == "node":
                    self.node_ids[int(line.values[0])] =\
                        len(self.node_coordinates)
                    self.node_coordinates.append(
                            [float(coord) for coord in line.values[1:4]]
                    )

                # read component
                elif line.name == "component":
                    current_component = HMComponent(line)
                    self.components.append(current_component)

                # read element
                elif line.name in HMComponent.element_types:
                    if not current_component:
                        raise RuntimeError(
                                'Encountered element before first '
                                'component.'
                        )
                    current_component.add_element(line)


def load(fname, element_type=''):
    """hmascii load.

    Parameters
    -----------
    fname: str
        The filename ending in '.hmascii'.
    element_type: str
        The volume element type to be imported. If this is not specified, the
        most common element type in the main component is chosen.

    Returns
    -----------
    mesh: Mesh
    """

    if not fname.endswith(".hmascii"):
        raise TypeError("Input file must be of Type .hmascii .")

    hm_model = HMModel(fname)

    if not element_type:
        # which element types occur in the mesh?
        element_types_in_model = [
                element_type for component in hm_model.components
                for element_type in component.elements
        ]
        preferred_element_types = [
                element_type
                for element_type in HMComponent.element_type_preference
                if element_type in element_types_in_model
        ]
        if len(preferred_element_types) < 1:
            raise RuntimeError(
                    "Couldn't find any usable element types in "
                    "model."
            )

        element_type = preferred_element_types[0]
        logging.info(f"Selected volume element type '{element_type}'.")

    # determine subelement type
    subelement_type = HMComponent.element_types[element_type].subelement

    hm_volume_elements_nonunique = np.ndarray(
            shape=(0, HMComponent.element_types[element_type].number_of_nodes),
            dtype=int
    )

    bcs = dict()

    # loop over all components to find volume elements
    for hm_component in hm_model.components:
        # can we use all elements?
        ignored_element_types = set(hm_component.elements).difference(
                {element_type, subelement_type}
        )
        if len(ignored_element_types) > 0:
            logging.warning(
                    f"Component '{hm_component.name}' contains "
                    f"unkown element types {ignored_element_types}. "
                    "They will be ignored."
            )

        # are there volume elements? append.
        if element_type in hm_component.elements:
            elements_in_component = hm_component.elements[element_type]

            # append elements
            hm_volume_elements_nonunique = np.concatenate(
                    (hm_volume_elements_nonunique, elements_in_component)
            )

        # try get bounds
        if subelement_type in hm_component.elements and subelement_type in [
                'tria3', 'quad4'
        ]:

            bcs[hm_component.name] = (
                    np.arange(len(elements_in_component))
                    + hm_volume_elements_nonunique.shape[0]
            )
        else:
            logging.info("Can`t find any bounds.")

    # create unique element list
    hm_volume_elements_sorted = np.sort(hm_volume_elements_nonunique, axis=1)
    hm_volume_elements_unique_indices = np.unique(
            hm_volume_elements_sorted, return_index=True, axis=0
    )[1]

    # sorting the unique indices isn't necessary, but it might maintain a more
    # contiguous element order
    volumes = np.squeeze(
            hm_volume_elements_nonunique[
                    hm_volume_elements_unique_indices.sort()]
    )

    # create minimal vertex array
    hm_node_indices = np.unique(volumes)
    number_of_vertices = len(hm_node_indices)
    if number_of_vertices == len(hm_model.node_coordinates):
        vertices = np.array(hm_model.node_coordinates, dtype=np.float64)
        node_perm = hm_model.node_ids
    else:
        # filter nodes
        vertices = np.full((number_of_vertices, 3), 0.0)
        node_perm = dict()
        for vertex_id, hm_node_id in enumerate(hm_node_indices):
            vertices[vertex_id, :] = hm_model.node_coordinates[
                    hm_model.node_ids[hm_node_id]]
            node_perm[hm_node_id] = vertex_id

    # finalize volumes array
    for hm_node_ids in volumes:
        for volume_node_index, hm_node_id in enumerate(hm_node_ids):
            hm_node_ids[volume_node_index] = node_perm[hm_node_id]

    MeshType = HMComponent.element_types[element_type].MeshType
    mesh = MeshType(vertices=vertices, elements=volumes)

    # bc
    if len(bcs) != 0:
        mesh.BC = bcs

    return mesh
