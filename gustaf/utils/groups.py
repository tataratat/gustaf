"""gustaf/gustaf/utils/groups.py

Collection classes for mesh entity groups.

So far, vertex and face groups can be stored. Volume and edge groups are
missing.
"""
import numpy as np

def element_to_vertex_group(element_connectivity, element_ids):
    """
    Take an element connectivity array and IDs referencing the faces and create
    a corresponding list of unique vertices.

    Elements could be edges, faces, or volumes.

    Parameters
    -----------
    element_connectivity: (ne, nn) np.ndarray
    element_ids: (n) np.ndarray

    Returns
    --------
    vertex_ids: (n) np.ndarray
    """
    vertex_ids = np.unique(element_connectivity[element_ids])
    return vertex_ids

def vertex_to_element_group(element_connectivity, vertex_ids):
    """
    Take an element connectivity array and IDs referencing the vertices and
    create a corresponding list of element IDs that contain all the vertices.

    Note that this operation is not always uniquely defined. In contiguous
    regions, individual elements could be left out even when all vertices are
    used. This implementation uses a greedy approach that adds all possible
    elements.

    Parameters
    -----------
    element_connectivity: (ne, nn) np.ndarray
    vertex_ids: (n) np.ndarray

    Returns
    --------
    element_ids: (n) np.ndarray
    """
    # create array of booleans that are True for all the elements in the
    # group
    element_in_group = np.isin(element_connectivity, vertex_ids).all(axis=1)
    element_ids = element_in_group.nonzero()[0]
    return element_ids

class VertexGroupCollection(dict):
    def __init__(self, mesh):
        """
        Constructs vertex group object.

        The vertex group is tied to a specific mesh instance, i.e., an instance
        of some subclass of Vertices.

        We cannot explicitly check this since that would cause a cyclic
        dependency.

        Parameters
        -----------
        mesh: Vertices

        Returns
        --------
        None
        """
        self.mesh = mesh

    def __setitem__(self, group_name, vertex_ids):
        """
        Add a group to the collection.

        This overrides dict's assignment routine to run some additional checks
        on the provided data.

        Parameters
        -----------
        group_name: str
        vertex_ids: (n, 1) np.ndarray

        Returns
        --------
        None
        """
        assert isinstance(vertex_ids, np.ndarray)
        assert np.issubdtype(vertex_ids.dtype, np.integer)
        assert vertex_ids.ndim == 1
        assert np.less(vertex_ids, self.mesh.vertices.shape[0]).all(),\
                f"Invalid vertex index in vertex group '{group_name}'."
        dict.__setitem__(self, group_name, vertex_ids)

class FaceGroupCollection(dict):
    def __init__(self, mesh):
        """
        Construct face group object.

        The face group is tied to a specific mesh instance, i.e., either a Faces
        or Volumes instance.

        We cannot explicitly check this since that would cause a cyclic
        dependency.

        Parameters
        -----------
        mesh: Faces

        Returns
        --------
        None
        """
        self.mesh = mesh

    def __setitem__(self, group_name, face_ids):
        """
        Add a group to the collection.

        This overrides dict's assignment routine to run some additional checks
        on the provided data.

        Parameters
        -----------
        group_name: str
        face_ids: (n, 1) np.ndarray

        Returns
        --------
        None
        """
        assert isinstance(face_ids, np.ndarray)
        assert np.issubdtype(face_ids.dtype, np.integer)
        assert face_ids.ndim == 1
        assert np.less(face_ids, self.mesh.get_number_of_faces()).all(),\
                f"Invalid face index in face group '{group_name}'."
        dict.__setitem__(self, group_name, face_ids)

    def import_vertex_group(self, group_name):
        """
        Convert a vertex group to a face group.

        This takes the vertex group specified by `group_name` from the mesh and
        creates a face group that contains all faces whose vertices are in the
        group.

        Note that this operation is not always uniquely defined. In contiguous
        surfaces, individual faces could be left out even when all vertices are
        used. This implementation uses a greedy approach that adds all possible
        faces.

        Parameters
        -----------
        group_name: str

        Returns
        --------
        None
        """
        vertex_group = self.mesh.vertex_groups[group_name]
        all_faces = self.mesh.get_faces()
        self[group_name] = vertex_to_element_group(all_faces, vertex_group)

    def import_all_vertex_groups(self):
        """
        Take all the vertex groups in the mesh and create corresponding face
        groups of the same name.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        all_faces = self.mesh.get_faces()
        for group_name, vertex_ids in self.mesh.vertex_groups.items():
            self[group_name] = vertex_to_element_group(all_faces, vertex_ids)

    def export_vertex_group(self, group_name):
        """
        Convert a face group to a vertex group.

        This takes the face group specified by `group_name` and creates a group
        of all vertices that are touched by the faces.

        This implementation creates the complete face connectivity array, even
        though only a subset would really be required.

        Parameters
        -----------
        group_name: str

        Returns
        --------
        None
        """
        all_faces = self.mesh.get_faces()
        face_group = self[group_name]
        self.mesh.vertex_groups[group_name] = element_to_vertex_group(
                all_faces, face_group)

    def export_all_vertex_groups(self):
        """
        Take all the face groups and add corresponding vertex groups of the same
        name to the mesh.

        Parameters
        -----------
        None

        Returns
        --------
        None
        """
        all_faces = self.mesh.get_faces()
        for group_name, face_ids in self.items():
            self.mesh.vertex_groups[group_name] = element_to_vertex_group(
                    all_faces, face_ids)

