"""gustaf/gustaf/io/mfem.py.

io functions for mfem. Supports simple linear elements (straight meshes)
For detailed information, see: https://mfem.org/mesh-
format-v1.0/#straight-meshes
"""

import numpy as np

from gustaf import settings
from gustaf.faces import Faces
from gustaf.volumes import Volumes

geometry_types = {
    "POINT": 0,
    "SEGMENT": 1,
    "TRIANGLE": 2,
    "SQUARE": 3,
    "TETRAHEDRON": 4,
    "CUBE": 5,
    "PRISM": 6,
}


def load(fname):
    """Load mesh in MFEM format. Loads vertices and their connectivity.
    Currently cannot process boundary.

    Parameters
    ------------
    fname: str

    Returns
    ------------
    mesh
    """

    def extract_values(fname, start_index, n_lines, total_lines, dtype):
        """Extract information from file. Reads [n_lines] lines from file
        [fname] starting at line [start_index] and returns array of type
        [dtype].

        Parameters
        ------------
        fname: str
        start_index: int
        n_lines: int
        total_lines: int
            Number of total lines of file
        dtype: (NumPy) data type

        Returns
        ------------
        NumPy Array
        """
        end_index = total_lines - (start_index + n_lines + 2)
        return np.genfromtxt(
            fname,
            delimiter=" ",
            skip_header=start_index,
            skip_footer=end_index,
            dtype=dtype,
        )

    with open(fname) as f:
        lines = f.readlines()
        total_lines = len(lines)

        # Read values of keywords
        keywords = ["dimension", "elements", "boundary", "vertices"]
        indices = [lines.index(f"{keyword}\n") for keyword in keywords]
        dimension, n_elements, n_boundaries, n_vertices = (
            int(lines[index + 1]) for index in indices
        )
        vdim = int(lines[indices[-1] + 2])
        # Extract values
        elements = extract_values(
            fname, indices[1] + 2, n_elements, total_lines, settings.INT_DTYPE
        )
        if elements.shape[0] != n_elements:
            raise ValueError("Number of elements do not match.")
        boundary = extract_values(
            fname,
            indices[2] + 2,
            n_boundaries,
            total_lines,
            settings.INT_DTYPE,
        )
        if boundary.shape[0] != n_boundaries:
            raise ValueError("Number of boundaries do not match.")
        vertices = extract_values(
            fname,
            indices[3] + 3,
            n_vertices,
            total_lines,
            settings.FLOAT_DTYPE,
        )
        if vertices.shape != (n_vertices, vdim):
            raise ValueError("Number of vertices do not match.")

        connectivity = elements[:, 2:]
        if dimension == 2:
            mesh = Faces(vertices=vertices, faces=connectivity)
        elif dimension == 3:
            mesh = Volumes(vertices=vertices, volumes=connectivity)

        return mesh


def export(fname, mesh):
    """Export mesh in MFEM format. Supports 2D triangle and quadrilateral
    meshes. Does not support different element attributes or difference in
    vertex dimension and mesh dimension.

    Parameters
    ------------
    fname: str
    mesh: Faces

    Returns
    ------------
    None
    """
    # Basic infos
    nvertices, dim = mesh.vertices.shape

    def format_array(array):
        """Format NumPy array into string. Each entry in a row is separated by
        a blank space and every row is separated by a new line.

        Parameters
        ------------
        array: NumPy array

        Returns
        ------------
        str
        """
        row_strings = [" ".join(map(str, row)) for row in array]
        return "\n".join(row_strings)

    # Export 2D mesh
    if dim == 2:
        # Elements
        element_attribute = 1  # Other numbers not yet supported
        elements = mesh.elements
        n_elements, n_element_vertices = elements.shape
        if n_element_vertices == 3:
            geometry_type = geometry_types["TRIANGLE"]
        elif n_element_vertices == 4:
            geometry_type = geometry_types["SQUARE"]
        else:
            raise NotImplementedError(
                "Sorry, we cannot export mesh with elements "
                f"with {n_element_vertices} vertices."
            )
        e = np.ones((n_elements, 1), dtype=settings.INT_DTYPE)
        elements_array = np.hstack(
            (element_attribute * e, geometry_type * e, elements)
        )
        elements_array_string = format_array(elements_array)
        elements_string = f"elements\n{n_elements}\n"
        elements_string += f"{elements_array_string}\n\n"

        # Boundary
        edges = mesh.edges()
        nboundary_edges = sum(map(len, mesh.BC.values()))
        boundary_array = np.empty(
            (nboundary_edges, 4), dtype=settings.INT_DTYPE
        )
        startrow = 0
        # Add boundary one by one as SEGMENTs
        for bid, edgeids in mesh.BC.items():
            nedges = len(edgeids)
            e = np.ones(nedges).reshape(-1, 1)
            vertex_list = edges[edgeids, :]
            boundary_array[startrow : (startrow + nedges), :] = np.hstack(
                (int(bid) * e, geometry_types["SEGMENT"] * e, vertex_list)
            )
            startrow += nedges

        boundary_array_string = format_array(boundary_array)
        boundary_string = f"boundary\n{nboundary_edges}\n"
        boundary_string += f"{boundary_array_string}\n\n"

        # Vertices
        vdim = 2  # Currently only option
        vertices_array_string = format_array(mesh.vertices)
        vertices_string = f"vertices\n{nvertices}\n{vdim}\n"
        vertices_string += f"{vertices_array_string}"

        with open(fname, "w") as f:
            f.write("MFEM mesh v1.0\n\n")
            f.write(f"dimension\n{dim}\n\n")
            f.write(elements_string)
            f.write(boundary_string)
            f.write(vertices_string)

    # Export 3D mesh
    else:
        raise NotImplementedError(
            f"Sorry, we cannot export mesh of dimension {dim}."
        )
