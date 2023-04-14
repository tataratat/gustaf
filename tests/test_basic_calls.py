import unittest

import pytest

import gustaf as gus


@pytest.mark.usefixtures("provide_data_to_unittest")
class BasicCallsTest(unittest.TestCase):
    """
    Checks basic calls - calls without specific arguments + without further
    dependencies.
    Does not really
    """

    def test_vertices_basics(self):
        """
        Call available properties/methods-without-args of Vertices.
        """
        v = gus.Vertices(self.V)
        v.vertices = v.vertices
        v.const_vertices
        v.whatami
        # v.unique_vertices()
        v.bounds()
        v.bounds_diagonal()
        v.bounds_diagonal_norm()
        v.select_vertices([[-1, 0.5], [-1, 0.5], [-1, 0.5]])
        v.copy()

        # v.update_vertices()
        # v.remove_vertices()
        # v.merge_vertices()
        # v.showable()
        # v.show()
        # gus.Vertices.concat()

    def test_edges_basics(self):
        """ """
        es = gus.Edges(self.V, self.E)
        es.edges = es.edges
        es.const_edges
        es.whatami
        es.sorted_edges()
        es.unique_edges()
        es.single_edges()
        es.elements = es.elements
        es.const_elements
        es.centers()
        es.referenced_vertices()
        es.dashed()
        es.shrink()
        es.to_vertices()
        es.single_edges()
        es.remove_unreferenced_vertices()

        # es.update_elements()
        # es.update_edges()

    def test_faces_basics(self):
        for fs in (gus.Faces(self.V, self.TF), gus.Faces(self.V, self.QF)):
            fs.edges()
            fs.whatami
            fs.faces = fs.faces
            fs.const_faces
            fs.sorted_faces()
            fs.unique_faces()
            fs.single_faces()

            fs.sorted_edges()
            fs.unique_edges()
            fs.single_edges()
            fs.centers()
            fs.referenced_vertices()
            fs.shrink()
            fs.to_vertices()
            fs.remove_unreferenced_vertices()

            # fs.update_faces()
            # gus.Faces.whatareyou()

    def test_volumes_bascis(self):
        for vs in (gus.Volumes(self.V, self.TV), gus.Volumes(self.V, self.HV)):
            vs.faces()
            vs.whatami
            vs.volumes = vs.volumes
            vs.const_volumes
            vs.sorted_volumes()
            vs.unique_volumes()
            vs.to_faces()

            vs.sorted_edges()
            vs.unique_edges()
            vs.single_edges()
            vs.centers()
            vs.referenced_vertices()
            vs.shrink()
            vs.to_vertices()
            vs.remove_unreferenced_vertices()

            vs.sorted_faces()
            vs.unique_faces()
            vs.single_faces()
            # gus.Faces.whatareyou()
