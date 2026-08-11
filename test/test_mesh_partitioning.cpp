// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


#include <iostream>

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/mesh/triangle_mesh_view.hpp"


using namespace bem;


void test_partition_root_correspondence()
{

    // Two unit squares glued along a shared edge, so a view spanning both tags below touches
    // a boundary edge (excluded neighbor), an internal edge (shared within a tag), and crosses
    // a component boundary.
    EigMatMN<Float, 3, 6> vertices;
    vertices.col(0) = EigColVecN<Float, 3> ({ 0, 0, 0 });
    vertices.col(1) = EigColVecN<Float, 3> ({ 1, 0, 0 });
    vertices.col(2) = EigColVecN<Float, 3> ({ 1, 1, 0 });
    vertices.col(3) = EigColVecN<Float, 3> ({ 0, 1, 0 });
    vertices.col(4) = EigColVecN<Float, 3> ({ 2, 0, 0 });
    vertices.col(5) = EigColVecN<Float, 3> ({ 2, 1, 0 });

    EigMatMN<Index, 3, 4> faces;
    faces.col(0) = EigColVecN<Index, 3> ({ 0, 1, 2 });
    faces.col(1) = EigColVecN<Index, 3> ({ 0, 2, 3 });
    faces.col(2) = EigColVecN<Index, 3> ({ 1, 4, 5 });
    faces.col(3) = EigColVecN<Index, 3> ({ 1, 5, 2 });

    EigRowVec<Index> face_tags(1, 4);
    face_tags << 0, 0, 1, 1;

    TriangleMesh<3> root (vertices, faces, face_tags, false);

    if (root.is_partition())
        std::cout << "Fail: root.is_partition() should be false" << std::endl;

    if (root.face_primitive(2).idx() != 2)
        std::cout << "Fail: root face_primitive idx should equal local index" << std::endl;

    // View spans both tags: faces {1, 2, 3} (face 0 excluded).
    EigRowVec<Index> view_faces(1, 3);
    view_faces << 1, 2, 3;
    TriangleMeshView<3> view (root, view_faces, "multi-tag view");

    TriangleMesh<3> partition = view.mesh();

    if (!partition.is_partition())
        std::cout << "Fail: partition.is_partition() should be true" << std::endl;

    if (partition.root_faces().size() != view.face_inds().size())
        std::cout << "Fail: root_faces()/face_inds() size mismatch" << std::endl;

    for (Index i = 0; i < view.face_inds().size(); ++i)
        if (partition.root_faces()[i] != view.face_inds()[i])
            std::cout << "Fail: root_faces()[" << i << "] != face_inds()[" << i << "]" << std::endl;

    // The core invariant this test exists for: view.edge_inds()[k] must correspond to the k-th
    // edge of the materialized partition, i.e. root_edges()[k] == edge_inds()[k], for every k.
    if (partition.root_edges().size() != view.edge_inds().size())
        std::cout << "Fail: root_edges()/edge_inds() size mismatch" << std::endl;

    for (Index i = 0; i < view.edge_inds().size(); ++i)
        if (partition.root_edges()[i] != view.edge_inds()[i])
            std::cout << "Fail: root_edges()[" << i << "] != edge_inds()[" << i << "]" << std::endl;

    // face_primitive() on a partition must stamp the true root face index, not the local one.
    for (Index i = 0; i < partition.num_faces(); ++i)
        if (partition.face_primitive(i).idx() != view.face_inds()[i])
            std::cout << "Fail: partition.face_primitive(" << i << ").idx() != root face" << std::endl;

    // Every partition vertex's root_vertex() must point at the coincident root vertex.
    for (Index i = 0; i < partition.num_vertices(); ++i)
        if (!(partition.vertices().col(i).isApprox(root.vertices().col(partition.root_vertex(i)))))
            std::cout << "Fail: partition vertex " << i << " coords don't match its root vertex" << std::endl;

    // Faces 2 and 3 share an edge that must be internal to the partition; the edge face 1 would
    // have shared with the excluded face 0 must be boundary within the partition.
    if (partition.internal_edges().size() < 1)
        std::cout << "Fail: expected at least one internal edge in the partition" << std::endl;

    if (partition.boundary_edges().size() < 1)
        std::cout << "Fail: expected at least one boundary edge in the partition" << std::endl;

    return;

}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_mesh_partitioning.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    test_partition_root_correspondence();

    return 0;

}
