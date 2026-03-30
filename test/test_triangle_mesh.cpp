// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


#include <iostream>
#include <stdexcept>

#include "types.hpp"
#include "constants.hpp"

#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/primitives/triangle.hpp"


using namespace bem;


TriangleMesh<3> test_unit_cube_triangles()
{

    Float tol = std::sqrt(float_eps);

    EigMatMN<Float, 3, 8> vertices;
    vertices.col(0) = EigColVecN<Float, 3> ({ 0, 0, 0 });
    vertices.col(1) = EigColVecN<Float, 3> ({ 1, 0, 0 });
    vertices.col(2) = EigColVecN<Float, 3> ({ 1, 1, 0 });
    vertices.col(3) = EigColVecN<Float, 3> ({ 0, 1, 0 });
    vertices.col(4) = EigColVecN<Float, 3> ({ 0, 0, 1 });
    vertices.col(5) = EigColVecN<Float, 3> ({ 1, 0, 1 });
    vertices.col(6) = EigColVecN<Float, 3> ({ 1, 1, 1 });
    vertices.col(7) = EigColVecN<Float, 3> ({ 0, 1, 1 });

    EigMatMN<Index, 3, 12> tri_vertices;
    tri_vertices.col(0) = EigColVecN<Index, 3> ({ 0, 3, 1 });
    tri_vertices.col(1) = EigColVecN<Index, 3> ({ 3, 2, 1 });
    tri_vertices.col(2) = EigColVecN<Index, 3> ({ 5, 6, 4 });
    tri_vertices.col(3) = EigColVecN<Index, 3> ({ 4, 6, 7 });
    tri_vertices.col(4) = EigColVecN<Index, 3> ({ 0, 7, 3 });
    tri_vertices.col(5) = EigColVecN<Index, 3> ({ 0, 4, 7 });
    tri_vertices.col(6) = EigColVecN<Index, 3> ({ 1, 2, 5 });
    tri_vertices.col(7) = EigColVecN<Index, 3> ({ 5, 2, 6 });
    tri_vertices.col(8) = EigColVecN<Index, 3> ({ 0, 1, 5 });
    tri_vertices.col(9) = EigColVecN<Index, 3> ({ 4, 0, 5 });
    tri_vertices.col(10) = EigColVecN<Index, 3> ({ 2, 3, 7 });
    tri_vertices.col(11) = EigColVecN<Index, 3> ({ 2, 7, 6 });

    TriangleMesh<3> mesh (vertices, tri_vertices);

    for (Index ii = 0; ii < mesh.elems().cols(); ++ii)
    {
        EigMatMN<Float, 3, 3> coords = mesh.verts()(Eigen::placeholders::all, mesh.elems(ii));
        Triangle<3> tri (coords);
        if (std::abs(tri.area() - 0.5) > tol)
            std::cout << "Fail: triangle idx: " << ii << ", vertices:\n" << coords << std::endl;
    }

    // IndexEigRowVec tags (3);
    // tags << 0, 1, 7;
    // std::vector<TriangleMesh<3>> meshes = mesh.partitions_by_tags(tags);
    // for (std::size_t ii = 0; ii < meshes.size(); ++ii)
    // {
    //     if (meshes[ii].elems().cols() != 1)
    //         std::cout << "Fail: mesh idx: " << ii << ", cols: " << meshes[ii].elems().cols() << std::endl;
    //     if (meshes[ii].face_tags(0) != tags[ii])
    //         std::cout << "Fail: mesh idx: " << ii << ", tags:\n" << meshes[ii].face_tags() << std::endl;
    // }

    EigMatMN<Float, 3, 2> bbox;
    bbox << -0.5, 1.1,
        -0.5, 1.2,
        -1e-4, 1e-4;

    TriangleMesh<3> mesh_bbox;
    mesh.partition_by_bbox(mesh_bbox, bbox);
    if (!(mesh_bbox.verts()(2, Eigen::placeholders::all).array() <= tol).all())
        std::cout << "Fail: mesh bbox, vertices:\n" << mesh_bbox.verts() << std::endl;

    return mesh;

}


TriangleMesh<3> test_two_unit_cubes_triangles()
{

    Float tol = std::sqrt(float_eps);

    EigMatMN<Float, 3, 16> vertices;
    vertices.col(0) = EigColVecN<Float, 3> ({ 0, 0, 0 });
    vertices.col(1) = EigColVecN<Float, 3> ({ 1, 0, 0 });
    vertices.col(2) = EigColVecN<Float, 3> ({ 1, 1, 0 });
    vertices.col(3) = EigColVecN<Float, 3> ({ 0, 1, 0 });
    vertices.col(4) = EigColVecN<Float, 3> ({ 0, 0, 1 });
    vertices.col(5) = EigColVecN<Float, 3> ({ 1, 0, 1 });
    vertices.col(6) = EigColVecN<Float, 3> ({ 1, 1, 1 });
    vertices.col(7) = EigColVecN<Float, 3> ({ 0, 1, 1 });
    vertices.col(8) = EigColVecN<Float, 3> ({ 2, 1, -1 });
    vertices.col(9) = EigColVecN<Float, 3> ({ 3, 1, -1 });
    vertices.col(10) = EigColVecN<Float, 3> ({ 3, 2, -1 });
    vertices.col(11) = EigColVecN<Float, 3> ({ 2, 2, -1 });
    vertices.col(12) = EigColVecN<Float, 3> ({ 2, 1, 0 });
    vertices.col(13) = EigColVecN<Float, 3> ({ 3, 1, 0 });
    vertices.col(14) = EigColVecN<Float, 3> ({ 3, 2, 0 });
    vertices.col(15) = EigColVecN<Float, 3> ({ 2, 2, 0 });

    EigMatMN<Index, 3, 24> tri_vertices;
    tri_vertices.col(0) = EigColVecN<Index, 3> ({ 0, 3, 1 });
    tri_vertices.col(1) = EigColVecN<Index, 3> ({ 3, 2, 1 });
    tri_vertices.col(2) = EigColVecN<Index, 3> ({ 5, 6, 4 });
    tri_vertices.col(3) = EigColVecN<Index, 3> ({ 4, 6, 7 });
    tri_vertices.col(4) = EigColVecN<Index, 3> ({ 0, 7, 3 });
    tri_vertices.col(5) = EigColVecN<Index, 3> ({ 0, 4, 7 });
    tri_vertices.col(6) = EigColVecN<Index, 3> ({ 1, 2, 5 });
    tri_vertices.col(7) = EigColVecN<Index, 3> ({ 5, 2, 6 });
    tri_vertices.col(8) = EigColVecN<Index, 3> ({ 0, 1, 5 });
    tri_vertices.col(9) = EigColVecN<Index, 3> ({ 4, 0, 5 });
    tri_vertices.col(10) = EigColVecN<Index, 3> ({ 2, 3, 7 });
    tri_vertices.col(11) = EigColVecN<Index, 3> ({ 2, 7, 6 });
    tri_vertices.col(12) = EigColVecN<Index, 3> ({ 8, 11, 9 });
    tri_vertices.col(13) = EigColVecN<Index, 3> ({ 11, 10, 9 });
    tri_vertices.col(14) = EigColVecN<Index, 3> ({ 13, 14, 12 });
    tri_vertices.col(15) = EigColVecN<Index, 3> ({ 12, 14, 15 });
    tri_vertices.col(16) = EigColVecN<Index, 3> ({ 8, 15, 11 });
    tri_vertices.col(17) = EigColVecN<Index, 3> ({ 8, 12, 15 });
    tri_vertices.col(18) = EigColVecN<Index, 3> ({ 9, 10, 13 });
    tri_vertices.col(19) = EigColVecN<Index, 3> ({ 13, 10, 14 });
    tri_vertices.col(20) = EigColVecN<Index, 3> ({ 8, 9, 13 });
    tri_vertices.col(21) = EigColVecN<Index, 3> ({ 12, 8, 13 });
    tri_vertices.col(22) = EigColVecN<Index, 3> ({ 10, 11, 15 });
    tri_vertices.col(23) = EigColVecN<Index, 3> ({ 10, 15, 14 });

    TriangleMesh<3> mesh (vertices, tri_vertices);
    TriangleMesh<3> mesh0 = test_unit_cube_triangles();

    EigMatMN<Float, 3, 2> bbox;
    bbox << -0.5, 1.1,
        -0.2, 1.2,
        -1e-4, 1.01;

    // IndexEigRowVec tags (1);
    // tags << 0;
    // std::vector<TriangleMesh<3>> mesh0_tags = mesh.partitions_by_tags(tags);
    TriangleMesh<3> mesh0_bbox;
    mesh.partition_by_bbox(mesh0_bbox, bbox);

    // if (mesh0_tags.size() != 1)
    //     std::cout << "Fail: mesh0_test size: " << mesh0_tags.size() << std::endl;

    // if (!(
    //     mesh0_tags[0].verts().rowwise().maxCoeff().array() <
    //     bbox.rowwise().maxCoeff().array()
    //     ).all() || !(
    //     mesh0_tags[0].verts().rowwise().minCoeff().array() >
    //     bbox.rowwise().minCoeff().array()
    //     ).all())
    //     std::cout << "Fail: mesh0_tags verts:\n" << mesh0_tags[0].verts() << std::endl;

    if (!(
            mesh0_bbox.verts().rowwise().maxCoeff().array() <
            bbox.rowwise().maxCoeff().array()
            ).all() || !(
                mesh0_bbox.verts().rowwise().minCoeff().array() >
                bbox.rowwise().minCoeff().array()
                ).all())
        std::cout << "Fail: mesh0_bbox verts:\n" << mesh0_bbox.verts() << std::endl;

    return mesh;

}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_triangle_mesh.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    TriangleMesh<3> unit_cube = test_unit_cube_triangles();
    TriangleMesh<3> two_cubes = test_two_unit_cubes_triangles();

    return 0;

}
