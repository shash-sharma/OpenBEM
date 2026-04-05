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
#include "materials.hpp"

#include "geometry/mesh/triangle_mesh.hpp"


using namespace bem;


TriangleMesh<3> test_two_unit_cubes()
{

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

    // IndexEigRowVec tri_tags = IndexEigRowVec::Constant(1, tri_vertices.cols(), 0);
    // tri_tags.rightCols(tri_vertices.cols() / 2) = IndexEigRowVec::Constant(
    //     1, tri_vertices.cols() / 2, 1
    //     );
    TriangleMesh<3> mesh (vertices, tri_vertices);

    // Material mat0 (1, 1, 1);
    // Material mat1 (2, 2, 1);
    // PerfectDielectricMaterial mat2 (3, 1);
    // ConstantLossTangentMaterial mat3 (4, 3, 0.01);

    // mesh.set_background_material(mat0);
    // mesh.set_material_by_tag(0, mat1);
    // mesh.set_material_by_tag(1, mat2);

    // IndexEigRowVec tags (1);
    // tags << 1;
    // TriangleMesh<3> mesh1 = mesh.partitions_by_tags(IndexEigRowVec::Constant(1, 1, 1))[0];

    // if (mesh1.tags().size() != 1)
    //     std::cout << "Fail: mesh1.tags().size() = 1, got " << mesh1.tags().size() << std::endl;

    // if (mesh1.tags(0) != 1)
    //     std::cout << "Fail: mesh1.tags(0) = 1, got " << mesh1.tags(0) << std::endl;

    // if (std::abs(mesh1.material_by_tag(mesh1.tags(0)).epsr() - 3.0) > 1e1 * float_eps)
    //     std::cout << "Fail: expected epsr = 3, got " << mesh1.material_by_tag(mesh1.tags(0)).epsr() << std::endl;

    // mesh1.set_background_material(mat2);
    // mesh1.set_material_by_tag(0, mat0);
    // mesh1.set_material_by_tag(1, mat3);

    // TriangleMesh<3> mesh2 = mesh1.partitions_by_tags(IndexEigRowVec::Constant(1, 1, 1))[0];

    // if (mesh2.tags().size() != 1)
    //     std::cout << "Fail: mesh2.tags().size() = 1, got " << mesh2.tags().size() << std::endl;

    // if (mesh2.tags(0) != 1)
    //     std::cout << "Fail: mesh2.tags(0) = 1, got " << mesh2.tags(0) << std::endl;

    // if (std::abs(mesh2.material_by_tag(mesh2.tags(0)).loss_tan(1) - 0.01) > 1e1 * float_eps)
    //     std::cout << "Fail: expected loss_tan = 0.01, got " << mesh2.material_by_tag(mesh2.tags(0)).loss_tan(1) << std::endl;

    // if (std::abs(mesh2.background_material().epsr() - 3.0) > 1e1 * float_eps)
    //     std::cout << "Fail: expected background epsr = 3, got " << mesh2.background_material().epsr() << std::endl;

    return mesh;

}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_materials.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    TriangleMesh<3> two_cubes = test_two_unit_cubes();

    return 0;

}
