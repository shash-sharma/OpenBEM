// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


#ifndef TEST_RWG_OPERATOR_MATS_H
#define TEST_RWG_OPERATOR_MATS_H

#include <iostream>
#include <ctime>
#include <limits>
#include <cmath>
#include <vector>
#include <array>
#include <tuple>

#include "test_operator_matrices.hpp"

#include "types.hpp"
#include "quadrature/line/gauss.hpp"

#include "rwg/assemblers/operator_matrix.hpp"

#include "rwg/integrators/src/strategic.hpp"
#include "rwg/integrators/obs/quadrature.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"

#include "matrix/eigen_matrix.hpp"

#include "geometry/mesh/triangle_mesh.hpp"


using namespace bem;
using namespace bem::rwg;


template <typename OpType>
void test_unit_cube(OpType& op, Complex k)
{

    EigMatMN<Float, 3, 8> vertices;
    vertices.col(0) = EigColVecN<Float, 3> ({ 0, 0, 0 });
    vertices.col(1) = EigColVecN<Float, 3> ({ 1, 0, 0 });
    vertices.col(2) = EigColVecN<Float, 3> ({ 1, 1, 0 });
    vertices.col(3) = EigColVecN<Float, 3> ({ 0, 1, 0 });
    vertices.col(4) = EigColVecN<Float, 3> ({ 0, 0, 1 });
    vertices.col(5) = EigColVecN<Float, 3> ({ 1, 0, 1 });
    vertices.col(6) = EigColVecN<Float, 3> ({ 1, 1, 1 });
    vertices.col(7) = EigColVecN<Float, 3> ({ 0, 1, 1 });

    vertices /= 10;

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

    std::vector<Triangle<3>> tris;
    tris.push_back(Triangle<3> ({ vertices.col(0), vertices.col(3), vertices.col(1) }));
    tris.push_back(Triangle<3> ({ vertices.col(3), vertices.col(2), vertices.col(1) }));
    tris.push_back(Triangle<3> ({ vertices.col(5), vertices.col(6), vertices.col(4) }));
    tris.push_back(Triangle<3> ({ vertices.col(4), vertices.col(6), vertices.col(7) }));
    tris.push_back(Triangle<3> ({ vertices.col(0), vertices.col(7), vertices.col(3) }));
    tris.push_back(Triangle<3> ({ vertices.col(0), vertices.col(4), vertices.col(7) }));
    tris.push_back(Triangle<3> ({ vertices.col(1), vertices.col(2), vertices.col(5) }));
    tris.push_back(Triangle<3> ({ vertices.col(5), vertices.col(2), vertices.col(6) }));
    tris.push_back(Triangle<3> ({ vertices.col(0), vertices.col(1), vertices.col(5) }));
    tris.push_back(Triangle<3> ({ vertices.col(4), vertices.col(0), vertices.col(5) }));
    tris.push_back(Triangle<3> ({ vertices.col(2), vertices.col(3), vertices.col(7) }));
    tris.push_back(Triangle<3> ({ vertices.col(2), vertices.col(7), vertices.col(6) }));

    TriangleMesh<3> mesh (vertices, tri_vertices);

    // std::cout << mesh.vertex_coords() << std::endl;
    // std::cout << "--" << std::endl;
    // std::cout << mesh.face_vertices() << std::endl;
    // std::cout << "--" << std::endl;
    // std::cout << mesh.face_edges() << std::endl;
    // std::cout << "--" << std::endl;
    // std::cout << mesh.face_edge_polarities() << std::endl;
    // std::cout << "--" << std::endl;
    // std::cout << mesh.edge_vertices() << std::endl;

    EigenMatrix<Complex, EigenMatrixType::EIGEN_DENSE> mat_dense;
    EigenMatrix<Complex, EigenMatrixType::EIGEN_SPARSE> mat_sparse;

    OperatorAssembler<
        typename OpType::TestSpaceType,
        typename OpType::ExpansionSpaceType
        > assembler (mesh);

    assembler.assemble(mat_dense, op, k);
    assembler.assemble(mat_sparse, op, k);

    Eigen::JacobiSVD<EigMat<Complex>> svd_dense (mat_dense.raw_matrix());
    Float cond_dense = svd_dense.singularValues()(0) /
                       svd_dense.singularValues()(svd_dense.singularValues().size() - 1);

    // mat_dense.print();
    // std::cout << cond_dense << std::endl;


    Eigen::JacobiSVD<EigMat<Complex>> svd_sparse (EigMat<Complex>(mat_sparse.raw_matrix()));
    Float cond_sparse = svd_sparse.singularValues()(0) /
                        svd_sparse.singularValues()(svd_sparse.singularValues().size() - 1);

    // mat_sparse.print();
    // std::cout << cond_sparse << std::endl;


    EigenMatrix<Complex> mat_error;
    mat_error.raw_matrix() = mat_dense.raw_matrix() - EigMat<Complex>(mat_sparse.raw_matrix());
    // mat_error.print();


    Float tol = std::abs(mat_dense.frobenius_norm()) * (float_eps * 1e4);
    bool pass = (std::abs(mat_error.raw_matrix().array().abs().maxCoeff()) <= tol);
    pass = pass && (std::abs(cond_dense - cond_sparse) <= tol);

    if (!pass)
    {
        std::cout << "====== operator " << typeid(op).name() << " ======" << std::endl;
        std::cout << "--- FAIL ---\n  op_name: " << typeid(op).name()
                  << "\n  max error: " << mat_error.raw_matrix().array().abs().maxCoeff() << std::endl;
        std::cout << "\n  dense mat:" << std::endl;
        mat_dense.print();
        std::cout << "\n sparse mat:" << std::endl;
        mat_sparse.print();
    }

    return;

}

#endif
