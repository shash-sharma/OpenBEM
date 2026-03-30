// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


#include <iostream>
#include <ctime>
#include <limits>
#include <cmath>
#include <vector>
#include <array>

#include "types.hpp"
#include "quadrature/line/gauss.hpp"

#include "rwg/assemblers/operator_matrix.hpp"

#include "rwg/integrators/src/strategic.hpp"
#include "rwg/integrators/obs/quadrature.hpp"
#include "rwg/operators/generic.hpp"

#include "matrix/eigen_dense.hpp"
#include "matrix/eigen_sparse.hpp"

#include "geometry/mesh/triangle_mesh.hpp"


using namespace bem;
using namespace bem::rwg;


const Float LAMBDA = 1;


void test_unit_cube(OperatorName op_name, Complex k)
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

    GaussTriangleQuadrature<2> src_tri_quad (4);
    GaussTriangleQuadrature<3> obs_tri_quad (4);
    GaussLineQuadrature<1> line_quad (10);

    // SingularitySubtractedTaylorHGF sthgf;
    // SrcSingularity src_int (src_tri_quad, sthgf);
    SrcStrategic src_int (SrcIntegrationSettings(), src_tri_quad, line_quad);
    ObsQuadrature obs_int (obs_tri_quad, src_int);

    EigenDenseMatrix<Complex> mat_dense;
    EigenSparseMatrix<Complex> mat_sparse;

    if (op_name == OperatorName::SCALAR_SINGLE_LAYER ||
        op_name == OperatorName::PULSE_PULSE)
    {
        GenericPulseOp op (op_name, obs_int);
        FaceOperatorAssembler face_matrix_assembler (mesh, mesh);
        face_matrix_assembler.assemble(mat_dense, op, k);
        face_matrix_assembler.assemble(mat_sparse, op, k);
    }
    else
    {
        GenericRwgOp op (op_name, obs_int);
        EdgeOperatorAssembler edge_matrix_assembler (mesh, mesh);
        edge_matrix_assembler.assemble(mat_dense, op, k);
        edge_matrix_assembler.assemble(mat_sparse, op, k);
    }

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


    EigenDenseMatrix<Complex> mat_error;
    mat_error.raw_matrix() = mat_dense.raw_matrix() - EigMat<Complex>(mat_sparse.raw_matrix());
    // mat_error.print();


    Float tol = std::abs(mat_dense.frobenius_norm()) * (float_eps * 1e4);
    bool pass = (std::abs(mat_error.raw_matrix().array().abs().maxCoeff()) <= tol);
    pass = pass && (std::abs(cond_dense - cond_sparse) <= tol);

    if (!pass)
    {
        std::cout << "====== operator " << op_name << " ======" << std::endl;
        std::cout << "--- FAIL ---\n  op_name: " << (int)op_name
                  << "\n  max error: " << mat_error.raw_matrix().array().abs().maxCoeff() << std::endl;
        std::cout << "\n  dense mat:" << std::endl;
        mat_dense.print();
        std::cout << "\n sparse mat:" << std::endl;
        mat_sparse.print();
    }

    return;

}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_operator_matrices.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    Float tol = 1e-3;
    Float f = c0 / LAMBDA;
    Complex k = two * pi * f * std::sqrt(eps0 * (one - (Float)0.1 * J) * mu0);

    std::vector<OperatorName> op_names ({
        OperatorName::VECTOR_SINGLE_LAYER,
        OperatorName::ROT_VECTOR_SINGLE_LAYER,
        OperatorName::VECTOR_DOUBLE_LAYER_PV,
        OperatorName::ROT_VECTOR_DOUBLE_LAYER_PV,
        OperatorName::SCALAR_SINGLE_LAYER,
        OperatorName::VECTOR_HYPERSINGULAR,
        OperatorName::ROT_VECTOR_HYPERSINGULAR,
        OperatorName::RWG_RWG,
        OperatorName::ROT_RWG_RWG
    });

    for (int ii = 0; ii < op_names.size(); ++ii)
    {
        OperatorName op_name = op_names[ii];
        Float tol = 1e-3;
        test_unit_cube(op_name, k);
    }

    return 0;

}
