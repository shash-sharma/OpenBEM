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

#include "kernels/hgf.hpp"
#include "quadrature/triangle/gauss.hpp"
#include "geometry/point_cloud.hpp"

#include "rwg/integrators/src/quadrature.hpp"

#include "matrix/eigen_dense.hpp"
#include "rwg/assemblers/projector_matrix.hpp"

#include "rwg/projectors/base.hpp"
#include "rwg/projectors/single_layer.hpp"
#include "rwg/projectors/double_layer.hpp"

#include "geometry/mesh/triangle_mesh.hpp"


using namespace bem;
using namespace bem::rwg;


const Float LAMBDA = 1;


void test_unit_cube_rwg_g(Complex k)
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

    Triangle<3> tri (vertices(Eigen::placeholders::all, tri_vertices.col(0)));
    EigMatMN<Float, 3, 2> obs_points;
    obs_points << 2, 3,
                  3, 3,
                  3, 5;

    GaussTriangleQuadrature<2> tri_quad (4);
    SrcQuadrature src_int (tri_quad, HGF());

    VectorSingleLayerProj proj_1 (src_int);
    GradScalarSingleLayerProj proj_2 (src_int);
    VectorHypersingularProj proj_3 (src_int);
    VectorDoubleLayerProj proj_4 (src_int);
    ScalarSingleLayerProj proj_5 (src_int);
    // std::cout << proj_1.compute(k, obs_points, tri) << std::endl;

    TriangleMesh<3> mesh (vertices, tri_vertices);

    EigColVecN<Float, 3> start, stop;
    EigColVecN<Index, 3> num_pts;
    start << 20, 0, half_pi;
    stop << 20, two_pi, half_pi;
    num_pts << 1, 9, 1;
    EigColVecN<Float, 3> center;
    center << 0, 0, 0;

    PointCloud<3> cloud;
    cloud.set_polar_data(start, stop, center, num_pts);


    EigenDenseMatrix<Complex> mat;

    EdgeProjectorAssembler<3> proj_1_mat (cloud, mesh);
    proj_1_mat.assemble(mat, proj_1, k);

    FaceProjectorAssembler<3> proj_2_mat (cloud, mesh);
    proj_2_mat.assemble(mat, proj_2, k);

    EdgeProjectorAssembler<3> proj_3_mat (cloud, mesh);
    proj_3_mat.assemble(mat, proj_3, k);

    EdgeProjectorAssembler<3> proj_4_mat (cloud, mesh);
    proj_4_mat.assemble(mat, proj_4, k);

    FaceProjectorAssembler<1> proj_5_mat (cloud, mesh);
    proj_5_mat.assemble(mat, proj_5, k);


    // EigenDenseMatrix<Complex> rhs;

    // EdgeExcitationAssembler<3> pw_mat;
    // pw_mat.build(rhs, pw, k, mesh);
    // rhs.print();


    return;

}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_projectors.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    Float tol = 1e-3;
    Float f = c0 / LAMBDA;
    Complex k = two * pi * f * std::sqrt(eps0 * (one - (Float)0.1 * J) * mu0);

    test_unit_cube_rwg_g(k);

    return 0;

}
