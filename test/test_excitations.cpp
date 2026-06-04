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
#include <vector>

#include "types.hpp"

#include "rwg/assemblers/excitation_matrix.hpp"
#include "rwg/excitations/plane_wave.hpp"
#include "rwg/excitations/inf_gap.hpp"

#include "matrix/eigen_matrix.hpp"

#include "geometry/mesh/triangle_mesh.hpp"


using namespace bem;
using namespace bem::rwg;


const Float LAMBDA = 1;


void test_unit_cube_plane_wave(Complex k)
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

    TriangleMesh<3> mesh (vertices, tri_vertices);

    EigColVecN<Float, 3> dir, pol, pos;
    EigRowVecN<Complex, 1> amp;

    dir << 0, 0, 1;
    pol << 1, 0, 0;
    pos << 0, 0, -5;
    amp << 1;

    RwgPlaneWave pw (dir, pol, pos, amp);
    // std::cout << pw.num_excitations() << std::endl;

    Triangle<3> tri (vertices(Eigen::placeholders::all, tri_vertices.col(0)));

    EigMatNX<Complex, 3> pw_vals = pw.compute(k, tri);
    // std::cout << pw_vals << std::endl;

    EigenMatrix<Complex> rhs;

    ExcitationAssembler<Rwg> pw_mat (mesh);
    pw_mat.assemble(rhs, pw, k);
    // rhs.print();

    return;

}


void test_unit_cube_inf_gap()
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

    EigRowVecN<Complex, 1> amp;
    amp << 1;

    EigMatMN<Float, 2, 3> segments;
    segments << 1, 1, 1,
                1, 0, 1;

    std::vector<EigMatNX<Float, 3>> segvec { segments.transpose() };
    InfinitesimalGap inf_gap (segvec, amp);

    Triangle<3> tri0 (vertices(Eigen::placeholders::all, tri_vertices.col(0)));
    Triangle<3> tri2 (vertices(Eigen::placeholders::all, tri_vertices.col(2)));
    Triangle<3> tri7 (vertices(Eigen::placeholders::all, tri_vertices.col(7)));

    EigMatNX<Complex, 3> tri0_vals = inf_gap.compute(0, tri0);
    EigMatNX<Complex, 3> tri2_vals = inf_gap.compute(0, tri2);
    EigMatNX<Complex, 3> tri7_vals = inf_gap.compute(0, tri7);

    // std::cout << tri0_vals << std::endl;
    // std::cout << tri2_vals << std::endl;
    // std::cout << tri7_vals << std::endl;

    EigenMatrix<Complex> rhs;

    ExcitationAssembler<Rwg> pw_mat (mesh);
    pw_mat.assemble(rhs, inf_gap, 0);

    // rhs.print();

    return;

}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_excitations.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    Float f = c0 / LAMBDA;
    Complex k = two * pi * f * std::sqrt(eps0 * (one - (Float)0.1 * J) * mu0);

    test_unit_cube_plane_wave(k);
    test_unit_cube_inf_gap();

    return 0;

}
