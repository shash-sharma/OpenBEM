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

#include <external/Eigen/Dense>

#include "test_rwg_operators.hpp"

#include "types.hpp"

#include "geometry/operations.hpp"
#include "quadrature/triangle/gauss.hpp"
#include "quadrature/line/gauss.hpp"

#include "kernels/hgf.hpp"

#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/src/quadrature.hpp"
#include "rwg/integrators/src/singularity.hpp"
#include "rwg/integrators/src/line.hpp"
#include "rwg/integrators/src/strategic.hpp"

#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/quadrature.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"
#include "rwg/operators/incidence.hpp"


using namespace bem;
using namespace bem::rwg;


const Float LAMBDA = 1;


void test_self_term(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 14;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_self_term"
        );

    return;
}


void test_shared_edge(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b * 3, -pi / 4);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 15;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_shared_edge"
        );

    return;
}


void test_shared_edge_partial(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a / 4, b * 3, -pi / 4);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 15;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_shared_edge_partial"
        );

    return;
}


void test_shared_vertex(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b * 2, pi / 4, {a, 0, 0});

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 10;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_shared_vertex"
        );

    return;
}


void test_parallel_above(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    EigMatMN<Float, 3, 3> offset_parallel;
    offset_parallel << a / 4, 0, b * 2,
        a / 4, 0, b * 2,
        a / 4, 0, b * 2;
    obs_tri.set_data(obs_tri.v() + offset_parallel.transpose());

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 10;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_parallel_above"
        );

    return;
}


void test_parallel_below(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    EigMatMN<Float, 3, 3> offset_parallel;
    offset_parallel << a / 4, 0, b * 2,
        a / 4, 0, b * 2,
        a / 4, 0, b * 2;
    obs_tri.set_data(obs_tri.v() - offset_parallel.transpose());

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 10;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_parallel_below"
        );

    return;
}


void test_parallel_exact(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    EigMatMN<Float, 3, 3> offset_parallel_exact;
    Float offset_factor = src_tri.shortest_edge_length() / 14.0;
    offset_parallel_exact.col(0) = src_tri.normal() * offset_factor;
    offset_parallel_exact.col(1) = src_tri.normal() * offset_factor;
    offset_parallel_exact.col(2) = src_tri.normal() * offset_factor;
    obs_tri.set_data(obs_tri.v() + offset_parallel_exact);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 12;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_parallel_exact"
        );

    return;
}


void test_coplanar(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    EigColVecN<Float, 3> edge_vec = src_tri.v(1) - src_tri.v(0);
    EigMatMN<Float, 3, 3> offset_coplanar;
    offset_coplanar.col(0) = edge_vec;
    offset_coplanar.col(1) = edge_vec;
    offset_coplanar.col(2) = edge_vec;
    obs_tri.set_data(obs_tri.v() + offset_coplanar);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 10;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_coplanar"
        );

    return;
}


void test_perpendicular(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3, EigColVecN<Float, 3>::Zero(3, 1), 0);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3, EigColVecN<Float, 3>::Zero(3, 1), 0);

    EigMatMN<Float, 3, 3> v;
    v.col(0) = src_tri.v(0);
    v.col(1) = src_tri.v(1);
    v.col(2) = (src_tri.v(0) + src_tri.v(1)) / 2;
    v.col(2)[2] = a / 1.5;
    obs_tri.set_data(v);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 10;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_perpendicular"
        );

    return;
}


void test_proj_on_vertex(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    EigMatMN<Float, 3, 3> v_proj_vertex;
    v_proj_vertex << -a / 2, -std::sqrt(3) / 4 * a, 0,
        a / 2, -std::sqrt(3) / 4 * a, 0,
        0, std::sqrt(3) / 2 * a, 0;
    v_proj_vertex.row(0) += src_tri.normal().transpose() * a / 5;
    v_proj_vertex.row(1) += src_tri.normal().transpose() * a / 5;
    v_proj_vertex.row(2) += src_tri.normal().transpose() * a / 5;
    obs_tri.set_data(v_proj_vertex.transpose());

    int src_quad_order = 4;
    int obs_quad_order = 1;
    int src_line_order = 10;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_proj_on_vertex"
        );

    return;
}


void test_proj_on_edge(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    EigMatMN<Float, 3, 3> v_proj_vertex;
    v_proj_vertex << -a / 2, -std::sqrt(3) / 4 * a, 0,
        a / 2, -std::sqrt(3) / 4 * a, 0,
        0, std::sqrt(3) / 2 * a, 0;
    v_proj_vertex.row(0) += src_tri.normal().transpose() * a / 5;
    v_proj_vertex.row(1) += src_tri.normal().transpose() * a / 5;
    v_proj_vertex.row(2) += src_tri.normal().transpose() * a / 5;

    EigMatMN<Float, 3, 3> v_proj_edge = v_proj_vertex;
    v_proj_edge.row(0) += (src_tri.v(1) - src_tri.v(0)).transpose() * a / 5;
    v_proj_edge.row(1) += (src_tri.v(1) - src_tri.v(0)).transpose() * a / 5;
    v_proj_edge.row(2) += (src_tri.v(1) - src_tri.v(0)).transpose() * a / 5;
    obs_tri.set_data(v_proj_edge.transpose());

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 10;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_proj_on_edge"
        );

    return;
}


void test_nudge(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a * 0.99, b * 0.99, pi / 3);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 12;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_nudge"
        );

    return;
}


void test_partial_overlap(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    EigMatMN<Float, 3, 3> offset_parallel_exact;
    Float offset_factor = src_tri.shortest_edge_length() / 20.0;
    offset_parallel_exact.col(0) = src_tri.normal() * offset_factor;
    offset_parallel_exact.col(1) = src_tri.normal() * offset_factor;
    offset_parallel_exact.col(2) = src_tri.normal() * offset_factor;

    EigColVecN<Float, 3> edge_vec = src_tri.v(1) - src_tri.v(0);
    EigMatMN<Float, 3, 3> offset_overlap;
    offset_overlap.col(0) = edge_vec * 0.05;
    offset_overlap.col(1) = edge_vec * 0.05;
    offset_overlap.col(2) = edge_vec * 0.05;
    obs_tri.set_data(obs_tri.v() + offset_overlap + offset_parallel_exact * 0);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 14;

    SingularitySubtractedTaylorHGF sthgf;

    compare_quad_and_line(
        k, sthgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, true, "test_partial_overlap"
        );

    return;
}


void test_reg(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    EigMatMN<Float, 3, 3> offset;
    offset.col(0) = EigColVecN<Float, 3> ({ LAMBDA * (Float)0.8, LAMBDA * (Float)2.4, LAMBDA * (Float)5.1 });
    offset.col(1) = EigColVecN<Float, 3> ({ LAMBDA * (Float)0.7, LAMBDA * (Float)2.3, LAMBDA * (Float)5.0 });
    offset.col(2) = EigColVecN<Float, 3> ({ LAMBDA * (Float)1.1, LAMBDA * (Float)2.2, LAMBDA * (Float)4.8 });
    obs_tri.set_data(obs_tri.v() + offset);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 10;

    HGF hgf;

    compare_quad_and_line(
        k, hgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, false, "test_reg"
        );

    return;
}


void test_far(
    Complex k, Float tol = 1e-3, bool print_anyway = false
    )
{
    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Triangle<3> src_tri = make_triangle(a, b, pi / 3);
    Triangle<3> obs_tri = make_triangle(a, b, pi / 3);

    EigMatMN<Float, 3, 3> offset;
    offset.col(0) = EigColVecN<Float, 3> ({ LAMBDA * (Float)110, LAMBDA * (Float)24, LAMBDA * (Float)58 });
    offset.col(1) = EigColVecN<Float, 3> ({ LAMBDA * (Float)110.1, LAMBDA * (Float)23.8, LAMBDA * (Float)58 });
    offset.col(2) = EigColVecN<Float, 3> ({ LAMBDA * (Float)109.9, LAMBDA * (Float)24.1, LAMBDA * (Float)58.3 });
    obs_tri.set_data(obs_tri.v() + offset);

    int src_quad_order = 4;
    int obs_quad_order = 4;
    int src_line_order = 10;

    HGF hgf;

    compare_quad_and_line(
        k, hgf, src_tri, obs_tri,
        src_quad_order, obs_quad_order, src_line_order,
        tol, print_anyway, false, "test_far"
        );

    return;
}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_rwg_operators.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    Float tol = 1e-3;
    Float f = c0 / LAMBDA;
    Complex k = two * pi * f * std::sqrt(eps0 * (one - (Float)0.1 * J) * mu0);

    test_self_term(k, tol);
    test_shared_edge(k, tol);
    test_shared_edge_partial(k, tol);
    test_shared_vertex(k, tol);
    test_parallel_above(k, tol);
    test_parallel_below(k, tol);
    test_parallel_exact(k, tol);
    test_coplanar(k, tol);
    test_perpendicular(k, 4e-2);
    test_proj_on_vertex(k, tol);
    test_proj_on_edge(k, tol);
    test_nudge(k, tol);
    test_partial_overlap(k, 5e-3);
    test_reg(k, tol);
    test_far(k, 1.7e-3);
    test_far(Complex(k.real(), 0), tol);

    return 0;

}
