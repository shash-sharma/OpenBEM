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
#include <ctime>

#include "types.hpp"
#include "geometry/operations.hpp"

#include "kernels/hgf.hpp"
#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/src/quadrature.hpp"
#include "rwg/integrators/src/singularity.hpp"
#include "rwg/integrators/src/line.hpp"
#include "rwg/integrators/src/strategic.hpp"

#include "quadrature/triangle/gauss.hpp"
#include "quadrature/triangle/iterative_gauss.hpp"
#include "quadrature/triangle/adaptive.hpp"

#include "quadrature/line/gauss.hpp"
#include "quadrature/line/iterative_gauss.hpp"
#include "quadrature/line/trapz.hpp"
#include "quadrature/line/iterative_trapz.hpp"

using namespace bem;
using namespace bem::rwg;


const Float LAMBDA = 1;


void linspace(Float start, Float stop, Index N, std::vector<Float>& output)
{

    output.resize(N);

    if (N == 1)
    {
        output[0] = start;
        return;
    }

    Float dx = (stop - start)/((Float)(N - 1));

    for (Index ii = 0; ii < N; ++ii)
        output[ii] = start + ((Float)ii) * dx;

    return;

}


void logspace(Float start, Float stop, Index N, std::vector<Float>& output)
{

    linspace(start, stop, N, output);

    for (Index ii = 0; ii < N; ++ii)
        output[ii] = std::pow(10.0, output[ii]);

    return;

}

Triangle<3> make_triangle(Float a, Float b, Float theta, EigColVecN<Float, 3> offset = EigColVecN<Float, 3>::Zero(3, 1))
{
    EigMatMN<Float, 3, 3> v;
    v.col(0) = EigColVecN<Float, 3> (0.0, 0.0, 0.0) + offset;
    v.col(1) = EigColVecN<Float, 3> (a, 0.0, -a / 3) + offset;
    v.col(2) = EigColVecN<Float, 3> (b * std::cos(theta), b * std::sin(theta), a / 4) + offset;
    return Triangle<3> (v);
}

EigMatNX<Float, 3> make_obs_pts(EigColVecN<Float, 3> p1, EigColVecN<Float, 3> p2, uint16_t num_pts, bool logspaced = false)
{
    std::vector<Float> pts;
    if (logspaced)
        logspace((Float)(-15.0), std::log10((p2 - p1).norm()), num_pts, pts);
    else
        linspace(zero, one, num_pts, pts);

    EigMatNX<Float, 3> obs_pts = EigMatNX<Float, 3>::Zero(3, num_pts);
    for (uint16_t ii = 0; ii < num_pts; ++ii)
        obs_pts.col(ii) = p1.array() + (p2 - p1).array() * pts[ii];

    return obs_pts;
}

std::vector<Triangle<3>> make_varying_triangles(Float start_angle, Float stop_angle, uint32_t num_angles = 1)
{
    // make a series of triangles with one angle ranging from very acute to very obtuse
    std::vector<Float> thetas;
    // linspace(2.0 * pi / 180.0, 178.0 * pi / 180.0, num_angles, thetas);
    linspace(start_angle, stop_angle, num_angles, thetas);

    std::vector<Triangle<3>> src_tris;

    for (const auto &theta : thetas)
    {
        Float a = LAMBDA / 10.0;
        Float b = 0.6 * a;

        EigMatMN<Float, 3, 3> v;
        v.col(0) = EigColVecN<Float, 3>(0.0, 0.0, 0.0);
        v.col(1) = EigColVecN<Float, 3>(a, 0.0, -a / 3);
        v.col(2) = EigColVecN<Float, 3>(b * std::cos(theta), b * std::sin(theta), a / 4);
        src_tris.push_back(Triangle<3> (v));

        v.col(1) = EigColVecN<Float, 3>(a, 0.0, -a / 3);
        v.col(2) = EigColVecN<Float, 3>(b * std::cos(theta), b * std::sin(theta), 0);
        v.col(1) = EigColVecN<Float, 3>(a, 0.0, 0);
        src_tris.push_back(Triangle<3> (v));

        v.col(2) = EigColVecN<Float, 3>(b * std::cos(theta), b * std::sin(theta), a / 4);
        v.col(1) = EigColVecN<Float, 3>(a, 0.0, 0);
        v.col(2) = EigColVecN<Float, 3>(b * std::cos(theta), b * std::sin(theta), 0);
        src_tris.push_back(Triangle<3> (v));
    }

    return src_tris;
}

EigMatNX<Float, 3> make_obs_pts_corner_cases(Float start, Float stop, uint32_t num_obs = 1)
{
    // make observation points at varying electrical distances from the source plane
    std::vector<Float> rs;
    logspace(start, stop, num_obs, rs);
    // rs.insert(rs.begin(), 0);

    EigMatNX<Float, 3> rs_obs = EigMatNX<Float, 3>::Zero(3, rs.size() * 7);
    for (int ii = 0; ii < rs.size(); ++ii)
    {
        Float r = rs[ii];
        rs_obs.col(ii * 7 + 0) = EigColVecN<Float, 3> ({ 0.02, 0.03, r * LAMBDA });
        rs_obs.col(ii * 7 + 1) = EigColVecN<Float, 3> ({ 0, 0, r * LAMBDA });
        rs_obs.col(ii * 7 + 2) = EigColVecN<Float, 3> ({ 0.02, 0, r * LAMBDA });
        rs_obs.col(ii * 7 + 3) = EigColVecN<Float, 3> ({ 0.02, 0.03*0.00001, r * LAMBDA });
        rs_obs.col(ii * 7 + 4) = EigColVecN<Float, 3> ({ 0.02, 0.03*0.0000001, r * LAMBDA });
        rs_obs.col(ii * 7 + 5) = EigColVecN<Float, 3> ({ 0.02, -0.03*0.00001, r * LAMBDA });
        rs_obs.col(ii * 7 + 6) = EigColVecN<Float, 3> ({ 0.02, -0.03*0.0000001, r * LAMBDA });
    }

    return rs_obs;
}

bool compare_src_result(
    SrcResult &ref,
    SrcResult &test,
    Float tol = 1e-3,
    bool print_anyway = false,
    std::string fail_message = ""
    )
{
    SrcResult src_error;

    src_error.g = (ref.g - test.g).array().abs() / ref.g.array().abs();
    src_error.rs_g = (ref.rs_g - test.rs_g).array().abs() / ref.rs_g.array().abs();
    src_error.grad_g = (ref.grad_g - test.grad_g).array().abs() / ref.grad_g.array().abs();

    bool pass_g = true;
    if ((src_error.g.array().abs() > tol).any())
        pass_g = false;
    if (!pass_g || print_anyway)
    {
        for (uint16_t rr = 0; rr < src_error.g.cols(); ++rr)
            if (std::abs(src_error.g[rr]) > tol || print_anyway)
                std::cout << "Fail: g, r_obs idx: " << (int)rr << ", " << ref.g[rr] << ", " << test.g[rr] << ", " << src_error.g[rr] << "; " << std::endl;
    }

    bool pass_rs_g = true;
    if ((src_error.rs_g.array().abs() > tol).any())
        pass_rs_g = false;
    if (!pass_rs_g || print_anyway)
    {
        for (uint16_t rr = 0; rr < src_error.rs_g.cols(); ++rr)
            for (uint16_t jj = 0; jj < src_error.rs_g.rows(); ++jj)
                if (std::abs(src_error.rs_g(jj, rr)) > tol || print_anyway)
                    std::cout << "Fail: rs_g, r_obs idx: " << (int)rr << ", " << "component: " << (int)jj << ", " << ref.rs_g(jj, rr) << ", " << test.rs_g(jj, rr) << ", " << src_error.rs_g(jj, rr) << "; " << std::endl;
    }

    bool pass_grad_g = true;
    if ((src_error.grad_g.array().abs() > tol).any())
        pass_grad_g = false;
    if (!pass_grad_g || print_anyway)
    {
        for (uint16_t rr = 0; rr < src_error.grad_g.cols(); ++rr)
            for (uint16_t jj = 0; jj < src_error.grad_g.rows(); ++jj)
                if (std::abs(src_error.grad_g(jj, rr)) > tol || print_anyway)
                    std::cout << "Fail: grad_g, r_obs idx: " << (int)rr << ", " << "component: " << (int)jj << ", " << ref.grad_g(jj, rr) << ", " << test.grad_g(jj, rr) << ", " << src_error.grad_g(jj, rr) << "; " << std::endl;
    }

    bool pass = pass_g && pass_rs_g && pass_grad_g;
    if (!pass && fail_message != "")
        std::cout << "Fail: " << fail_message << std::endl;

    return pass;
}

int test_src_quadrature(Complex k, Float tol = 1e-3, bool print_anyway = false)
{

    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Float theta = pi / 3;
    Triangle<3> src_tri = make_triangle(a, b, theta);

    EigColVecN<Float, 3> p1, p2;
    p1 << -0.3 * a, 0, LAMBDA / 3;
    p2 << 2.3 * a, 4 * b, LAMBDA * 100;
    EigMatNX<Float, 3> r_obs = make_obs_pts(p1, p2, 50, true);

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigColVecN<Float, 3> local_origin = src_tri.local_origin();
    Triangle<2> src_tri_local (
        GeometryOps<3>::transform_coordinate_system(
            src_tri.v(), local_origin, local_uvw
            ).topRows(2)
        );
    EigMatNX<Float, 3> r_obs_local = GeometryOps<3>::transform_coordinate_system(
        r_obs, local_origin, local_uvw
        );

    GaussTriangleQuadrature<2> gauss_tri_quad_ref (TRI_MAX_ORDER);
    GaussTriangleQuadrature<2> gauss_tri_quad (4);
    IterativeGaussTriangleQuadrature<2> iter_gauss_tri_quad;
    AdaptiveTriangleQuadrature<2> adapt_tri_quad;

    HGF hgf;

    SrcQuadrature src_gauss_tri_quad_ref (gauss_tri_quad_ref, hgf);
    SrcQuadrature src_gauss_tri_quad (gauss_tri_quad, hgf);
    SrcQuadrature src_iter_gauss_tri_quad (iter_gauss_tri_quad, hgf);
    SrcQuadrature src_adapt_tri_quad (adapt_tri_quad, hgf);

    SrcResult result_gauss_tri_quad_ref = src_gauss_tri_quad_ref.integrate(
        k, src_tri_local, r_obs_local
        );
    SrcResult result_gauss_tri_quad = src_gauss_tri_quad.integrate(
        k, src_tri_local, r_obs_local
        );
    SrcResult result_iter_gauss_tri_quad = src_iter_gauss_tri_quad.integrate(
        k, src_tri_local, r_obs_local
        );
    SrcResult result_adapt_tri_quad = src_adapt_tri_quad.integrate(
        k, src_tri_local, r_obs_local
        );

    bool pass_gauss_tri_quad = true, pass_iter_gauss_tri_quad = true, pass_adapt_tri_quad = true;
    pass_gauss_tri_quad = compare_src_result(
        result_gauss_tri_quad_ref, result_gauss_tri_quad, tol, print_anyway, "test_src_quadrature");
    pass_iter_gauss_tri_quad = compare_src_result(
        result_gauss_tri_quad_ref, result_iter_gauss_tri_quad, tol, print_anyway, "test_src_quadrature");
    pass_adapt_tri_quad = compare_src_result(
        result_gauss_tri_quad_ref, result_adapt_tri_quad, tol, print_anyway, "test_src_quadrature");

    if (!pass_gauss_tri_quad)
        std::cout << "Fail: GaussTriangleQuadrature" << std::endl;
    if (!pass_iter_gauss_tri_quad)
        std::cout << "Fail: IterativeGaussTriangleQuadrature" << std::endl;
    if (!pass_adapt_tri_quad)
        std::cout << "Fail: AdaptiveTriangleQuadrature" << std::endl;

    return !(pass_gauss_tri_quad && pass_iter_gauss_tri_quad && pass_adapt_tri_quad);

}

int test_src_singularity(Complex k, Float tol = 1e-3, bool print_anyway = false)
{

    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Float theta = pi / 3;
    Triangle<3> src_tri = make_triangle(a, b, theta);

    EigColVecN<Float, 3> p1, p2;
    p1 << -0.3 * a, 0, LAMBDA / 15;
    p2 << 2.3 * a, 4 * b, LAMBDA;
    EigMatNX<Float, 3> r_obs = make_obs_pts(p1, p2, 20);

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigColVecN<Float, 3> local_origin = src_tri.local_origin();
    Triangle<2> src_tri_local (
        GeometryOps<3>::transform_coordinate_system(
            src_tri.v(), local_origin, local_uvw
            ).topRows(2)
        );
    EigMatNX<Float, 3> r_obs_local = GeometryOps<3>::transform_coordinate_system(
        r_obs, local_origin, local_uvw
        );

    GaussTriangleQuadrature<2> quad (4);

    HGF hgf;
    SingularitySubtractedHGF shgf;
    SingularitySubtractedTaylorHGF sthgf;

    SrcQuadrature src_hgf (quad, hgf);
    SrcSingularity src_shgf (quad, shgf);
    SrcSingularity src_sthgf (quad, sthgf);

    SrcResult result_hgf = src_hgf.integrate(k, src_tri_local, r_obs_local);
    SrcResult result_shgf = src_shgf.integrate(k, src_tri_local, r_obs_local);
    SrcResult result_sthgf = src_sthgf.integrate(k, src_tri_local, r_obs_local);

    bool pass_shgf = compare_src_result(result_hgf, result_shgf, tol, print_anyway, "test_src_singularity");
    bool pass_sthgf = compare_src_result(result_hgf, result_sthgf, tol, print_anyway, "test_src_singularity");

    if (!pass_shgf)
        std::cout << "Fail: SingularitySubtractedHGF" << std::endl;
    if (!pass_sthgf)
        std::cout << "Fail: SingularitySubtractedTaylorHGF" << std::endl;

    return !(pass_shgf && pass_sthgf);

}

int test_src_line_integration(Complex k, Float tol = 1e-3, bool print_anyway = false)
{

    Float a = LAMBDA / 10;
    Float b = 0.6 * a;
    Float theta = pi / 3;
    Triangle<3> src_tri = make_triangle(a, b, theta);

    EigColVecN<Float, 3> p1, p2;
    p1 << -0.3 * a, 0, LAMBDA / 15;
    // p2 << 2.3 * a, 4 * b, LAMBDA * 10;
    p2 << 2.3 * a, 4 * b, LAMBDA / 4;
    EigMatNX<Float, 3> r_obs = make_obs_pts(p1, p2, 50, false);

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigColVecN<Float, 3> local_origin = src_tri.local_origin();
    Triangle<2> src_tri_local (
        GeometryOps<3>::transform_coordinate_system(
            src_tri.v(), local_origin, local_uvw
            ).topRows(2)
        );
    EigMatNX<Float, 3> r_obs_local = GeometryOps<3>::transform_coordinate_system(
        r_obs, local_origin, local_uvw
        );

    GaussTriangleQuadrature<2> gauss_tri_quad_ref (TRI_MAX_ORDER);
    GaussLineQuadrature<1> gauss_line_quad (4);
    IterativeGaussLineQuadrature<1> iter_gauss_line_quad;
    iter_gauss_line_quad.set_tol(2e-4);
    TrapzLineQuadrature<1> trapz_line_quad (60);
    IterativeTrapzLineQuadrature<1> iter_trapz_line_quad;
    iter_trapz_line_quad.set_tol(1e-4);

    HGF hgf;

    SrcQuadrature src_gauss_tri_quad_ref (gauss_tri_quad_ref, hgf);
    SrcLineIntegrator src_gauss_line_quad (gauss_line_quad);
    // SrcLineIntegrator src_iter_gauss_line_quad (iter_gauss_line_quad);
    SrcLineIntegrator src_trapz_line_quad (trapz_line_quad);
    // SrcLineIntegrator src_iter_trapz_line_quad (iter_trapz_line_quad);

    SrcResult result_gauss_tri_quad_ref = src_gauss_tri_quad_ref.integrate(k, src_tri_local, r_obs_local);
    SrcResult result_gauss_line_quad = src_gauss_line_quad.integrate(k, src_tri_local, r_obs_local);
    // SrcResult result_iter_gauss_line_quad = src_iter_gauss_line_quad.integrate(k, src_tri_local, r_obs_local);
    SrcResult result_trapz_line_quad = src_trapz_line_quad.integrate(k, src_tri_local, r_obs_local);
    // SrcResult result_iter_trapz_line_quad = src_iter_trapz_line_quad.integrate(k, src_tri_local, r_obs_local);

    bool pass_gauss_line_quad = compare_src_result(
        result_gauss_tri_quad_ref, result_gauss_line_quad, tol, print_anyway, "test_src_line_integration");
    // bool pass_iter_gauss_line_quad = compare_src_result(
    //     result_gauss_tri_quad_ref, result_iter_gauss_line_quad, tol, print_anyway, "test_src_line_integration");
    bool pass_trapz_line_quad = compare_src_result(
        result_gauss_tri_quad_ref, result_trapz_line_quad, 5e-3, print_anyway, "test_src_line_integration");
    // bool pass_iter_trapz_line_quad = compare_src_result(
    //     result_gauss_tri_quad_ref, result_iter_trapz_line_quad, tol, print_anyway, "test_src_line_integration");

    if (!pass_gauss_line_quad)
        std::cout << "Fail: GaussLineQuadrature" << std::endl;
    // if (!pass_iter_gauss_line_quad)
    //     std::cout << "Fail: IterativeGaussLineQuadrature" << std::endl;
    if (!pass_trapz_line_quad)
        std::cout << "Fail: TrapzLineQuadrature" << std::endl;
    // if (!pass_iter_trapz_line_quad)
    //     std::cout << "Fail: IterativeTrapzLineQuadrature" << std::endl;

    return !(
        pass_gauss_line_quad
        // && pass_iter_gauss_line_quad
        && pass_trapz_line_quad
        // && pass_iter_trapz_line_quad
        );

}

int test_corner_cases(Complex k, bool print_anyway = false)
{

    std::vector<Triangle<3>> src_tris = make_varying_triangles(2.0 * pi / 180.0, 178.0 * pi / 180.0, 10);
    // std::vector<Triangle<3>> src_tris = make_varying_triangles(10.0 * pi / 180.0, 178.0 * pi / 180.0, 1);

    EigMatNX<Float, 3> r_obs = make_obs_pts_corner_cases(-1.0, 0.0, 10);
    // EigMatNX<Float, 3> r_obs = make_obs_pts_corner_cases(0.0, 1.0, 1);

    GaussTriangleQuadrature<2> gauss_tri_quad_ref (TRI_MAX_ORDER);
    GaussTriangleQuadrature<2> gauss_tri_quad (7);
    IterativeGaussTriangleQuadrature<2> iter_gauss_tri_quad;
    AdaptiveTriangleQuadrature<2> adapt_tri_quad;
    GaussLineQuadrature<1> gauss_line_quad (10);
    IterativeGaussLineQuadrature<1> iter_gauss_line_quad;
    iter_gauss_line_quad.set_starting_order(4);
    TrapzLineQuadrature<1> trapz_line_quad (40);
    IterativeTrapzLineQuadrature<1> iter_trapz_line_quad;
    iter_trapz_line_quad.set_starting_num_segments(4);

    HGF hgf;
    SingularitySubtractedHGF shgf;
    SingularitySubtractedTaylorHGF sthgf;

    SrcQuadrature src_ref (gauss_tri_quad_ref, hgf);

    SrcQuadrature src_hgf_gauss_tri_quad (gauss_tri_quad, hgf);
    SrcQuadrature src_hgf_iter_gauss_tri_quad (iter_gauss_tri_quad, hgf);
    SrcQuadrature src_hgf_adapt_tri_quad (adapt_tri_quad, hgf);

    SrcSingularity src_shgf_gauss_tri_quad (gauss_tri_quad, shgf);
    SrcSingularity src_shgf_iter_gauss_tri_quad (iter_gauss_tri_quad, shgf);
    SrcSingularity src_shgf_adapt_tri_quad (adapt_tri_quad, shgf);

    SrcSingularity src_sthgf_gauss_tri_quad (gauss_tri_quad, sthgf);
    SrcSingularity src_sthgf_iter_gauss_tri_quad (iter_gauss_tri_quad, sthgf);
    SrcSingularity src_sthgf_adapt_tri_quad (adapt_tri_quad, sthgf);

    SrcLineIntegrator src_gauss_line_quad (gauss_line_quad);
    // SrcLineIntegrator src_iter_gauss_line_quad (iter_gauss_line_quad);
    SrcLineIntegrator src_trapz_line_quad (trapz_line_quad);
    // SrcLineIntegrator src_iter_trapz_line_quad (iter_trapz_line_quad);

    // --- Execute ---

    bool all_pass = true;

    // std::clock_t t_start = clock();
    for (auto src_tri : src_tris)
    {
        EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
        EigColVecN<Float, 3> local_origin = src_tri.local_origin();
        Triangle<2> src_tri_local (
            GeometryOps<3>::transform_coordinate_system(
                src_tri.v(), local_origin, local_uvw
                ).topRows(2)
            );
        EigMatNX<Float, 3> r_obs_local = GeometryOps<3>::transform_coordinate_system(
            r_obs, local_origin, local_uvw
            );

        SrcResult result_ref = src_ref.integrate(k, src_tri_local, r_obs_local);

        SrcResult result_hgf_gauss_tri_quad = src_hgf_gauss_tri_quad.integrate(k, src_tri_local, r_obs_local);
        SrcResult result_hgf_iter_gauss_tri_quad = src_hgf_iter_gauss_tri_quad.integrate(k, src_tri_local, r_obs_local);
        SrcResult result_hgf_adapt_tri_quad = src_hgf_adapt_tri_quad.integrate(k, src_tri_local, r_obs_local);

        SrcResult result_shgf_gauss_tri_quad = src_shgf_gauss_tri_quad.integrate(k, src_tri_local, r_obs_local);
        SrcResult result_shgf_iter_gauss_tri_quad = src_shgf_iter_gauss_tri_quad.integrate(k, src_tri_local, r_obs_local);
        SrcResult result_shgf_adapt_tri_quad = src_shgf_adapt_tri_quad.integrate(k, src_tri_local, r_obs_local);

        SrcResult result_sthgf_gauss_tri_quad = src_sthgf_gauss_tri_quad.integrate(k, src_tri_local, r_obs_local);
        SrcResult result_sthgf_iter_gauss_tri_quad = src_sthgf_iter_gauss_tri_quad.integrate(k, src_tri_local, r_obs_local);
        SrcResult result_sthgf_adapt_tri_quad = src_sthgf_adapt_tri_quad.integrate(k, src_tri_local, r_obs_local);

        SrcResult result_gauss_line_quad = src_gauss_line_quad.integrate(k, src_tri_local, r_obs_local);
        // SrcResult result_iter_gauss_line_quad = src_iter_gauss_line_quad.integrate(k, src_tri_local, r_obs_local);
        // SrcResult result_trapz_line_quad = src_trapz_line_quad.integrate(k, src_tri_local, r_obs_local);
        // SrcResult result_iter_trapz_line_quad = src_iter_trapz_line_quad.integrate(k, src_tri_local, r_obs_local);

        all_pass = compare_src_result(
            result_ref, result_hgf_gauss_tri_quad, 2.4e-3, print_anyway, "test_corner_cases, Gauss quadrature, HGF") && all_pass;
        all_pass = compare_src_result(
            result_ref, result_hgf_iter_gauss_tri_quad, 5.5e-2, print_anyway, "test_corner_cases, Iterative Gauss quadrature, HGF") && all_pass;
        all_pass = compare_src_result(
            result_ref, result_hgf_adapt_tri_quad, 3.8e-2, print_anyway, "test_corner_cases, Adaptive quadrature, HGF") && all_pass;

        all_pass = compare_src_result(
            result_ref, result_shgf_gauss_tri_quad, 1e-3, print_anyway, "test_corner_cases, Gauss quadrature, sHGF") && all_pass;
        all_pass = compare_src_result(
            result_ref, result_shgf_iter_gauss_tri_quad, 1e-3, print_anyway, "test_corner_cases, Iterative Gauss quadrature, sHGF") && all_pass;
        all_pass = compare_src_result(
            result_ref, result_shgf_adapt_tri_quad, 3e-2, print_anyway, "test_corner_cases, Adaptive quadrature, sHGF") && all_pass;

        all_pass = compare_src_result(
            result_ref, result_sthgf_gauss_tri_quad, 1e-3, print_anyway, "test_corner_cases, Gauss quadrature, stHGF") && all_pass;
        all_pass = compare_src_result(
            result_ref, result_sthgf_iter_gauss_tri_quad, 1e-3, print_anyway, "test_corner_cases, Iterative Gauss quadrature, stHGF") && all_pass;
        all_pass = compare_src_result(
            result_ref, result_sthgf_adapt_tri_quad, 3e-2, print_anyway, "test_corner_cases, Adaptive quadrature, stHGF") && all_pass;

        all_pass = compare_src_result(
            result_ref, result_gauss_line_quad, 1e-3, print_anyway, "test_corner_cases, Gauss line integration") && all_pass;
        // all_pass = compare_src_result(
        //     result_ref, result_iter_gauss_line_quad, 1e-3, print_anyway, "test_corner_cases, Iterative Gauss line integration") && all_pass;
        // all_pass = compare_src_result(
        //     result_ref, result_trapz_line_quad, 4.1e-2, print_anyway, "test_corner_cases, Trapz line integration") && all_pass;
        // all_pass = compare_src_result(
        //     result_ref, result_iter_trapz_line_quad, 1e-3, print_anyway, "test_corner_cases, Iterative trapz line integration") && all_pass;
    }
    // std::clock_t t_stop = clock();
    // double dt = (t_stop - t_start)/(double)CLOCKS_PER_SEC;
    // std::cout << "[CPU TIME] (Standard reference, release mode: ): " << std::fixed << std::scientific << dt << " s." << std::endl;

    return !all_pass;

}


int test_src_reg(Complex k, Float tol = 1e-3, bool print_anyway = false)
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

    int src_line_order = 10;

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigColVecN<Float, 3> local_origin = src_tri.local_origin();
    Triangle<2> src_tri_local (
        GeometryOps<3>::transform_coordinate_system(
            src_tri.v(), local_origin, local_uvw
            ).topRows(2)
        );
    Triangle<3> obs_tri_local (
        GeometryOps<3>::transform_coordinate_system(
            obs_tri.v(), local_origin, local_uvw
            )
        );

    EigMatNX<Float, 3> r_obs_local = obs_tri_local.centroid();

    GaussTriangleQuadrature<2> gauss_tri_quad_ref (TRI_MAX_ORDER);
    GaussLineQuadrature<1> gauss_line_quad (src_line_order);

    HGF hgf;

    SrcQuadrature src_gauss_tri_quad_ref (gauss_tri_quad_ref, hgf);
    SrcLineIntegrator src_gauss_line_quad (gauss_line_quad);

    SrcResult result_gauss_tri_quad_ref = src_gauss_tri_quad_ref.integrate(k, src_tri_local, r_obs_local);
    SrcResult result_gauss_line_quad = src_gauss_line_quad.integrate(k, src_tri_local, r_obs_local);

    bool pass_gauss_line_quad = compare_src_result(
        result_gauss_tri_quad_ref, result_gauss_line_quad, tol, print_anyway, "test_src_reg");

    if (!pass_gauss_line_quad)
        std::cout << "Fail: GaussLineQuadrature" << std::endl;

    return !(pass_gauss_line_quad);

}


int test_src_far(Complex k, Float tol = 1e-3, bool print_anyway = false)
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

    int src_line_order = 10;

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigColVecN<Float, 3> local_origin = src_tri.local_origin();
    Triangle<2> src_tri_local (
        GeometryOps<3>::transform_coordinate_system(
            src_tri.v(), local_origin, local_uvw
            ).topRows(2)
        );
    Triangle<3> obs_tri_local (
        GeometryOps<3>::transform_coordinate_system(
            obs_tri.v(), local_origin, local_uvw
            )
        );

    EigMatNX<Float, 3> r_obs_local = obs_tri_local.centroid();

    GaussTriangleQuadrature<2> gauss_tri_quad_ref (TRI_MAX_ORDER);
    GaussLineQuadrature<1> gauss_line_quad (src_line_order);

    HGF hgf;

    SrcQuadrature src_gauss_tri_quad_ref (gauss_tri_quad_ref, hgf);
    SrcLineIntegrator src_gauss_line_quad (gauss_line_quad);

    SrcResult result_gauss_tri_quad_ref = src_gauss_tri_quad_ref.integrate(k, src_tri_local, r_obs_local);
    SrcResult result_gauss_line_quad = src_gauss_line_quad.integrate(k, src_tri_local, r_obs_local);

    bool pass_gauss_line_quad = compare_src_result(
        result_gauss_tri_quad_ref, result_gauss_line_quad, tol, print_anyway, "test_src_far");

    if (!pass_gauss_line_quad)
        std::cout << "Fail: GaussLineQuadrature" << std::endl;

    return !(pass_gauss_line_quad);

}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_src_int.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    Float tol = 1e-3;
    Float f = c0 / LAMBDA;
    Complex k = two * pi * f * std::sqrt(eps0 * (one - (Float)0.1 * J) * mu0);

    test_src_quadrature(k, tol, false);
    test_src_singularity(k, tol, false);
    test_src_line_integration(k, tol, false);

    test_src_quadrature(0, tol, false);
    test_src_singularity(0, tol, false);
    test_src_line_integration(0, 4e-3, false);

    test_src_reg(k, tol, false);
    test_src_far(k, 1.6e-3, false);

    test_src_reg(0, tol, false);
    test_src_far(0, 1.2e-3, false);

    test_corner_cases(k, false);
    test_corner_cases(0, false);


    SrcIntegrationSettings settings;
    SrcStrategic src_int_strategic (settings);

    // test default declarations
    SrcLineIntegrator src_line_default;
    SrcQuadrature src_quad_default;
    SrcSingularity src_sing_default;
    SrcStrategic src_str_default;


    // TODO: improvements for single precision

    return 0;

}
