// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* 2D quadrature over the source triangle with singularity treatment for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_SINGULARITY_I
#define BEM_RWG_OPINT_SRC_SINGULARITY_I

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/edge.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

template <typename TriangleQuadratureType, typename ScalarKernelType>
SrcResult SrcSingularity<TriangleQuadratureType, ScalarKernelType>::integrate(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs,
    const bool g_terms,
    const bool grad_g_terms
    )
{
    SrcResult result_singular = integrate_singular(k, src_tri, r_obs, g_terms, grad_g_terms);
    SrcResult result_nonsingular = src_quad_.integrate(k, src_tri, r_obs, g_terms, grad_g_terms);

    SrcResult result;
    result.g = result_singular.g + result_nonsingular.g;
    result.rs_g = result_singular.rs_g + result_nonsingular.rs_g;
    result.grad_g = result_singular.grad_g + result_nonsingular.grad_g;

    return result;
};


template <typename TriangleQuadratureType, typename ScalarKernelType>
SrcResult SrcSingularity<TriangleQuadratureType, ScalarKernelType>::integrate_singular(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs,
    const bool g_terms,
    const bool grad_g_terms
    )
{

    // projection of the observation point on the source triangle's local plane, assuming the
    // observation points are in the source's local coordinate system
    EigMatNX<Float, 2> rho_obs = r_obs.topRows(2);
    EigRowVec<Float> z = r_obs.row(2);
    EigRowVec<Float> z_abs = z.cwiseAbs();
    EigRowVec<Float> z_sq = z.array() * z.array();

    EigRowVec<Float> beta = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigRowVec<Float> t0_f2 = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigMatNX<Float, 2> u_f2 = EigMatNX<Float, 2>::Zero(2, r_obs.cols());
    EigMatNX<Float, 2> u_f3 = EigMatNX<Float, 2>::Zero(2, r_obs.cols());

    EigMatNX<Float, 2> plus_to_rho (2, r_obs.cols());
    EigMatNX<Float, 2> minus_to_rho (2, r_obs.cols());

    EigRowVec<Float> s_plus (1, r_obs.cols());
    EigRowVec<Float> s_minus (1, r_obs.cols());

    EigRowVec<Float> t0 (1, r_obs.cols());
    EigRowVec<Float> R0 (1, r_obs.cols());

    EigRowVec<Float> R_plus (1, r_obs.cols());
    EigRowVec<Float> R_minus (1, r_obs.cols());

    EigRowVec<Float> R0_sq (1, r_obs.cols());
    EigRowVec<Float> atan_plus (1, r_obs.cols());
    EigRowVec<Float> atan_minus (1, r_obs.cols());

    EigRowVec<Float> f2 (1, r_obs.cols());
    EigRowVec<Float> f3 (1, r_obs.cols());

    for (uint8_t idx = 0; idx < 3; idx++)
    {

        // Indices of the edge's vertices
        uint8_t idx_plus = idx;
        uint8_t idx_minus = (idx + 1) % 3;

        Edge<2> edge (src_tri.v(idx_plus), src_tri.v(idx_minus));
        EigColVecN<Float, 2> u_hat = { edge.unit_vec()[1], -edge.unit_vec()[0] };

        // Projection of the projected observation point to the line defined by the edge
        plus_to_rho = rho_obs.colwise() - edge.v(0);
        minus_to_rho = rho_obs.colwise() - edge.v(1);
        s_plus = edge.unit_vec().transpose() * plus_to_rho;
        s_minus = s_plus.array() - edge.length();
        t0 = -(u_hat.transpose() * plus_to_rho);
        R0 = Eigen::sqrt((t0.array() * t0.array()) + z_sq.array());

        // Distance from the observation points to each end point of the edge
        R_plus = Eigen::sqrt(plus_to_rho.colwise().squaredNorm().array() + z_sq.array());
        R_minus = Eigen::sqrt(minus_to_rho.colwise().squaredNorm().array() + z_sq.array());

        // Compute the terms which compose the final integrals
        R0_sq = R0.array() * R0.array();

        atan_plus = (t0.array() * s_plus.array()).atan2(
            R0_sq.array() + z_abs.array() * R_plus.array()
            );

        atan_minus = (t0.array() * s_minus.array()).atan2(
            R0_sq.array() + z_abs.array() * R_minus.array()
            );

        f2 = Eigen::log(((R_plus + s_plus).array() / (R_minus + s_minus).array()));
        f2 = (!f2.array().isFinite()).select(EigRowVec<Float>::Zero(1, f2.cols()), f2);

        f3 = s_plus.array() * R_plus.array() -
            s_minus.array() * R_minus.array() +
            R0_sq.array() * f2.array();

        beta += atan_plus - atan_minus;
        t0_f2 += (t0.array() * f2.array()).matrix();
        u_f2 += u_hat * f2;
        u_f3 += u_hat * f3;

        assert(atan_plus.array().isFinite().all() &&
            "SrcSingularity::compute_integral_terms(): atan_plus has nan or inf.");
        assert(atan_minus.array().isFinite().all() &&
            "SrcSingularity::compute_integral_terms(): atan_minus has nan or inf.");
        assert(f2.array().isFinite().all() &&
            "SrcSingularity::compute_integral_terms(): f2 has nan or inf.");
        assert(f3.array().isFinite().all() &&
            "SrcSingularity::compute_integral_terms(): f3 has nan or inf.");

    }

    SrcResult result;

    if (g_terms)
    {
        result.g = (t0_f2 - z_abs.cwiseProduct(beta)) / four_pi;

        result.rs_g = 0.5 * u_f3 / four_pi;
        result.rs_g += r_obs.topRows(2) * result.g.asDiagonal();
    }

    if (grad_g_terms)
    {
        EigRowVec<Float> sign = z.array() / z_abs.array();
        sign = (
            z_abs.array() < TRIANGLE_DEFAULT_TOL * src_tri.shortest_edge_length()
        ).select(1, sign);

        result.grad_g = EigMatNX<Complex, 3>::Zero(3, r_obs.cols());
        result.grad_g.topRows(2) = -u_f2;
        result.grad_g.row(2) = -sign.array() * beta.array();

        EigMatNX<Float, 3> grad_r = EigMatNX<Float, 3>::Zero(3, r_obs.cols());
        grad_r.topRows(2) = 0.5 * u_f3;
        grad_r.row(2) = z.array() * z_abs.array() * beta.array() - z.array() * t0_f2.array();

        result.grad_g += (half * k * k) * grad_r;
        result.grad_g /= four_pi;
    }

    return result;

};

}

#endif
