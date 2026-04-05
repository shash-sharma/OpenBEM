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
    ConstEigRef<EigMatNX<Float, 3>> r_obs
    )
{
    compute_integral_terms(src_tri, r_obs);

    SrcResult result_singular = assemble_integrals(k, src_tri, r_obs);

    src_quad_.set_compute_terms(compute_g_terms_, compute_grad_g_terms_);
    SrcResult result_nonsingular = src_quad_.integrate(k, src_tri, r_obs);

    SrcResult result;
    result.g = result_singular.g + result_nonsingular.g;
    result.rs_g = result_singular.rs_g + result_nonsingular.rs_g;
    result.grad_g = result_singular.grad_g + result_nonsingular.grad_g;

    return result;
};


template <typename TriangleQuadratureType, typename ScalarKernelType>
void SrcSingularity<TriangleQuadratureType, ScalarKernelType>::compute_integral_terms(
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs
    )
{

    // Projection of the observation point on the source triangle's local plane
    EigMatNX<Float, 2> rho_obs = r_obs.topRows(2);
    z_ = r_obs.row(2);
    z_abs_ = z_.cwiseAbs();
    EigRowVec<Float> z_sq = z_.array() * z_.array();

    beta_ = EigRowVec<Float>::Zero(1, r_obs.cols());
    t0_f2_ = EigRowVec<Float>::Zero(1, r_obs.cols());
    u_f2_ = EigMatNX<Float, 2>::Zero(2, r_obs.cols());
    u_f3_ = EigMatNX<Float, 2>::Zero(2, r_obs.cols());

    for (uint8_t idx = 0; idx < 3; idx++)
    {

        // Indices of the edge's vertices
        uint8_t idx_plus = idx;
        uint8_t idx_minus = (idx + 1) % 3;

        Edge<2> edge (src_tri.v(idx_plus), src_tri.v(idx_minus));
        EigColVecN<Float, 2> u_hat = { edge.unit_vec()[1], -edge.unit_vec()[0] };

        // Projection of the projected observation point to the line defined by the edge
        EigMatNX<Float, 2> plus_to_rho = rho_obs.colwise() - edge.v(0);
        EigMatNX<Float, 2> minus_to_rho = rho_obs.colwise() - edge.v(1);
        EigRowVec<Float> s_plus = edge.unit_vec().transpose() * plus_to_rho;
        EigRowVec<Float> s_minus = s_plus.array() - edge.length();
        EigRowVec<Float> t0 = -(u_hat.transpose() * plus_to_rho);
        EigRowVec<Float> R0 = Eigen::sqrt((t0.array() * t0.array()) + z_sq.array());

        // Distance from the observation points to each end point of the edge
        EigRowVec<Float> R_plus = Eigen::sqrt(plus_to_rho.colwise().squaredNorm().array() + z_sq.array());
        EigRowVec<Float> R_minus = Eigen::sqrt(minus_to_rho.colwise().squaredNorm().array() + z_sq.array());

        // Compute the terms which compose the final integrals
        EigRowVec<Float> R0_sq = R0.array() * R0.array();
        EigRowVec<Float> atan_plus (t0.size()), atan_minus (t0.size());
        for (std::size_t ii = 0; ii < t0.size(); ++ii)
        {
            atan_plus[ii] = std::atan2(
                t0[ii] * s_plus[ii],
                R0_sq[ii] + z_abs_[ii] * R_plus[ii]
            );
            atan_minus[ii] = std::atan2(
                t0[ii] * s_minus[ii],
                R0_sq[ii] + z_abs_[ii] * R_minus[ii]
            );
        }

        EigRowVec<Float> f2 = Eigen::log(((R_plus + s_plus).array() / (R_minus + s_minus).array()));

        // f2 = (t0.array().abs() < tol &&
        //       ((R_plus + s_plus).array() < tol || (R_minus + s_minus).array() < tol)).select(
        //     EigRowVec<Float>::Zero(1, f2.cols()),
        //     f2
        //     );
        f2 = (!f2.array().isFinite()).select(EigRowVec<Float>::Zero(1, f2.cols()), f2);

        EigRowVec<Float> f3 =
            s_plus.array() * R_plus.array() -
            s_minus.array() * R_minus.array() +
            R0_sq.array() * f2.array();

        beta_ += atan_plus - atan_minus;
        t0_f2_ += (t0.array() * f2.array()).matrix();
        u_f2_ += u_hat * f2;
        u_f3_ += u_hat * f3;

        assert(atan_plus.array().isFinite().all() &&
            "SrcSingularity::compute_integral_terms(): atan_plus has nan or inf.");
        assert(atan_minus.array().isFinite().all() &&
            "SrcSingularity::compute_integral_terms(): atan_minus has nan or inf.");
        assert(f2.array().isFinite().all() &&
            "SrcSingularity::compute_integral_terms(): f2 has nan or inf.");
        assert(f3.array().isFinite().all() &&
            "SrcSingularity::compute_integral_terms(): f3 has nan or inf.");

        // std::cout << R_plus << ", " << R_minus << ", " << s_plus << ", " << s_minus << ", " << t0 << ", " << R0 << ", " << atan_plus << ", " << atan_minus << ", " << it.beta << ", " << it.f2 << ", " << it.t0_f2 << ", " << it.f3 << ", " << it.f1 << std::endl;

    }

    // _it.beta = (_it.beta.array().isFinite()).select(_it.beta, 0.0);

    // for (int ii = 0; ii < r_obs.cols(); ++ii)
    // {
    //     EigColVecN<Float, 3> r_obs_col = r_obs.col(ii);
    //     if (src_tri.point_in_triangle(r_obs_col))
    //         std::cout << _it.u_f2.col(ii) << std::endl;
    // }

    return;

};


template <typename TriangleQuadratureType, typename ScalarKernelType>
SrcResult SrcSingularity<TriangleQuadratureType, ScalarKernelType>::assemble_integrals(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs
    )
{

    const Float four_pi = 4.0 * pi;

    SrcResult result;

    if (base::compute_g_terms_)
    {
        result.g = (t0_f2_ - z_abs_.cwiseProduct(beta_)) / four_pi;

        result.rs_g = 0.5 * u_f3_ / four_pi;
        result.rs_g += r_obs.topRows(2) * result.g.asDiagonal();
    }

    if (base::compute_grad_g_terms_)
    {
        EigRowVec<Float> sign = z_.array() / z_abs_.array();
        sign = (
            z_abs_.array() < TRIANGLE_DEFAULT_TOL * src_tri.shortest_edge_length()
        ).select(1, sign);

        result.grad_g = EigMatNX<Complex, 3>::Zero(3, r_obs.cols());
        result.grad_g.topRows(2) = -u_f2_;
        result.grad_g.row(2) = -sign.array() * beta_.array();

        EigMatNX<Float, 3> grad_r = EigMatNX<Float, 3>::Zero(3, r_obs.cols());
        grad_r.topRows(2) = 0.5 * u_f3_;
        grad_r.row(2) = z_.array() * z_abs_.array() * beta_.array() - z_.array() * t0_f2_.array();

        result.grad_g += (half * k * k) * grad_r;
        result.grad_g /= four_pi;
    }

    return result;

};

}

#endif
