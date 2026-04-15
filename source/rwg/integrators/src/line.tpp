// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Line integration over the source triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_LINE_I
#define BEM_RWG_OPINT_SRC_LINE_I

#include <cassert>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/edge.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/operations.hpp"


namespace bem::rwg
{

template <typename LineQuadratureType>
SrcResult SrcLineIntegrator<LineQuadratureType>::integrate_TEMP(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs
    )
{
    bool dc = true;
    if (std::abs(k) > float_eps)
        dc = false;

    const Complex jk = J * k;

    EigRowVec<Complex> I_alpha = EigRowVec<Complex>::Zero(1, r_obs.cols());
    EigRowVec<Complex> I_perp = EigRowVec<Complex>::Zero(1, r_obs.cols());
    EigMatNX<Complex, 2> I_beta = EigMatNX<Complex, 2>::Zero(2, r_obs.cols());
    EigMatNX<Complex, 2> I_par = EigMatNX<Complex, 2>::Zero(2, r_obs.cols());

    EigMat<Float> points_r = EigMat<Float>::Zero(line_quad_.ref_points().cols(), r_obs.cols());
    EigMat<Float> points_x = EigMat<Float>::Zero(line_quad_.ref_points().cols(), r_obs.cols());
    EigMat<Float> weights_r = EigMat<Float>::Zero(line_quad_.ref_points().cols(), r_obs.cols());
    EigMat<Float> weights_x = EigMat<Float>::Zero(line_quad_.ref_points().cols(), r_obs.cols());

    EigRowVec<Float> proj_h = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigMatNX<Float, 2> h_vec = EigMatNX<Float, 2>::Zero(2, r_obs.cols());
    EigMatNX<Float, 2> rproj_to_vminus = EigMatNX<Float, 2>::Zero(2, r_obs.cols());
    EigMatNX<Float, 2> rproj_to_vplus = EigMatNX<Float, 2>::Zero(2, r_obs.cols());

    EigRowVec<Float> theta_minus = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigRowVec<Float> theta_plus = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigRowVec<Float> x_minus = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigRowVec<Float> x_plus = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigRowVec<Float> r_minus = EigRowVec<Float>::Zero(1, r_obs.cols());
    EigRowVec<Float> r_plus = EigRowVec<Float>::Zero(1, r_obs.cols());

    EigRowVec<Complex> exp_jkr_minus = EigRowVec<Complex>::Zero(1, r_obs.cols());
    EigRowVec<Complex> exp_jkr_plus = EigRowVec<Complex>::Zero(1, r_obs.cols());

    EigMat<Float> rho = EigMat<Float>::Zero(line_quad_.ref_points().cols(), r_obs.cols());
    EigMat<Float> rhosq = EigMat<Float>::Zero(line_quad_.ref_points().cols(), r_obs.cols());
    EigMat<Float> points_rx = EigMat<Float>::Zero(line_quad_.ref_points().cols(), r_obs.cols());
    EigMat<Complex> exp_jkrt = EigMat<Complex>::Zero(line_quad_.ref_points().cols(), r_obs.cols());
    EigMat<Complex> exp_jkrx = EigMat<Complex>::Zero(line_quad_.ref_points().cols(), r_obs.cols());
    EigMat<Complex> x_sq_exp = EigMat<Complex>::Zero(line_quad_.ref_points().cols(), r_obs.cols());

    EigRowVec<Complex> I_alpha_edge, I_beta_edge, I_par_edge, I_perp_edge;
    if (base::compute_g_terms_)
    {
        I_alpha_edge.resize(1, r_obs.cols());
        I_beta_edge.resize(1, r_obs.cols());
    }
    if (base::compute_grad_g_terms_)
    {
        I_par_edge.resize(1, r_obs.cols());
        I_perp_edge.resize(1, r_obs.cols());
    }

    // Get the projection of the obs point on the triangle's plane, along
    // with information about where the projection point falls
    EigMatNX<Float, 2> proj_r;
    EigRowVec<Float> proj_d;
    src_tri.get_plane_projection(proj_r, proj_d, r_obs);

    EigRowVec<Float> proj_d_sign = proj_d.array().sign();
    proj_d = Eigen::abs(proj_d.array());

    EigRowVec<Float> proj_dsq = proj_d.array() * proj_d.array();

    EigRowVec<Complex> exp_jkd = Eigen::exp(-jk * proj_d.array());

    for (uint8_t idx_edge = 0; idx_edge < 3; ++idx_edge)
    {
        // Indices of the edge's vertices
        uint8_t idx_minus = idx_edge;
        uint8_t idx_plus = (idx_edge + 1) % 3;

        Edge<2> edge (src_tri.v(idx_minus), src_tri.v(idx_plus));

        // unit vector perpendicular to the edge pointing out of the triangle
        EigColVecN<Float, 2> u_hat = { edge.unit_vec()[1], -edge.unit_vec()[0] };

        // vectors from the projection point to to edge vertices
        rproj_to_vminus = (-proj_r).colwise() + edge.v(0);
        rproj_to_vplus = (-proj_r).colwise() + edge.v(1);

        // perpendicular displacement and vector from the projected obs point to this edge
        proj_h.noalias() = u_hat.transpose() * rproj_to_vminus;
        h_vec.noalias() = u_hat * proj_h;

        // local 1D integration limits for this edge
        x_minus.noalias() = -edge.unit_vec().transpose() * rproj_to_vminus;
        x_plus.noalias() = -edge.unit_vec().transpose() * rproj_to_vplus;

        assert(
            (Eigen::abs(
                (x_plus - x_minus).colwise().norm().array() - edge.length()
                ) <= TRIANGLE_DEFAULT_TOL * edge.length()).all() &&
                "SrcLineIntegrator::assemble_integrals(): x_plus and x_minus are incorrect"
            );

        for (std::size_t rr = 0; rr < r_obs.cols(); ++rr)
        {
            // Get linear quadrature points and weights
            line_quad_.compute_points_weights(x_minus.middleCols(rr, 1), x_plus.middleCols(rr, 1));

            // matrices of weights and local 1D coords of src points (rows) per obs points (cols)
            weights_x.col(rr) = line_quad_.ref_weights().transpose() * edge.length();
            points_x.col(rr) = line_quad_.points().transpose();
        }

        rhosq = (proj_h.array() * proj_h.array()).replicate(points_x.rows(), 1) +
            (points_x.array() * points_x.array());
        rho = Eigen::sqrt(rhosq.array());

        // Use angular quadrature points and weights for the (near-)singular case
        Float near_threshold = edge.length() * 1e-2;
        Float edge_len_lambda = edge.length() * std::real(k) / two_pi;
        bool use_angular = (rho.array() <= near_threshold).any() ||
            edge_len_lambda >= half;

        // Do not use angular points when an observation point lies along an edge
        if ((proj_h.array().abs() < float_eps).any() &&
            (proj_d.array().abs() < float_eps).any())
            use_angular = false;

        // Use simplified line integrals for non-singular cases - currently disabled
        bool use_simplified = (rho.array() > edge.length()).all();
        use_simplified = false; // TODO - test this

        if (!use_angular)
        {
            weights_r = weights_x.array() *
                Eigen::abs(proj_h.replicate(points_x.rows(), 1).array()) / rhosq.array();
            points_r = Eigen::sqrt(
                proj_dsq.array().replicate(points_x.rows(), 1) + rhosq.array()
                );
            points_rx = points_r;
        }
        else
        {
            // directed angles between the perpendicular projection and vectors to edge vertices
            theta_minus.noalias() = GeometryOps<2>::directed_angles_between_vectors(
                h_vec, rproj_to_vminus
                );
            theta_plus.noalias() = GeometryOps<2>::directed_angles_between_vectors(
                h_vec, rproj_to_vplus
                );

            for (std::size_t rr = 0; rr < r_obs.cols(); ++rr)
            {
                line_quad_.compute_points_weights(
                    theta_minus.middleCols(rr, 1), theta_plus.middleCols(rr, 1)
                    );

                // matrices of weights and distances between src (rows) and obs (cols) points
                weights_r.col(rr) = line_quad_.weights().transpose();
                points_r.col(rr) = Eigen::sqrt(
                    proj_dsq[rr] +
                    Eigen::pow(proj_h[rr] / Eigen::cos(line_quad_.points().array()), 2)
                    ).transpose();
            }

            points_rx = Eigen::sqrt(
                proj_dsq.array().replicate(points_x.rows(), 1) + rhosq.array()
                );
        }

        exp_jkrt = Eigen::exp(-jk * points_r.array());
        if (!use_angular)
            exp_jkrx = exp_jkrt;
        else
            exp_jkrx = Eigen::exp(-jk * points_rx.array());

        x_sq_exp = weights_x.array() * points_x.array() * points_x.array() * exp_jkrx.array();

        r_minus =
            Eigen::sqrt(proj_dsq.array() +
            rproj_to_vminus.colwise().squaredNorm().array());

        r_plus =
            Eigen::sqrt(proj_dsq.array() +
            rproj_to_vplus.colwise().squaredNorm().array());

        exp_jkr_minus = Eigen::exp(-jk * r_minus.array());
        exp_jkr_plus = Eigen::exp(-jk * r_plus.array());

        if (base::compute_g_terms_)
        {
            if (dc)
                I_alpha_edge = -(
                    weights_r.array() * (
                        points_r.array() - proj_d.replicate(points_r.rows(), 1).array()
                    )).colwise().sum();

            else
                I_alpha_edge = (
                    weights_r.array() * (exp_jkrt.array() -
                    exp_jkd.replicate(points_r.rows(), 1).array())
                    ).colwise().sum() / jk;

            I_alpha += (proj_h.array() < 0).select(-I_alpha_edge, I_alpha_edge);

            if (use_simplified && !use_angular)
            {
                if (!dc)
                    I_beta.noalias() -= u_hat * (
                        weights_x.array() * exp_jkrx.array()
                        ).colwise().sum().matrix() / jk;

                else
                    I_beta.noalias() -= u_hat * (
                        weights_x.array() * points_rx.array()
                        ).colwise().sum().matrix();
            }
            else
            {
                I_beta_edge = (x_sq_exp.array() / points_rx.array()).colwise().sum();
                I_beta -= u_hat * I_beta_edge;

                if (!dc)
                    I_beta.noalias() += edge.unit_vec() * ((-proj_h.array() / jk) * (exp_jkr_plus - exp_jkr_minus).array()).matrix();

                else
                    I_beta.noalias() += edge.unit_vec() * (proj_h.array() * (r_plus - r_minus).array()).matrix();
            }
        }

        if (base::compute_grad_g_terms_)
        {
            I_par_edge = -(x_sq_exp.array() / Eigen::pow(points_rx.array(), 3) * (-one - jk * points_rx.array())).colwise().sum();
            I_par.noalias() += u_hat * I_par_edge;
            I_par.noalias() += edge.unit_vec() * (proj_h.array() * (exp_jkr_plus.array() / r_plus.array() - exp_jkr_minus.array() / r_minus.array())).matrix();

            I_perp_edge = (
                weights_r.array() * exp_jkrt.array() / points_r.array()
                ).colwise().sum() * proj_d.array() - (weights_r.array() * exp_jkd.replicate(points_r.rows(), 1).array()).colwise().sum();
            I_perp_edge.array() *= proj_d_sign.array();
            I_perp += (proj_h.array() < 0).select(-I_perp_edge, I_perp_edge);
        }
    }

    SrcResult result;

    if (base::compute_g_terms_)
    {
        result.g = -I_alpha.array() / four_pi;
        result.rs_g.resize(2, r_obs.cols());
        result.rs_g.row(0) = (proj_r.row(0).array() * result.g.array() + I_beta.row(0).array() / four_pi);
        result.rs_g.row(1) = (proj_r.row(1).array() * result.g.array() + I_beta.row(1).array() / four_pi);
    }

    if (base::compute_grad_g_terms_)
    {
        result.grad_g.resize(3, r_obs.cols());
        result.grad_g.row(0) = -I_par.row(0).array() / four_pi;
        result.grad_g.row(1) = -I_par.row(1).array() / four_pi;
        result.grad_g.row(2) = I_perp.array() / four_pi;
    }

    return result;

};


template <typename LineQuadratureType>
SrcResult SrcLineIntegrator<LineQuadratureType>::integrate(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs
    )
{
    bool dc = true;
    if (std::abs(k) > float_eps)
        dc = false;

    const Complex jk = J * k;

    SrcResult result;
    if (base::compute_g_terms_)
    {
        result.g.resize(1, r_obs.cols());
        result.rs_g.resize(2, r_obs.cols());
    }
    if (base::compute_grad_g_terms_)
    {
        result.grad_g.resize(3, r_obs.cols());
    }

    for (Index ii = 0; ii < r_obs.cols(); ++ii)
    {

        // Get the projection of the obs point on the triangle's plane, along
        // with information about where the projection point falls
        EigColVecN<Float, 2> proj_r;
        Float proj_d;
        src_tri.get_plane_projection(proj_r, proj_d, r_obs.col(ii));

        Float proj_d_sign = (0 < proj_d) - (proj_d < 0);
        proj_d = std::abs(proj_d);
        Float proj_dsq = proj_d * proj_d;
        Complex exp_jkd = std::exp(-jk * proj_d);

        Complex I_alpha = 0;
        Complex I_perp = 0;
        EigColVecN<Complex, 2> I_beta = EigColVecN<Complex, 2>::Zero(2, 1);
        EigColVecN<Complex, 2> I_par = EigColVecN<Complex, 2>::Zero(2, 1);

        for (uint8_t idx_edge = 0; idx_edge < 3; ++idx_edge)
        {

            // Indices of the edge's vertices
            uint8_t idx_minus = idx_edge;
            uint8_t idx_plus = (idx_edge + 1) % 3;

            Edge<2> edge (src_tri.v(idx_minus), src_tri.v(idx_plus));

            // unit vector perpendicular to the edge pointing out of the triangle
            EigColVecN<Float, 2> u_hat = { edge.unit_vec()[1], -edge.unit_vec()[0] };

            // vectors from the projection point to to edge vertices
            EigColVecN<Float, 2> rproj_to_vminus = -proj_r + edge.v(0);
            EigColVecN<Float, 2> rproj_to_vplus = -proj_r + edge.v(1);

            // perpendicular displacement and vector from the projected obs point to this edge
            Float proj_h = u_hat.dot(rproj_to_vminus);
            EigColVecN<Float, 2> h_vec = u_hat * proj_h;

            // local 1D integration limits for this edge
            Float x_minus = -edge.unit_vec().dot(rproj_to_vminus);
            Float x_plus = -edge.unit_vec().dot(rproj_to_vplus);

            // Get linear quadrature points and weights
            line_quad_.compute_points_weights(
                EigColVecN<Float, 1> { x_minus }, EigColVecN<Float, 1> { x_plus }
                );

            points_x_ = line_quad_.points();
            weights_x_ = line_quad_.ref_weights() * edge.length();

            EigRowVec<Float> rhosq = (proj_h * proj_h) + (points_x_.array() * points_x_.array());
            EigRowVec<Float> rho = Eigen::sqrt(rhosq.array());

            // Use angular quadrature points and weights for the (near-)singular case
            Float near_threshold = edge.length() * 1e-2;
            Float edge_len_lambda = edge.length() * std::real(k) / two_pi;
            bool use_angular = (rho.array() <= near_threshold).any() ||
                edge_len_lambda >= half;

            // Do not use angular points when an observation point lies along an edge
            if ((std::abs(proj_h) < float_eps) && (proj_d < float_eps))
                use_angular = false;

            // Use simplified line integrals for non-singular cases - currently disabled
            bool use_simplified = (rho.array() > edge.length()).all();
            use_simplified = false; // TODO - test this

            if (!use_angular)
            {
                weights_r_ = (weights_x_.array() * std::abs(proj_h)) / rhosq.array();
                points_r_ = Eigen::sqrt(proj_dsq + rhosq.array());
                points_rx_ = points_r_;
            }
            else
            {
                // directed angles between the perpendicular projection and vectors to edge vertices
                Float theta_minus = GeometryOps<2>::directed_angle_between_vectors(
                    h_vec, rproj_to_vminus
                    );
                Float theta_plus = GeometryOps<2>::directed_angle_between_vectors(
                    h_vec, rproj_to_vplus
                    );

                line_quad_.compute_points_weights(
                    EigColVecN<Float, 1> { theta_minus }, EigColVecN<Float, 1> { theta_plus }
                    );

                weights_r_ = line_quad_.weights();
                points_r_ = Eigen::sqrt(
                    proj_dsq + Eigen::pow(proj_h / Eigen::cos(line_quad_.points().array()), 2)
                    );

                points_rx_ = Eigen::sqrt(proj_dsq + rhosq.array());
            }

            exp_jkrt_ = Eigen::exp(-jk * points_r_.array());
            if (!use_angular)
                exp_jkrx_ = exp_jkrt_;
            else
                exp_jkrx_ = Eigen::exp(-jk * points_rx_.array());

            x_sq_exp_ = weights_x_.array() * points_x_.array() * points_x_.array() * exp_jkrx_.array();

            Float r_minus = std::sqrt(proj_dsq + rproj_to_vminus.squaredNorm());
            Float r_plus = std::sqrt(proj_dsq + rproj_to_vplus.squaredNorm());

            Complex exp_jkr_minus = std::exp(-jk * r_minus);
            Complex exp_jkr_plus = std::exp(-jk * r_plus);

            Float sign = proj_h < 0 ? -one : one;

            if (base::compute_g_terms_)
            {
                if (dc)
                {
                    I_alpha += -(weights_r_.array() * (points_r_.array() - proj_d)).sum() * sign;

                    if (use_simplified && !use_angular)
                        I_beta.array() -= u_hat.array() * (weights_x_.array() * points_rx_.array()).sum();
                    else
                    {
                        I_beta.array() -= u_hat.array() * (x_sq_exp_.array() / points_rx_.array()).sum();
                        I_beta.array() += edge.unit_vec().array() * proj_h * (r_plus - r_minus);
                    }
                }
                else
                {
                    I_alpha += (weights_r_.array() * (exp_jkrt_.array() - exp_jkd)).sum() / jk * sign;

                    if (use_simplified && !use_angular)
                        I_beta.array() -= u_hat.array() * (weights_x_.array() * exp_jkrx_.array()).sum() / jk;
                    else
                    {
                        I_beta.array() -= u_hat.array() * (x_sq_exp_.array() / points_rx_.array()).sum();
                        I_beta.array() += edge.unit_vec().array() * (-proj_h / jk) *
                            (exp_jkr_plus - exp_jkr_minus);
                    }
                }
            }

            if (base::compute_grad_g_terms_)
            {
                I_par.array() -= u_hat.array() * (
                    x_sq_exp_.array() / Eigen::pow(points_rx_.array(), 3) *
                    (-one - jk * points_rx_.array())
                    ).sum();

                I_par.array() += edge.unit_vec().array() * proj_h *
                    (exp_jkr_plus / r_plus - exp_jkr_minus / r_minus);

                I_perp += (
                    (
                        weights_r_.array() * exp_jkrt_.array() / points_r_.array()
                        ).sum() * proj_d - (weights_r_.array() * exp_jkd).sum()
                    ) * proj_d_sign * sign;
            }

        }

        if (base::compute_g_terms_)
        {
            result.g[ii] = -I_alpha / four_pi;
            result.rs_g.col(ii) = (proj_r.array() * result.g[ii] + I_beta.array() / four_pi);
        }

        if (base::compute_grad_g_terms_)
        {
            result.grad_g.col(ii).topRows(2) = -I_par.array() / four_pi;
            result.grad_g(2, ii) = I_perp / four_pi;
        }

    }

    return result;

};

}

#endif
