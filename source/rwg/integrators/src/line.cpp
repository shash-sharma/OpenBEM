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

#include "rwg/integrators/src/line.hpp"

#include <cassert>
#include <stdexcept>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/edge.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/operations.hpp"


namespace bem::rwg
{

SrcResult SrcLineIntegrator::integrate(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs,
    const bool g_terms,
    const bool grad_g_terms
    )
{

    EigRowVec<Float> ref_points;
    EigRowVec<Float> ref_weights;

    if (auto line_quad = std::dynamic_pointer_cast<GaussLineQuadrature<1>> (line_quad_))
    {
        ref_points = line_quad->ref_points();
        ref_weights = line_quad->ref_weights();
    }
    else if (auto line_quad = std::dynamic_pointer_cast<TrapzLineQuadrature<1>> (line_quad_))
    {
        ref_points = line_quad->ref_points();
        ref_weights = line_quad->ref_weights();
    }
    else
        throw std::runtime_error(
            "SrcLineIntegrator::integrate(): Given line quadrature type is not currently supported."
            );

    Index num_points = ref_points.cols();

    EigRowVec<Float> weights_x = EigRowVec<Float>::Zero(1, num_points);
    EigRowVec<Float> weights_r = EigRowVec<Float>::Zero(1, num_points);
    EigRowVec<Float> points_x = EigRowVec<Float>::Zero(1, num_points);
    EigRowVec<Float> points_r = EigRowVec<Float>::Zero(1, num_points);
    EigRowVec<Float> points_rx = EigRowVec<Float>::Zero(1, num_points);

    EigRowVec<Complex> exp_jkrt = EigRowVec<Complex>::Zero(1, num_points);
    EigRowVec<Complex> exp_jkrx = EigRowVec<Complex>::Zero(1, num_points);
    EigRowVec<Complex> x_sq_exp = EigRowVec<Complex>::Zero(1, num_points);

    bool dc = true;
    if (std::abs(k) > float_eps)
        dc = false;

    const Complex jk = J * k;

    SrcResult result;
    if (g_terms)
    {
        result.g.resize(1, r_obs.cols());
        result.rs_g.resize(2, r_obs.cols());
    }
    if (grad_g_terms)
    {
        result.grad_g.resize(3, r_obs.cols());
    }

    for (Index ii = 0; ii < r_obs.cols(); ++ii)
    {

        // projection of the observation point on the source triangle's local plane, assuming the
        // observation points are in the source's local coordinate system
        EigColVecN<Float, 2> proj_r = r_obs.col(ii).topRows(2);
        Float proj_d = r_obs(2, ii);

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

            // indices of the edge's vertices
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

            // linear quadrature points and weights
            QuadratureData<1> qd = line_quad_->compute(
                EigColVecN<Float, 1> { x_minus }, EigColVecN<Float, 1> { x_plus }
                );

            points_x = qd.points;
            weights_x = ref_weights * edge.length();
            // weights_x = qd.weights;

            EigRowVec<Float> rhosq = (proj_h * proj_h) + (points_x.array() * points_x.array());
            EigRowVec<Float> rho = Eigen::sqrt(rhosq.array());

            // use angular quadrature points and weights for the (near-)singular case
            Float near_threshold = edge.length() * 1e-2;
            Float edge_len_lambda = edge.length() * std::real(k) / two_pi;
            bool use_angular = (rho.array() <= near_threshold).any() ||
                edge_len_lambda >= half;

            // do not use angular points when an observation point lies along an edge
            if ((std::abs(proj_h) < float_eps) && (proj_d < float_eps))
                use_angular = false;

            // use simplified line integrals for non-singular cases - currently disabled
            bool use_simplified = (rho.array() > edge.length()).all();
            use_simplified = false; // TODO - test this

            if (!use_angular)
            {
                weights_r = (weights_x.array() * std::abs(proj_h)) / rhosq.array();
                points_r = Eigen::sqrt(proj_dsq + rhosq.array());
                points_rx = points_r;
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

                qd = line_quad_->compute(
                    EigColVecN<Float, 1> { theta_minus }, EigColVecN<Float, 1> { theta_plus }
                    );

                weights_r = qd.weights;
                points_r = Eigen::sqrt(
                    proj_dsq + Eigen::pow(proj_h / Eigen::cos(qd.points.array()), 2)
                    );

                points_rx = Eigen::sqrt(proj_dsq + rhosq.array());
            }

            exp_jkrt = Eigen::exp(-jk * points_r.array());
            if (!use_angular)
                exp_jkrx = exp_jkrt;
            else
                exp_jkrx = Eigen::exp(-jk * points_rx.array());

            x_sq_exp = weights_x.array() * points_x.array() * points_x.array() * exp_jkrx.array();

            Float r_minus = std::sqrt(proj_dsq + rproj_to_vminus.squaredNorm());
            Float r_plus = std::sqrt(proj_dsq + rproj_to_vplus.squaredNorm());

            Complex exp_jkr_minus = std::exp(-jk * r_minus);
            Complex exp_jkr_plus = std::exp(-jk * r_plus);

            Float sign = proj_h < 0 ? -one : one;

            if (g_terms)
            {
                if (dc)
                {
                    I_alpha += -(weights_r.array() * (points_r.array() - proj_d)).sum() * sign;

                    if (use_simplified && !use_angular)
                        I_beta.array() -= u_hat.array() * (weights_x.array() * points_rx.array()).sum();
                    else
                    {
                        I_beta.array() -= u_hat.array() * (x_sq_exp.array() / points_rx.array()).sum();
                        I_beta.array() += edge.unit_vec().array() * proj_h * (r_plus - r_minus);
                    }
                }
                else
                {
                    I_alpha += (weights_r.array() * (exp_jkrt.array() - exp_jkd)).sum() / jk * sign;

                    if (use_simplified && !use_angular)
                        I_beta.array() -= u_hat.array() * (weights_x.array() * exp_jkrx.array()).sum() / jk;
                    else
                    {
                        I_beta.array() -= u_hat.array() * (x_sq_exp.array() / points_rx.array()).sum();
                        I_beta.array() += edge.unit_vec().array() * (-proj_h / jk) *
                            (exp_jkr_plus - exp_jkr_minus);
                    }
                }
            }

            if (grad_g_terms)
            {
                I_par.array() -= u_hat.array() * (
                    x_sq_exp.array() / Eigen::pow(points_rx.array(), 3) *
                    (-one - jk * points_rx.array())
                    ).sum();

                I_par.array() += edge.unit_vec().array() * proj_h *
                    (exp_jkr_plus / r_plus - exp_jkr_minus / r_minus);

                I_perp += (
                    (
                        weights_r.array() * exp_jkrt.array() / points_r.array()
                        ).sum() * proj_d - (weights_r.array() * exp_jkd).sum()
                    ) * proj_d_sign * sign;
            }

        }

        if (g_terms)
        {
            result.g[ii] = -I_alpha / four_pi;
            result.rs_g.col(ii) = (proj_r.array() * result.g[ii] + I_beta.array() / four_pi);
        }

        if (grad_g_terms)
        {
            result.grad_g.col(ii).topRows(2) = -I_par.array() / four_pi;
            result.grad_g(2, ii) = I_perp / four_pi;
        }

    }

    return result;

};

}

