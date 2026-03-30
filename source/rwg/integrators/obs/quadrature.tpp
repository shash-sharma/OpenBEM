// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Quadrature over the observation triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_OBS_QUAD_I
#define BEM_RWG_OPINT_OBS_QUAD_I

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/operations.hpp"
#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/src/base.hpp"


namespace bem::rwg
{

template <typename TriangleQuadratureType, typename SrcIntegratorType>
ObsResult ObsQuadrature<TriangleQuadratureType, SrcIntegratorType>::integrate(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<2>& src_tri
    )
{

    // Evaluation function for iterative or adaptive numerical integration
    auto eval = [&](ConstEigRef<EigMatNX<Float, 3>> r_obs) -> EigRowVec<Complex>
    {
        src_integrator_.set_compute_terms(true, false);
        return src_integrator_.integrate(k, src_tri, r_obs).g;
    };

    tri_quad_.compute_points_weights(obs_tri, eval);

    // Assemble the integration results
    src_integrator_.set_compute_terms(
        base::compute_g_term_ || base::compute_rs_g_terms_,
        base::compute_grad_g_terms_ || base::compute_rot_grad_g_terms_
        );
    SrcResult src_result = src_integrator_.integrate(k, src_tri, tri_quad_.points());

    ObsResult obs_result;

    if (base::compute_g_term_)
        obs_result.g = g_term(src_result);

    if (base::compute_rs_g_terms_)
        obs_result.rs_g = rs_g_terms(src_result);

    if (base::compute_grad_g_terms_)
        obs_result.grad_g = grad_g_terms(src_result);

    if (base::compute_rot_grad_g_terms_)
        obs_result.rot_grad_g = rot_grad_g_terms(src_result);

    return obs_result;

};


template <typename TriangleQuadratureType, typename SrcIntegratorType>
Complex ObsQuadrature<TriangleQuadratureType, SrcIntegratorType>::g_term(
    const SrcResult& src_result
    )
{
    return tri_quad_.weights().dot(src_result.g);
};


template <typename TriangleQuadratureType, typename SrcIntegratorType>
EigRowVecN<Complex, 12> ObsQuadrature<TriangleQuadratureType, SrcIntegratorType>::rs_g_terms(
    const SrcResult& src_result
    )
{

    EigRowVec<Complex> wgw = src_result.g * tri_quad_.weights().asDiagonal();
    EigMatNX<Complex, 2> rs_wgw = src_result.rs_g * tri_quad_.weights().asDiagonal();

    EigColVecN<Complex, 2> sum_rs_wgw = rs_wgw.rowwise().sum();
    EigRowVecN<Complex, 3> p_wgw = wgw * tri_quad_.points().transpose();
    EigMatMN<Complex, 2, 3> p_rs_wgw = rs_wgw * tri_quad_.points().transpose();

    EigRowVecN<Complex, 12> I = EigRowVecN<Complex, 12>::Zero(1, 12);

    I[0] = wgw.sum();
    I[1] = sum_rs_wgw[0];
    I[2] = sum_rs_wgw[1];

    I[3] = p_wgw[0];
    I[4] = p_wgw[1];
    I[5] = p_wgw[2];

    I[6] = p_rs_wgw(0, 0);
    I[7] = p_rs_wgw(0, 1);
    I[8] = p_rs_wgw(0, 2);

    I[9] = p_rs_wgw(1, 0);
    I[10] = p_rs_wgw(1, 1);
    I[11] = p_rs_wgw(1, 2);

    return I;

};


template <typename TriangleQuadratureType, typename SrcIntegratorType>
EigRowVecN<Complex, 9> ObsQuadrature<TriangleQuadratureType, SrcIntegratorType>::grad_g_terms(
    const SrcResult& src_result
    )
{

    EigMatNX<Complex, 3> grad_wgw = src_result.grad_g * tri_quad_.weights().asDiagonal();
    EigColVecN<Complex, 3> sum_grad_wgw = grad_wgw.rowwise().sum();
    EigMatMN<Complex, 3, 3> p_grad_wgw = grad_wgw * tri_quad_.points().transpose();

    EigRowVecN<Complex, 9> I = EigRowVecN<Complex, 9>::Zero(1, 9);

    I[0] = sum_grad_wgw[0];
    I[1] = p_grad_wgw(0, 1);
    I[2] = p_grad_wgw(0, 2);

    I[3] = sum_grad_wgw[1];
    I[4] = p_grad_wgw(1, 0);
    I[5] = p_grad_wgw(1, 2);

    I[6] = sum_grad_wgw[2];
    I[7] = p_grad_wgw(2, 0);
    I[8] = p_grad_wgw(2, 1);

    return I;

};


template <typename TriangleQuadratureType, typename SrcIntegratorType>
EigRowVecN<Complex, 15> ObsQuadrature<TriangleQuadratureType, SrcIntegratorType>::rot_grad_g_terms(
    const SrcResult& src_result
    )
{

    EigMatNX<Complex, 3> grad_wgw = src_result.grad_g * tri_quad_.weights().asDiagonal();
    EigMatMN<Complex, 3, 3> p_grad_wgw = grad_wgw * tri_quad_.points().transpose();

    EigMatMN<Complex, 3, 1> xy_grad_wgw = grad_wgw * (
        tri_quad_.points().row(0).array() * tri_quad_.points().row(1).array()
        ).matrix().transpose();
    EigMatMN<Complex, 3, 1> xz_grad_wgw = grad_wgw * (
        tri_quad_.points().row(0).array() * tri_quad_.points().row(2).array()
        ).matrix().transpose();
    EigMatMN<Complex, 3, 1> yz_grad_wgw = grad_wgw * (
        tri_quad_.points().row(1).array() * tri_quad_.points().row(2).array()
        ).matrix().transpose();

    EigMatMN<Complex, 3, 1> xx_grad_wgw = grad_wgw * (
        tri_quad_.points().row(0).array() * tri_quad_.points().row(0).array()
        ).matrix().transpose();
    EigMatMN<Complex, 3, 1> yy_grad_wgw = grad_wgw * (
        tri_quad_.points().row(1).array() * tri_quad_.points().row(1).array()
        ).matrix().transpose();
    EigMatMN<Complex, 3, 1> zz_grad_wgw = grad_wgw * (
        tri_quad_.points().row(2).array() * tri_quad_.points().row(2).array()
        ).matrix().transpose();

    EigRowVecN<Complex, 15> I = EigRowVecN<Complex, 15>::Zero(1, 15);

    I[0] = p_grad_wgw(0, 0);
    I[5] = p_grad_wgw(1, 1);
    I[12] = p_grad_wgw(2, 2);

    I[1] = xy_grad_wgw[0];
    I[2] = xz_grad_wgw[0];
    I[3] = yy_grad_wgw[0];
    I[4] = zz_grad_wgw[0];

    I[6] = xy_grad_wgw[1];
    I[7] = yz_grad_wgw[1];
    I[8] = xx_grad_wgw[1];
    I[9] = zz_grad_wgw[1];

    I[10] = xz_grad_wgw[2];
    I[11] = yz_grad_wgw[2];
    I[13] = xx_grad_wgw[2];
    I[14] = yy_grad_wgw[2];

    return I;

};

}

#endif
