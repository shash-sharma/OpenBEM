// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based single-layer potential BEM projectors.
*/

#ifndef BEM_RWG_PROJ_SINGLE_LAYER_I
#define BEM_RWG_PROJ_SINGLE_LAYER_I

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"


namespace bem::rwg
{

template <typename SrcIntegratorType>
EigMatXN<Complex, 3> VectorSingleLayerProj<SrcIntegratorType>::compute(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{

    // Convert to the source triangle's local coordinate system
    EigMatNX<Float, 3> obs_points_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_points_local, src_tri_local, obs_points, src_tri);

    src_integrator_.set_compute_terms(true, false);
    const SrcResult src_result = src_integrator_.integrate(k, src_tri_local, obs_points_local);

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigRowVecN<Float, 3> norms = Rwg::normalization(src_tri);
    EigMatXN<Complex, 3> result = EigMatXN<Complex, 3>::Zero(3 * obs_points.cols(), 3);

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        EigMatNX<Complex, 2> rwg_g = src_result.rs_g - src_tri_local.v((jj + 2) % 3) * src_result.g;
        result.col(jj) = (local_uvw.leftCols(2) * rwg_g).reshaped(3 * obs_points.cols(), 1) *
                            norms(jj);
    }

    return result;

};


template <typename SrcIntegratorType>
EigMatXN<Complex, 1> ScalarSingleLayerProj<SrcIntegratorType>::compute(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{

    // Convert to the source triangle's local coordinate system
    EigMatNX<Float, 3> obs_points_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_points_local, src_tri_local, obs_points, src_tri);

    src_integrator_.set_compute_terms(true, false);
    const SrcResult src_result = src_integrator_.integrate(k, src_tri_local, obs_points_local);

    return src_result.g.transpose() * Pulse::normalization(src_tri);

};


template <typename SrcIntegratorType>
EigMatXN<Complex, 1> GradScalarSingleLayerProj<SrcIntegratorType>::compute(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{

    // Convert to the source triangle's local coordinate system
    EigMatNX<Float, 3> obs_points_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_points_local, src_tri_local, obs_points, src_tri);

    src_integrator_.set_compute_terms(false, true);
    const SrcResult src_result = src_integrator_.integrate(k, src_tri_local, obs_points_local);

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigMatXN<Complex, 1> result = (
        local_uvw * src_result.grad_g
        ).reshaped(3 * obs_points.cols(), 1) * Pulse::normalization(src_tri);

    return result;

};


template <typename SrcIntegratorType>
EigMatXN<Complex, 3> VectorHypersingularProj<SrcIntegratorType>::compute(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{
    EigMatXN<Complex, 3> result = proj_g_.compute(k, obs_points, src_tri);
    EigMatXN<Complex, 1> hessg_term = proj_gradg_.compute(k, obs_points, src_tri) / k / k;
    result += hessg_term * src_tri.edge_polarities();
    return result;
};

}

#endif
