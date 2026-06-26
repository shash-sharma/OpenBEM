// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based double-layer potential BEM projectors.
*/

#include "rwg/projectors/double_layer.hpp"

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"


namespace bem::rwg
{

EigMat<Complex> VectorDoubleLayerProj::compute(
    const Complex k,
    ConstEigRef<EigMatNX<Float, 3>> obs_points,
    const Triangle<3>& src_tri
    )
{

    // Convert to the source triangle's local coordinate system
    EigMatNX<Float, 3> obs_points_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_points_local, src_tri_local, obs_points, src_tri);

    const SrcResult src_result = src_integrator_->integrate(
        k, src_tri_local, obs_points_local, false, true
        );

    EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
    EigRowVecN<Float, 3> norms = Rwg::normalization(src_tri);
    EigMatXN<Complex, 3> result = EigMatXN<Complex, 3>::Zero(3 * obs_points.cols(), 3);
    EigMatNX<Complex, 3> result_local = EigMatNX<Complex, 3>::Zero(3, obs_points.cols());
    EigMatNX<Float, 3> r_diff = EigMatNX<Float, 3>::Zero(3, obs_points.cols());

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        r_diff = obs_points_local;
        r_diff.topRows(2).colwise() -= src_tri_local.v((jj + 2) % 3);
        result_local.row(0) = r_diff.row(1).array() * src_result.grad_g.row(2).array() -
                              r_diff.row(2).array() * src_result.grad_g.row(1).array();
        result_local.row(1) = r_diff.row(2).array() * src_result.grad_g.row(0).array() -
                              r_diff.row(0).array() * src_result.grad_g.row(2).array();
        result_local.row(2) = r_diff.row(0).array() * src_result.grad_g.row(1).array() -
                              r_diff.row(1).array() * src_result.grad_g.row(0).array();
        result.col(jj) = -(local_uvw * result_local).reshaped(3 * obs_points.cols(), 1) *
                            norms(jj);
    }

    return result;

};

}

