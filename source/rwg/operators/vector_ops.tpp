// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes to generate the full set of vector RWG operators.
*/

#ifndef BEM_RWG_OPS_VECTOR_OPS_I
#define BEM_RWG_OPS_VECTOR_OPS_I

#include <stdexcept>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/integrators/obs/base.hpp"


namespace bem::rwg
{

template <typename ObsIntegratorType>
EigMatMN<Complex, 12, 3> VectorRwgOps<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{

    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    obs_integrator_.set_compute_terms(true, true, true, true);
    const ObsResult obs_result = obs_integrator_.integrate(k, obs_tri_local, src_tri_local);

    EigMatMN<Complex, 12, 3> ops;

    ops.middleRows(0, 3) = vector_single_layer_.assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
    ops.middleRows(3, 3) = vector_double_layer_pv_.assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
    ops.middleRows(6, 3) = rot_vector_single_layer_.assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
    ops.middleRows(9, 3) = rot_vector_double_layer_pv_.assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);

    if (!helmholtz_kernel_)
    {
        Complex h = scalar_single_layer_.assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result)[0] / k / k;
        ops.middleRows(0, 3) -= h * obs_tri.edge_polarities().transpose() * src_tri.edge_polarities();

        EigMatMN<Complex, 3, 1> rot_h = rot_grad_scalar_single_layer_.assemble(
            k, obs_tri_local, src_tri_local.to_3d(), obs_result
            ) / k / k;
        ops.middleRows(6, 3) += rot_h * src_tri.edge_polarities();
    }

    return ops;

};

}

#endif
