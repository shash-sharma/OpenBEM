// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based double-layer potential BEM operators.
*/

#ifndef BEM_RWG_OPS_DOUBLE_LAYER_I
#define BEM_RWG_OPS_DOUBLE_LAYER_I

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/operations.hpp"
#include "rwg/function_space.hpp"
#include "rwg/integrators/obs/base.hpp"


namespace bem::rwg
{

template <typename ObsIntegratorType>
EigMat<Complex> VectorDoubleLayerPvOp<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    obs_integrator_.set_compute_terms(false, false, true, false);
    const ObsResult obs_result = obs_integrator_.integrate(k, obs_tri_local, src_tri_local);

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


template <typename ObsIntegratorType>
EigMat<Complex> VectorDoubleLayerPvOp<ObsIntegratorType>::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    )
{

    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    if (GeometryOps<3>::common_vertices(obs_tri, src_tri) == 3 ||
        GeometryOps<3>::check_coplanar_triangles(obs_tri, src_tri))
        return result;

    const EigRowVecN<Complex, 9>& I = obs_result.grad_g;

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        Float xn = src_tri.v((jj + 2) % 3)[0];
        Float yn = src_tri.v((jj + 2) % 3)[1];

        // observer triangle edges
        for (uint8_t ii = 0; ii < 3; ++ii)
        {
            Float xm = obs_tri.v((ii + 2) % 3)[0];
            Float ym = obs_tri.v((ii + 2) % 3)[1];
            Float zm = obs_tri.v((ii + 2) % 3)[2];

            result(ii, jj) = -(-zm * yn * I[0]
                + zm * I[1] + (yn - ym) * I[2]
                + zm * xn * I[3] - zm * I[4] + (xm - xn) * I[5]
                + (xm * yn - ym * xn) * I[6]
                + (ym - yn) * I[7] + (xn - xm) * I[8]);
        }
    }

    result.array() *= (
        Rwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    return result;

};


template <typename ObsIntegratorType>
EigMat<Complex> RotVectorDoubleLayerPvOp<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    obs_integrator_.set_compute_terms(false, false, true, true);
    const ObsResult obs_result = obs_integrator_.integrate(k, obs_tri_local, src_tri_local);

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


template <typename ObsIntegratorType>
EigMat<Complex> RotVectorDoubleLayerPvOp<ObsIntegratorType>::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    )
{

    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    if (GeometryOps<3>::common_vertices(obs_tri, src_tri) == 3 ||
        GeometryOps<3>::check_coplanar_triangles(obs_tri, src_tri))
        return result;

    const EigRowVecN<Complex, 9>& I = obs_result.grad_g;
    const EigRowVecN<Complex, 15>& I_rot = obs_result.rot_grad_g;

    Float nx = obs_tri.normal()[0];
    Float ny = obs_tri.normal()[1];
    Float nz = obs_tri.normal()[2];

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        Float xn = src_tri.v((jj + 2) % 3)[0];
        Float yn = src_tri.v((jj + 2) % 3)[1];

        // observer triangle edges
        for (uint8_t ii = 0; ii < 3; ++ii)
        {
            Float xm = obs_tri.v((ii + 2) % 3)[0];
            Float ym = obs_tri.v((ii + 2) % 3)[1];
            Float zm = obs_tri.v((ii + 2) % 3)[2];

            result(ii, jj) =
                yn * (nx * ym - ny * xm) * I[0]
                + (ny * xm - nx * (ym + yn)) * I[1]
                + (nz * xm - nx * zm) * I[2]
                + xn * (ny * xm - nx * ym) * I[3]
                + (nx * ym - ny * (xm + xn)) * I[4]
                + (nz * ym - ny * zm) * I[5]
                + (xn * (nz * xm - nx * zm) + yn * (nz * ym - ny * zm)) * I[6]
                + (zm * nx - nz * (xm + xn)) * I[7]
                + (zm * ny - nz * (ym + yn)) * I[8]
                + yn * ny * I_rot[0] - ny * I_rot[1] - nz * I_rot[2]
                + nx * I_rot[3] + nx * I_rot[4]
                + xn * nx * I_rot[5] - nx * I_rot[6] - nz * I_rot[7]
                + ny * I_rot[8] + ny * I_rot[9]
                - nx * I_rot[10] - ny * I_rot[11] + (nx * xn + ny * yn) * I_rot[12]
                + nz * I_rot[13] + nz * I_rot[14];
        }
    }

    result.array() *= (
        NxRwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    return result;

};

}

#endif
