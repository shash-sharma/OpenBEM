// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based single-layer potential BEM operators.
*/

#ifndef BEM_RWG_OPS_SINGLE_LAYER_I
#define BEM_RWG_OPS_SINGLE_LAYER_I

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"
#include "rwg/integrators/obs/base.hpp"


namespace bem::rwg
{

template <typename ObsIntegratorType>
EigMat<Complex> VectorSingleLayerOp<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    obs_integrator_.set_compute_terms(false, true, false, false);
    const ObsResult obs_result = obs_integrator_.integrate(k, obs_tri_local, src_tri_local);

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


template <typename ObsIntegratorType>
EigMat<Complex> VectorSingleLayerOp<ObsIntegratorType>::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    )
{
    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    const EigRowVecN<Complex, 12>& I = obs_result.rs_g;

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

            result(ii, jj) = (xm * xn + ym * yn) * I[0]
                - xm * I[1] - ym * I[2]
                - xn * I[3] - yn * I[4]
                + I[6] + I[10];
        }
    }

    result.array() *= (
        Rwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    return result;
};


template <typename ObsIntegratorType>
EigMat<Complex> RotVectorSingleLayerOp<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    obs_integrator_.set_compute_terms(false, true, false, false);
    const ObsResult obs_result = obs_integrator_.integrate(k, obs_tri_local, src_tri_local);

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


template <typename ObsIntegratorType>
EigMat<Complex> RotVectorSingleLayerOp<ObsIntegratorType>::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    )
{
    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    const EigRowVecN<Complex, 12>& I = obs_result.rs_g;

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

            result(ii, jj) = -(
                (ny * zm - nz * ym) * I[1]
                + (nz * xm - nx * zm) * I[2]
                + (nx * zm * yn - ny* zm * xn + nz * ym * xn - nz * xm * yn) * I[0]
                + nz * yn * I[3] - nz * xn * I[4] + (ny * xn - nx * yn) * I[5]
                + nz * I[7] - nz * I[9] - ny * I[8] + nx * I[11]
            );
        }
    }

    result.array() *= (
        Rwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    return result;
};


template <typename ObsIntegratorType>
EigMat<Complex> ScalarSingleLayerOp<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{
    Triangle<3> obs_tri_local;
    Triangle<2> src_tri_local;
    transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

    obs_integrator_.set_compute_terms(true, false, false, false);
    const ObsResult obs_result = obs_integrator_.integrate(k, obs_tri_local, src_tri_local);

    return assemble(k, obs_tri_local, src_tri_local.to_3d(), obs_result);
};


template <typename ObsIntegratorType>
EigMat<Complex> ScalarSingleLayerOp<ObsIntegratorType>::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    )
{
    EigMatMN<Complex, 1, 1> result = EigMatMN<Complex, 1, 1>::Zero(1, 1);

    result[0] = obs_result.g;
    result[0] *= Pulse::normalization(obs_tri) * Pulse::normalization(src_tri);

    return result;
};


template <typename ObsIntegratorType>
EigMat<Complex> RotGradScalarSingleLayerOp<ObsIntegratorType>::compute(
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
EigMat<Complex> RotGradScalarSingleLayerOp<ObsIntegratorType>::assemble(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri,
    const ObsResult& obs_result
    )
{
    EigMatMN<Complex, 3, 1> result = EigMatMN<Complex, 3, 1>::Zero(3, 1);

    if (GeometryOps<3>::common_vertices(obs_tri, src_tri) == 3)
        return result;

    const EigRowVecN<Complex, 9>& I = obs_result.grad_g;

    Float nx = obs_tri.normal()[0];
    Float ny = obs_tri.normal()[1];
    Float nz = obs_tri.normal()[2];

    // observer triangle edges
    for (uint8_t ii = 0; ii < 3; ++ii)
    {
        Float xm = obs_tri.v((ii + 2) % 3)[0];
        Float ym = obs_tri.v((ii + 2) % 3)[1];
        Float zm = obs_tri.v((ii + 2) % 3)[2];

        result[ii] = -(
            (zm * ny - ym * nz) * I[0] + (xm * nz - zm * nx) * I[3]
            + nz * I[1] - ny * I[2] - nz * I[4] + nx * I[5]
            + (ym * nx - ny * xm) * I[6]
            + ny * I[7] - nx * I[8]
            );

        // result[ii] = -(-nx * ((I[3] * zm - I[6] * ym) - (I[5] - I[8]))
        //     - ny * ((I[6] * xm - I[0] * zm) - (I[7] - I[2]))
        //     - nz * ((I[0] * ym - I[3] * xm) - (I[1] - I[4])));
    }

    result.array() *= Rwg::normalization(obs_tri).array() * Pulse::normalization(src_tri);

    return result;

};


template <typename ObsIntegratorType>
EigMat<Complex> VectorHypersingularOp<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{
    EigMatMN<Complex, 3, 3> result = op_g_.compute(k, obs_tri, src_tri);
    Complex hessg_term = op_hessg_.compute(k, obs_tri, src_tri)(0, 0) / k / k;
    result -= hessg_term * obs_tri.edge_polarities().transpose() * src_tri.edge_polarities();
    return result;
};


template <typename ObsIntegratorType>
EigMat<Complex> RotVectorHypersingularOp<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{
    EigMatMN<Complex, 3, 3> result = op_g_.compute(k, obs_tri, src_tri);
    EigMatMN<Complex, 3, 1> hessg_term = op_hessg_.compute(k, obs_tri, src_tri) / k / k;
    result += hessg_term * src_tri.edge_polarities();
    return result;
};

}

#endif
