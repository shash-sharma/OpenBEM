// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based Gram matrix operators.
*/

#include "rwg/operators/gram.hpp"

#include <external/Eigen/Dense>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"


namespace bem::rwg
{

EigMat<Complex> VectorIdentityOp::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    ) const
{

    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    if (GeometryOps<3>::common_vertices(obs_tri, src_tri) < 3)
        return result;

    QuadratureData<3> qd = tri_quad_->compute(obs_tri);

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        // observer triangle edges
        for (uint8_t ii = 0; ii < 3; ++ii)
        {
            result(ii, jj) = qd.weights.dot(
                ((qd.points.colwise() - obs_tri.v((ii + 2) % 3)).transpose() *
                    (qd.points.colwise() - src_tri.v((jj + 2) % 3))).diagonal()
            );
        }
    }

    result.array() *= (
        Rwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    return result;

};


EigMat<Complex> RotVectorIdentityOp::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    ) const
{

    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    if (GeometryOps<3>::common_vertices(obs_tri, src_tri) < 3)
        return result;

    QuadratureData<3> qd = tri_quad_->compute(obs_tri);

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        // observer triangle edges
        for (uint8_t ii = 0; ii < 3; ++ii)
        {
            result(ii, jj) = -qd.weights.dot(
                (((qd.points.colwise() - obs_tri.v((ii + 2) % 3)
                    ).colwise().cross(obs_tri.normal())).transpose() * (
                        (qd.points.colwise() - src_tri.v((jj + 2) % 3)))).diagonal());
        }
    }

    result.array() *= (
        NxRwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    return result;

};

}

