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

#ifndef BEM_RWG_OPS_GRAM_I
#define BEM_RWG_OPS_GRAM_I

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"


namespace bem::rwg
{

template <typename TriangleQuadratureType>
EigMatMN<Complex, 3, 3> RwgRwgOp<TriangleQuadratureType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{

    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    if (GeometryOps<3>::common_vertices(obs_tri, src_tri) < 3)
        return result;

    tri_quad_.compute_points_weights(obs_tri);

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        // observer triangle edges
        for (uint8_t ii = 0; ii < 3; ++ii)
        {
            result(ii, jj) = tri_quad_.weights().dot(
                ((tri_quad_.points().colwise() - obs_tri.v((ii + 2) % 3)).transpose() *
                    (tri_quad_.points().colwise() - src_tri.v((jj + 2) % 3))).diagonal()
            );
        }
    }

    result.array() *= (
        Rwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    // if (src_tri.normal().dot(obs_tri.normal()) < 0)
    //     result.array() *= -1;

    // if (src_tri.normal().dot(obs_tri.normal()) < 0)
    //     result.array() *= 0;

    return result;

};


template <typename TriangleQuadratureType>
EigMatMN<Complex, 3, 3> RotRwgRwgOp<TriangleQuadratureType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{

    EigMatMN<Complex, 3, 3> result = EigMatMN<Complex, 3, 3>::Zero(3, 3);

    if (GeometryOps<3>::common_vertices(obs_tri, src_tri) < 3)
        return result;

    tri_quad_.compute_points_weights(obs_tri);

    // source triangle edges
    for (uint8_t jj = 0; jj < 3; ++jj)
    {
        // observer triangle edges
        for (uint8_t ii = 0; ii < 3; ++ii)
        {
            result(ii, jj) = -tri_quad_.weights().dot(
                (((tri_quad_.points().colwise() - obs_tri.v((ii + 2) % 3)
                    ).colwise().cross(obs_tri.normal())).transpose() * (
                        (tri_quad_.points().colwise() - src_tri.v((jj + 2) % 3)))).diagonal());
        }
    }

    result.array() *= (
        Rwg::normalization(obs_tri).transpose() * Rwg::normalization(src_tri)
        ).array();

    // if (src_tri.normal().dot(obs_tri.normal()) < 0)
    //     result.array() *= -1;

    return result;

};

}

#endif
