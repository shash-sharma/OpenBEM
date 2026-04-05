// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Quadrature over the source triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_QUAD_I
#define BEM_RWG_OPINT_SRC_QUAD_I

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

template <typename TriangleQuadratureType, typename ScalarKernelType>
SrcResult SrcQuadrature<TriangleQuadratureType, ScalarKernelType>::integrate(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs
    )
{

    // Lambda used for iterative or adaptive numerical integration
    auto eval = [&](ConstEigRef<EigMatNX<Float, 2>> r_src) -> EigRowVec<Complex>
    {
        EigMatNX<Float, 3> r_src_3d = EigMatNX<Float, 3>::Zero(3, r_src.cols());
        r_src_3d.topRows(2) = r_src;
        return kernel_.compute(r_obs.rowwise().mean(), r_src_3d, k);
    };

    // Get the quadrature points and weights
    tri_quad_.compute_points_weights(src_tri, eval);

    EigMatNX<Float, 3> points_3d = EigMatNX<Float, 3>::Zero(3, tri_quad_.points().cols());
    points_3d.topRows(2) = tri_quad_.points();

    // Assemble the integration results
    SrcResult result;

    if (base::compute_g_terms_)
    {
        result.g.setZero(1, r_obs.cols());
        result.rs_g.setZero(2, r_obs.cols());

        for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
        {
            EigRowVec<Complex> gw = kernel_.compute(
                r_obs.col(ro), points_3d, k
                ).array() * tri_quad_.weights().array();
            result.g[ro] = gw.array().sum();
            result.rs_g.col(ro) = tri_quad_.points() * gw.transpose();
        }
    }

    if (base::compute_grad_g_terms_)
    {
        result.grad_g.setZero(3, r_obs.cols());

        for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
        {
            result.grad_g.col(ro) = kernel_.compute_grad(
                r_obs.col(ro), points_3d, k
                ) * tri_quad_.weights().transpose();
        }
    }

    return result;

};

}

#endif
