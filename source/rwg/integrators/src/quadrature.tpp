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
        EigRowVec<Complex> vals = EigRowVec<Complex>::Zero(1, r_src.cols());
        for (std::size_t rs = 0; rs < r_src.cols(); ++rs)
            vals[rs] = kernel_.kernel(r_obs.rowwise().mean(), r_src_3d.col(rs), k);
        return vals;
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

        for (std::size_t rs = 0; rs < points_3d.cols(); ++rs)
        {
            for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
            {
                Complex g = kernel_.kernel(
                    r_obs.col(ro), points_3d.col(rs), k
                    ) * tri_quad_.weights()[rs];

                result.g[ro] += g;
                for (uint8_t ii = 0; ii < 2; ii++)
                    result.rs_g(ii, ro) += g * points_3d(ii, rs);
            }
        }
    }

    if (base::compute_grad_g_terms_)
    {
        result.grad_g.setZero(3, r_obs.cols());

        for (std::size_t rs = 0; rs < points_3d.cols(); ++rs)
        {
            for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
            {
                EigColVecN<Complex, 3> grad_g = kernel_.grad_kernel(
                    r_obs.col(ro), points_3d.col(rs), k
                    ) * tri_quad_.weights()[rs];

                for (uint8_t ii = 0; ii < 3; ii++)
                    result.grad_g(ii, ro) += grad_g[ii];
            }
        }
    }


    return result;

};

}

#endif
