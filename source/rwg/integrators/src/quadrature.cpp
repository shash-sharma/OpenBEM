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

#include "rwg/integrators/src/quadrature.hpp"

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

SrcResult SrcQuadrature::integrate(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs,
    const bool g_terms,
    const bool grad_g_terms
    )
{

    // Lambda used for iterative or adaptive numerical integration
    auto eval = [&](ConstEigRef<EigMatNX<Float, 2>> r_src) -> EigRowVec<Complex>
    {
        EigMatNX<Float, 3> r_src_3d = EigMatNX<Float, 3>::Zero(3, r_src.cols());
        r_src_3d.topRows(2) = r_src;
        return kernel_->compute(r_obs.rowwise().mean(), r_src_3d, k);
    };

    // Get the quadrature points and weights
    QuadratureData<2> qd = tri_quad_->compute(src_tri, eval);

    EigMatNX<Float, 3> points_3d = EigMatNX<Float, 3>::Zero(3, qd.points.cols());
    points_3d.topRows(2) = qd.points;

    // Assemble the integration results
    SrcResult result;

    if (g_terms)
    {
        result.g.setZero(1, r_obs.cols());
        result.rs_g.setZero(2, r_obs.cols());

        for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
        {
            EigRowVec<Complex> gw = kernel_->compute(
                r_obs.col(ro), points_3d, k
                ).array() * qd.weights.array();
            result.g[ro] = gw.array().sum();
            result.rs_g.col(ro) = qd.points * gw.transpose();
        }
    }

    if (grad_g_terms)
    {
        result.grad_g.setZero(3, r_obs.cols());

        for (std::size_t ro = 0; ro < r_obs.cols(); ++ro)
        {
            result.grad_g.col(ro) = kernel_->compute_grad(
                r_obs.col(ro), points_3d, k
                ) * qd.weights.transpose();
        }
    }

    return result;

};

}

