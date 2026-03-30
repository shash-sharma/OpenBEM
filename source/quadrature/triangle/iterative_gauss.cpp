// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Iterative Gaussian quadrature over a triangle.
*/

#include "quadrature/triangle/iterative_gauss.hpp"

#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/utility.hpp"


namespace bem
{

template <uint8_t dim>
void IterativeGaussTriangleQuadrature<dim>::compute_points_weights(
    const Triangle<dim>& tri,
    std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval
    )
{
    if (!eval)
        throw std::invalid_argument(
            "IterativeGaussTriangleQuadrature::compute_points_weights(): invalid or missing eval."
            );

    uint8_t order = starting_order_;
    gauss_quad_.set_order(order);
    gauss_quad_.compute_points_weights(tri);

    EigRowVec<Complex> vals = eval(gauss_quad_.points());
    Complex val_ref = gauss_quad_.weights().dot(vals);

    if (max_iters_ > TRI_MAX_ORDER)
        throw std::runtime_error(
            "IterativeGaussTriangleQuadrature::compute_points_weights(): max iterations must not exceed " + std::to_string(TRI_MAX_ORDER));

    for (order = starting_order_ + 1; order <= max_iters_; ++order)
    {
        // Order 3 and order 4 have the same rule
        if (order == 4 && starting_order_ != 4)
            continue;

        gauss_quad_.set_order(order);
        gauss_quad_.compute_points_weights(tri);
        vals = eval(gauss_quad_.points());
        Complex val = gauss_quad_.weights().dot(vals);

        bool equal = val_ref == val;
        bool converged = compare_with_tol(val, val_ref, tol_, 1);

        assert((equal || val_ref != zero) &&
               "IterativeGaussTriangleQuadrature::compute_points_weights(): divide by zero.");

        if (equal || converged)
        {
            base::points_ = gauss_quad_.points();
            base::weights_ = gauss_quad_.weights();
            base::points_weights_computed_ = true;
            converged_order_ = order;
            converged_ = true;
            return;
        }
        val_ref = val;
    }

    base::points_ = gauss_quad_.points();
    base::weights_ = gauss_quad_.weights();
    base::points_weights_computed_ = true;
    converged_ = false;

    return;
};


template class IterativeGaussTriangleQuadrature<2>;
template class IterativeGaussTriangleQuadrature<3>;

}
