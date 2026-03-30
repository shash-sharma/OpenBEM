// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Iterative Gaussian quadrature over a line segment.
*/

#include "quadrature/line/iterative_gauss.hpp"

#include <stdexcept>
#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "quadrature/utility.hpp"


namespace bem
{

template <uint8_t dim>
void IterativeGaussLineQuadrature<dim>::compute_points_weights(
    ConstEigRef<EigColVecN<Float, dim>> p1,
    ConstEigRef<EigColVecN<Float, dim>> p2,
    std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval
    )
{
    if (!eval)
        throw std::invalid_argument(
            "IterativeGaussLineQuadrature::compute_points_weights(): invalid or missing eval."
            );

    if ((p2 - p1).norm() == 0.0)
    {
        base::points_ = p1;
        base::weights_ = EigRowVec<Float>::Zero(1, 1);
        base::points_weights_computed_ = true;
        converged_ = true;
        return;
    }

    uint8_t order = starting_order_;
    gauss_quad_.compute_points_weights(p1, p2);

    EigRowVec<Complex> vals = eval(gauss_quad_.points());
    Complex val_ref = gauss_quad_.weights().dot(vals);

    if (max_iters_ > LINE_MAX_ORDER)
        throw std::domain_error(
            "IterativeGaussLineQuadrature::compute_points_weights(): max iterations must not exceed " + std::to_string(LINE_MAX_ORDER));

    for (order = starting_order_ + 1; order <= max_iters_; order++)
    {
        gauss_quad_.set_order(order);
        gauss_quad_.compute_points_weights(p1, p2);
        vals = eval(gauss_quad_.points());
        Complex val = gauss_quad_.weights().dot(vals);

        if (compare_with_tol(val, val_ref, tol_, 1) == true)
        {
            base::points_ = gauss_quad_.points();
            base::weights_ = gauss_quad_.weights();
            base::points_weights_computed_ = true;
            converged_ = true;
            converged_order_ = order;
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


template class IterativeGaussLineQuadrature<1>;
template class IterativeGaussLineQuadrature<2>;
template class IterativeGaussLineQuadrature<3>;

}
