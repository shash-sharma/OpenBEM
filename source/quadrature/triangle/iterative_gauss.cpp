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

#include <stdexcept>
#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem
{

template <uint8_t dim>
QuadratureData<dim> IterativeGaussTriangleQuadrature<dim>::compute(
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

    QuadratureData<dim> qd = gauss_quad_.compute(tri);

    EigRowVec<Complex> vals = eval(qd.points);
    Complex val_ref = qd.weights.dot(vals);

    if (max_iters_ > TRI_MAX_ORDER)
        throw std::runtime_error(
            "IterativeGaussTriangleQuadrature::compute(): max iterations must not exceed " + std::to_string(TRI_MAX_ORDER));

    for (order = starting_order_ + 1; order <= max_iters_; ++order)
    {
        // Order 3 and order 4 have the same rule
        if (order == 4 && starting_order_ != 4)
            continue;

        gauss_quad_.set_order(order);
        qd = gauss_quad_.compute(tri);
        vals = eval(qd.points);
        Complex val = qd.weights.dot(vals);

        bool equal = val_ref == val;
        bool converged = compare_with_tol(val, val_ref, tol_, 1);

        assert((equal || val_ref != zero) &&
               "IterativeGaussTriangleQuadrature::compute(): divide by zero.");

        if (equal || converged)
        {
            qd.converged_iter = order;
            qd.converged = true;
            return qd;
        }
        val_ref = val;
    }

    qd.converged = false;
    return qd;

};


template class IterativeGaussTriangleQuadrature<2>;
template class IterativeGaussTriangleQuadrature<3>;

}
