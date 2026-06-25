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


namespace bem
{

template <uint8_t dim>
QuadratureData<dim> IterativeGaussLineQuadrature<dim>::compute(
    ConstEigRef<EigColVecN<Float, dim>> p1,
    ConstEigRef<EigColVecN<Float, dim>> p2,
    std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval
    )
{

    if (!eval)
        throw std::invalid_argument(
            "IterativeGaussLineQuadrature::compute(): invalid or missing eval."
            );

    QuadratureData<dim> qd;

    if ((p2 - p1).norm() == 0.0)
    {
        qd.points = p1;
        qd.weights = EigRowVec<Float>::Zero(1, 1);
        return qd;
    }

    uint8_t order = starting_order_;
    qd = gauss_quad_.compute(p1, p2);

    EigRowVec<Complex> vals = eval(qd.points);
    Complex val_ref = qd.weights.dot(vals);

    if (max_iters_ > LINE_MAX_ORDER)
        throw std::domain_error(
            "IterativeGaussLineQuadrature::compute(): max iterations must not exceed " + std::to_string(LINE_MAX_ORDER));

    for (order = starting_order_ + 1; order <= max_iters_; order++)
    {
        gauss_quad_.set_order(order);
        qd = gauss_quad_.compute(p1, p2);
        vals = eval(qd.points);
        Complex val = qd.weights.dot(vals);

        if (compare_with_tol(val, val_ref, tol_, 1) == true)
        {
            // converged_ = true;
            // converged_order_ = order;
            return qd;
        }

        val_ref = val;
    }

    // converged_ = false;
    return qd;

};


template class IterativeGaussLineQuadrature<1>;
template class IterativeGaussLineQuadrature<2>;
template class IterativeGaussLineQuadrature<3>;

}
