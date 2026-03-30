// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Gaussian quadrature over a line segment.
*/

#include "quadrature/line/gauss.hpp"

#include <functional>

#include "types.hpp"


namespace bem
{

template <uint8_t dim>
void GaussLineQuadrature<dim>::set_order(const uint8_t order)
{
    if (order > LINE_MAX_ORDER)
        throw std::domain_error(
            "GaussLineQuadrature::set_order(): order must be less than or equal to "
            + std::to_string(LINE_MAX_ORDER) + "."
            );

    base::order_ = order;
    base::points_.resize(dim, rules_[order - 1].num_nodes);
    base::weights_.resize(1, rules_[order - 1].num_nodes);
    base::points_weights_computed_ = false;

    return;
};


template <uint8_t dim>
void GaussLineQuadrature<dim>::compute_points_weights(
    ConstEigRef<EigColVecN<Float, dim>> p1,
    ConstEigRef<EigColVecN<Float, dim>> p2,
    std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval
    )
{
    EigColVecN<Float, dim> p = p2 - p1;
    Float p_len = p.norm();

    base::points_.noalias() = (p * ref_points()).colwise() + p1;
    base::weights_.noalias() = ref_weights() * p_len;
    base::points_weights_computed_ = true;

    return;
};


template class GaussLineQuadrature<1>;
template class GaussLineQuadrature<2>;
template class GaussLineQuadrature<3>;

}
