// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Trapezoidal integration over a line segment.
*/

#include "quadrature/line/trapz.hpp"

#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "quadrature/utility.hpp"


namespace bem
{

template <uint8_t dim>
void TrapzLineQuadrature<dim>::set_num_segments(const uint16_t num_segments)
{
    if (num_segments > TRAPZ_LINE_MAX_NUM_SEGMENTS)
        throw std::domain_error(
            "TrapzLineQuadrature::set_num_segments(): max number of segments must be less than "
            + std::to_string(TRAPZ_LINE_MAX_NUM_SEGMENTS) + ".");

    num_segments_ = num_segments;

    base::points_.resize(dim, num_segments_ + 1);
    base::weights_.resize(1, num_segments_ + 1);
    base::points_weights_computed_ = false;
    return;
};


template <uint8_t dim>
void TrapzLineQuadrature<dim>::compute_points_weights(
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


template class TrapzLineQuadrature<1>;
template class TrapzLineQuadrature<2>;
template class TrapzLineQuadrature<3>;

}
