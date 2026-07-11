// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Gaussian quadrature over a triangle.
*/

#include "quadrature/triangle/gauss.hpp"

#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem
{

template <uint8_t dim>
void GaussTriangleQuadrature<dim>::set_order(const uint8_t order)
{
    if (order > TRI_MAX_ORDER)
        throw std::domain_error(
            "GaussTriangleQuadrature::set_order(): order must be less than or equal to "
            + std::to_string(TRI_MAX_ORDER) + "."
            );
    base::order_ = order;
    return;
};


template <uint8_t dim>
QuadratureData<dim> GaussTriangleQuadrature<dim>::compute(
    const Triangle<dim>& tri,
    std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval
    )
{
    QuadratureData<dim> qd;

    qd.points =
        tri.v().rightCols(2) * ref_points() + tri.v().col(0) * (
            one
            - ref_points().row(0).array()
            - ref_points().row(1).array()
            ).matrix();
    qd.weights = tri.area() * ref_weights() * 2;

    return qd;
};


template class GaussTriangleQuadrature<2>;
template class GaussTriangleQuadrature<3>;

}
