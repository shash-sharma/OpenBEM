// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for quadrature over a triangle.
*/

#ifndef BEM_TRI_QUAD_BASE_H
#define BEM_TRI_QUAD_BASE_H

#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "quadrature/base.hpp"


namespace bem
{

const uint8_t TRI_DEFAULT_ORDER = 4;
const uint8_t TRI_MAX_ORDER = 30;

/**
* \ingroup triquad
* @{
*/

/**
* @brief Base class for quadrature over a triangle.
* @tparam dim - Dimension of the triangle (2 or 3).
*/
template <uint8_t dim>
class TriangleQuadratureBase: public QuadratureBase<dim>
{

    static_assert((dim == 2 || dim == 3), "`dim` must be 2 or 3.");

public:

    /**
    * @brief Computes quadrature evaluation points and corresponding weights.
    * @param[in] tri - Triangle for quadrature evaluation.
    * @param[in] eval - Function or functor to evaluate the integrand (optional).
    * @return Quadrature points and weights.
    */
    virtual QuadratureData<dim> compute(
        const Triangle<dim>& tri,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~TriangleQuadratureBase() = default;

};

/**
* @}
*/

}

#endif
