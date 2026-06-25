// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for quadrature over a line segment.
*/

#ifndef LINE_QUAD_BASE_H
#define LINE_QUAD_BASE_H

#include <stdexcept>
#include <string>
#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "quadrature/base.hpp"


namespace bem
{

const uint8_t LINE_DEFAULT_ORDER = 10;
const uint8_t LINE_MAX_ORDER = 30;

/**
* \ingroup linequad
* @{
*/

/**
* @brief Base class for quadrature over a line segment.
* @tparam dim - Dimension of the line segment (1, 2, or 3).
*/
template <uint8_t dim>
class LineQuadratureBase: public QuadratureBase<dim>
{

    static_assert((dim == 1 || dim == 2 || dim == 3), "`dim` must be 1, 2, or 3.");

public:

    /**
    * @brief Computes quadrature evaluation points and corresponding weights.
    * @param[in] p1 - First point of the line segment.
    * @param[in] p2 - Second point of the line segment.
    * @param[in] eval - Function or functor to evaluate the integrand (optional).
    * @return Quadrature points and weights.
    */
    virtual QuadratureData<dim> compute(
        ConstEigRef<EigColVecN<Float, dim>> p1,
        ConstEigRef<EigColVecN<Float, dim>> p2,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~LineQuadratureBase() = default;

};

/**
* @}
*/

}

#endif
