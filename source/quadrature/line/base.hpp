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
class LineQuadratureBase
{

    static_assert((dim == 1 || dim == 2 || dim == 3), "`dim` must be 1, 2, or 3.");

public:

    /**
    * @brief Computes and stores the points on which to evaluate the integrand, and the corresponding weights.
    * @param[in] p1 - First point of the line segment.
    * @param[in] p2 - Second point of the line segment.
    * @param[in] eval - Function or class with `operator()` that evaluates the integrand (optional).
    */
    virtual void compute_points_weights(
        ConstEigRef<EigColVecN<Float, dim>> p1,
        ConstEigRef<EigColVecN<Float, dim>> p2,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) = 0;


    /**
    * @brief Sets the quadrature order.
    * @param[in] order - Quadrature order.
    */
    virtual void set_order(const uint8_t order) { order_ = order; return; };


    /**
    * @brief Returns the quadrature order.
    * @return Quadrature order.
    */
    uint8_t order() const { return order_; };


    /**
    * @brief Returns the points on which to evaluate the integrand.
    * @returns Read-only reference to the evaluation points.
    */
    const EigMatNX<Float, dim>& points() const
    {
        if (!points_weights_computed_)
            throw std::runtime_error(
                "LineQuadratureBase::points(): must call `compute_points_weights()` first.");
        return points_;
    };


    /**
    * @brief Returns the weights associated with the points on which the integrand is evaluated.
    * @returns Read-only reference to the weights.
    */
    const EigRowVec<Float>& weights() const
    {
        if (!points_weights_computed_)
            throw std::runtime_error(
                "LineQuadratureBase::weights(): must call `compute_points_weights()` first.");
        return weights_;
    };


protected:

    EigMatNX<Float, dim> points_;
    EigRowVec<Float> weights_;

    uint8_t order_ = LINE_DEFAULT_ORDER;
    bool points_weights_computed_ = false;

};

/**
* @}
*/

}

#endif
