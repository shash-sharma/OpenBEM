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

#ifndef BEM_TRAPZ_LINE_QUAD_H
#define BEM_TRAPZ_LINE_QUAD_H

#include <functional>

#include "types.hpp"
#include "quadrature/line/base.hpp"


namespace bem
{

const uint16_t TRAPZ_LINE_DEFAULT_NUM_SEGMENTS = 10;
const uint16_t TRAPZ_LINE_MAX_NUM_SEGMENTS = 1000;

/**
* \ingroup linequad
* @{
*/

/**
* @brief Class for trapezoidal integration over a line segment.
* @tparam dim - Dimension of the line segment (1, 2, or 3).
*/
template <uint8_t dim>
class TrapzLineQuadrature: public LineQuadratureBase<dim>
{

    using base = LineQuadratureBase<dim>;

public:

    /**
    * @brief Constructs a `TrapzLineQuadrature` object with a specified number of subdivisions.
    * @param[in] num_segments - Number of sub-segments into which the given line segment is divided (optional).
    */
    TrapzLineQuadrature(const uint8_t num_segments = TRAPZ_LINE_DEFAULT_NUM_SEGMENTS)
    {
        set_num_segments(num_segments);
        return;
    };


    /**
    * @brief Sets the number of sub-segments into which the given line segment is divided.
    * @param[in] num_segments - Number of sub-segments.
    */
    void set_num_segments(const uint16_t num_segments);


    /**
    * @brief Returns the number of sub-segments into which the given line segment is divided.
    * @return Number of sub-segments.
    */
    uint16_t num_segments() const { return num_segments_; };


    /**
    * @brief Computes quadrature evaluation points and corresponding weights.
    * @param[in] p1 - First point of the line segment.
    * @param[in] p2 - Second point of the line segment.
    * @param[in] eval - Function or functor to evaluate the integrand (optional, unused).
    * @return Quadrature points and weights.
    */
    QuadratureData<dim> compute(
        ConstEigRef<EigColVecN<Float, dim>> p1,
        ConstEigRef<EigColVecN<Float, dim>> p2,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) override;


    /**
    * @brief Returns the evaluation points in the reference unit line segment.
    * @return Reference segment evaluation points.
    */
    EigRowVec<Float> ref_points() const
    { return EigRowVec<Float>::LinSpaced(num_segments_ + 1, 0.0, 1.0); };


    /**
    * @brief Returns the weights associated with the evaluation points in the reference unit line segment.
    * @return Reference segment weights.
    */
    EigRowVec<Float> ref_weights() const
    {
        EigRowVec<Float> weights = EigRowVec<Float>::Constant(
            1, num_segments_ + 1, 1.0 / num_segments_
            );
        weights[0] = 0.5 / num_segments_;
        weights[num_segments_] = 0.5 / num_segments_;
        return weights;
    };


protected:

    uint16_t num_segments_ = TRAPZ_LINE_DEFAULT_NUM_SEGMENTS;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "quadrature/line/trapz.cpp"
#endif

#endif
