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

#ifndef BEM_GAUSS_LINE_QUAD_H
#define BEM_GAUSS_LINE_QUAD_H

#include <vector>
#include <string>
#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "quadrature/line/base.hpp"


namespace bem
{

/**
* \ingroup linequad
* @{
*/

/**
* @brief Class for Gaussian quadrature over a line segment.
* @tparam dim - Dimension of the line segment (1, 2, or 3).
*/
template <uint8_t dim>
class GaussLineQuadrature: public LineQuadratureBase<dim>
{

    using base = QuadratureBase<dim>;

public:

    /**
    * @brief Constructs a `GaussLineQuadrature` object with a specified quadrature order.
    * @param[in] order - Quadrature order (optional).
    * @details
    * Important: this constructor loads tabulated quadrature data from a json file, and stores
    * it for all possible orders up to `LINE_MAX_ORDER`. Therefore, it is strongly recommended
    * that objects of this class should be reused within loops, rather than creating a new
    * object at each iteration of a loop.
    */
    GaussLineQuadrature(const uint8_t order = LINE_DEFAULT_ORDER)
    {
        EigColVec<Index> orders = EigColVec<Index>::LinSpaced(LINE_MAX_ORDER, 1, LINE_MAX_ORDER);
        rules_ = QuadratureBase<1>::load_rules(rule_file_, orders);
        set_order(order);
        return;
    };


    /**
    * @brief Sets the quadrature order.
    * @param[in] order - Quadrature order.
    */
    void set_order(const uint8_t order) override;


    /**
    * @brief Computes quadrature evaluation points and corresponding weights.
    * @param[in] p1 - First point of the line segment.
    * @param[in] p2 - Second point of the line segment.
    * @param[in] eval - Function functor to evaluate the integrand (optional, unused).
    * @return Quadrature points and weights.
    */
    QuadratureData<dim> compute(
        ConstEigRef<EigColVecN<Float, dim>> p1,
        ConstEigRef<EigColVecN<Float, dim>> p2,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) override;


    /**
    * @brief Returns the evaluation points in the reference unit line segment.
    * @return Read-only reference to the reference segment evaluation points.
    */
    const EigRowVec<Float>& ref_points() const
    { return rules_[base::order_ - 1].points; };


    /**
    * @brief Returns the weights associated with the evaluation points in the reference unit line segment.
    * @return Read-only reference to the reference segment weights.
    */
    const EigRowVec<Float>& ref_weights() const
    { return rules_[base::order_ - 1].weights; };


protected:

    std::string rule_file_ = "line_quad_gauss.json";
    std::vector<QuadratureData<1>> rules_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "quadrature/line/gauss.cpp"
#endif

#endif
