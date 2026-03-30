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

#ifndef GAUSS_TRI_QUAD_H
#define GAUSS_TRI_QUAD_H

#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/utility.hpp"
#include "quadrature/triangle/base.hpp"


namespace bem
{

/**
* \ingroup triquad
* @{
*/

/**
* @brief Class for Gaussian quadrature over a triangle.
* @tparam dim - Dimension of the triangle (2 or 3).
*/
template <uint8_t dim>
class GaussTriangleQuadrature: public TriangleQuadratureBase<dim>
{

    using base = TriangleQuadratureBase<dim>;

public:

    /**
    * @brief Constructs a `GaussTriangleQuadrature` object with a specified quadrature order.
    * @param[in] order - Quadrature order (optional).
    * @details
    * Important: this constructor loads tabulated quadrature data from a json file, and stores the
    * tabulated quadrature data for all possible orders up to `TRI_MAX_ORDER`. Therefore, it is
    * strongly recommended that objects of this class should be reused within loops, rather than
    * creating a new object at each iteration of a long loop.
    */
    GaussTriangleQuadrature(const uint8_t order = TRI_DEFAULT_ORDER)
    {
        EigColVec<Index> orders = EigColVec<Index>::LinSpaced(TRI_MAX_ORDER, 1, TRI_MAX_ORDER);
        rules_ = load_rules<2> (rule_file_, orders);
        set_order(order);
        return;
    };


    /**
    * @brief Sets the quadrature order.
    * @param[in] order - Quadrature order.
    */
    void set_order(const uint8_t order) override;


    /**
    * @brief Computes and stores the points on which to evaluate the integrand, and the corresponding weights.
    * @param[in] tri - Triangle for quadrature evaluation.
    * @param[in] eval - Function or class with `operator()` that evaluates the integrand (optional, unused).
    */
    void compute_points_weights(
        const Triangle<dim>& tri,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) override;


    /**
    * @brief Returns the evaluation points in the reference unit triangle.
    * @return Read-only reference to the reference triangle evaluation points.
    */
    const EigMatNX<Float, 2>& ref_points() const { return rules_[base::order_ - 1].nodes;};


    /**
    * @brief Returns the weights associated with the evaluation points in the reference unit triangle.
    * @return Read-only reference to the reference triangle weights.
    */
    const EigRowVec<Float>& ref_weights() const { return rules_[base::order_ - 1].weights; };


private:

    std::string rule_file_ = "tri_quad_xiao_gimbutas.json";
    std::vector<QuadratureRule<2>> rules_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "quadrature/triangle/gauss.cpp"
#endif

#endif
