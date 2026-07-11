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

#ifndef BEM_GAUSS_TRI_QUAD_H
#define BEM_GAUSS_TRI_QUAD_H

#include <vector>
#include <string>
#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
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

    using base = QuadratureBase<dim>;

public:

    /**
    * @brief Constructs a `GaussTriangleQuadrature` object with a specified quadrature order.
    * @param[in] order - Quadrature order (optional).
    * @details
    * Important: this constructor loads tabulated quadrature data from a json file, and stores
    * it for all possible orders up to `TRI_MAX_ORDER`. Therefore, it is strongly recommended
    * that objects of this class should be reused within loops, rather than creating a new
    * object at each iteration of a loop.
    */
    GaussTriangleQuadrature(const uint8_t order = TRI_DEFAULT_ORDER): rules_(&get_rules())
    { set_order(order); return; };


    /**
    * @brief Sets the quadrature order.
    * @param[in] order - Quadrature order.
    */
    void set_order(const uint8_t order) override;


    /**
    * @brief Computes quadrature evaluation points and corresponding weights.
    * @param[in] tri - Triangle for quadrature evaluation.
    * @param[in] eval - Function or functor to evaluate the integrand (optional, unused).
    * @return Quadrature points and weights.
    */
    QuadratureData<dim> compute(
        const Triangle<dim>& tri,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) override;


    /**
    * @brief Returns the evaluation points in the reference unit triangle.
    * @return Read-only reference to the reference triangle evaluation points.
    */
    const EigMatNX<Float, 2>& ref_points() const { return (*rules_)[base::order_ - 1].points; };


    /**
    * @brief Returns the weights associated with the evaluation points in the reference unit triangle.
    * @return Read-only reference to the reference triangle weights.
    */
    const EigRowVec<Float>& ref_weights() const { return (*rules_)[base::order_ - 1].weights; };


protected:

    static const std::vector<QuadratureData<2>>& get_rules()
    {
        EigColVec<Index> orders = EigColVec<Index>::LinSpaced(TRI_MAX_ORDER, 1, TRI_MAX_ORDER);
        static const std::vector<QuadratureData<2>> shared_data = QuadratureBase<2>::load_rules(
            rule_file_, orders
            );
        return shared_data;
    }

    inline static const std::string rule_file_ = "tri_quad_xiao_gimbutas.json";
    const std::vector<QuadratureData<2>>* rules_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "quadrature/triangle/gauss.cpp"
#endif

#endif
