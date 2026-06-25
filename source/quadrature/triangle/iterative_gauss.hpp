// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Iterative Gaussian quadrature over a triangle.
*/

#ifndef BEM_ITER_GAUSS_TRI_QUAD_H
#define BEM_ITER_GAUSS_TRI_QUAD_H

#include <functional>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/triangle/base.hpp"
#include "quadrature/triangle/gauss.hpp"


namespace bem
{

const Float ITER_GAUSS_TRI_DEFAULT_TOL = 1e-3;

/**
* \ingroup triquad
* @{
*/

/**
* @brief Class for iterative Gaussian quadrature over a triangle.
* @tparam dim - Dimension of the triangle (2 or 3)
* @details
* Important: upon construction, this class loads tabulated quadrature data from a json file,
* and stores it for all possible orders up to `TRI_MAX_ORDER`. Therefore, it is strongly
* recommended that objects of this class should be reused within loops, rather than creating
* a new object at each iteration of a loop.
*/
template <uint8_t dim>
class IterativeGaussTriangleQuadrature: public TriangleQuadratureBase<dim>
{

    using base = QuadratureBase<dim>;

public:

    /**
    * @brief Sets the quadrature order at which iterations should start.
    * @param[in] starting_order - Starting quadrature order.
    */
    void set_starting_order(const uint8_t starting_order)
    { starting_order_ = starting_order; return; };


    /**
    * @brief Returns the quadrature order at which iterations start.
    * @return Starting quadrature order.
    */
    uint8_t starting_order() const { return starting_order_; };


    /**
    * @brief Sets the relative convergence tolerance defining when iterations should stop.
    * @param[in] tol - Tolerance.
    */
    void set_tol(const Float tol)
    { tol_ = tol; return; };


    /**
    * @brief Returns the relative convergence tolerance defining when iterations should stop.
    * @return Tolerance.
    */
    Float tol() const { return tol_; };


    /**
    * @brief Sets the maximum number of iterations allowed even if not converged.
    * @param[in] max_iters - Maximum allowed iterations.
    */
    void set_max_iters(const uint16_t max_iters)
    { max_iters_ = max_iters; return; };


    /**
    * @brief Returns the maximum number of iterations allowed even if not converged.
    * @return Maximum allowed iterations.
    */
    uint16_t max_iters() const { return max_iters_; };


    /**
    * @brief Computes quadrature evaluation points and corresponding weights.
    * @param[in] tri - Triangle for quadrature evaluation.
    * @param[in] eval - Function or functor to evaluate the integrand.
    * @return Quadrature points and weights.
    */
    QuadratureData<dim> compute(
        const Triangle<dim>& tri,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) override;


protected:

    uint8_t starting_order_ = 1;
    uint16_t max_iters_ = TRI_MAX_ORDER;
    Float tol_ = ITER_GAUSS_TRI_DEFAULT_TOL;
    GaussTriangleQuadrature<dim> gauss_quad_ = GaussTriangleQuadrature<dim> (starting_order_);

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "quadrature/triangle/iterative_gauss.cpp"
#endif

#endif
