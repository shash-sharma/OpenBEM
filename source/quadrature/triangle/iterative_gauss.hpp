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

#ifndef ITER_GAUSS_TRI_QUAD_H
#define ITER_GAUSS_TRI_QUAD_H

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
* Important: upon construction, this class loads tabulated quadrature data from a json file, and
* stores the tabulated quadrature data for all possible orders up to `TRI_MAX_ORDER`. Therefore, it
* is strongly recommended that objects of this class should be reused within loops, rather than
* creating a new object at each iteration of a long loop.
*/
template <uint8_t dim>
class IterativeGaussTriangleQuadrature: public TriangleQuadratureBase<dim>
{

    using base = TriangleQuadratureBase<dim>;

public:

    /**
    * @brief Sets the quadrature order at which iterations should start.
    * @param[in] starting_order - Starting quadrature order.
    */
    void set_starting_order(const uint8_t starting_order)
    {
        starting_order_ = starting_order;
        converged_ = false;
        return;
    };


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
    {
        tol_ = tol;
        converged_ = false;
        return;
    };


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
    {
        max_iters_ = max_iters;
        converged_ = false;
        return;
    };


    /**
    * @brief Returns the maximum number of iterations allowed even if not converged.
    * @return Maximum allowed iterations.
    */
    uint16_t max_iters() const { return max_iters_; };


    /**
    * @brief Computes and stores the points on which to evaluate the integrand, and the corresponding weights.
    * @param[in] tri - Triangle for quadrature evaluation.
    * @param[in] eval - Function or class with `operator()` that evaluates the integrand.
    */
    void compute_points_weights(
        const Triangle<dim>& tri,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) override;


    /**
    * @brief Checks whether the iterations converged.
    * @return `true` if the iterations converged, `false` otherwise.
    */
    bool converged() const
    {
        if (!base::points_weights_computed_)
            throw std::runtime_error(
                "IterativeGaussTriangleQuadrature::converged(): must call `compute_points_weights()` first.");
        return converged_;
    };


    /**
    * @brief Returns the quadrature order at which the iterations converged.
    * @return Quadrature order at which the iterations converged; 0 if not converged.
    */
    uint8_t converged_order() const
    {
        if (!base::points_weights_computed_)
            throw std::runtime_error(
                "IterativeGaussTriangleQuadrature::converged_order(): must call `compute_points_weights()` first.");
        return converged_order_;
    };


private:

    bool converged_ = false;
    uint8_t starting_order_ = 1;
    uint8_t converged_order_ = 0;
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
