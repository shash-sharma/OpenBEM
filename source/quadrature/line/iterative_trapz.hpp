// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Iterative trapezoidal integration over a line segment.
*/

#ifndef ITER_TRAPZ_LINE_QUAD_H
#define ITER_TRAPZ_LINE_QUAD_H

#include <functional>
#include <stdexcept>

#include "types.hpp"
#include "quadrature/line/base.hpp"
#include "quadrature/line/trapz.hpp"


namespace bem
{

const Float ITER_TRAPZ_LINE_DEFAULT_TOL = 1e-3;

/**
* \ingroup linequad
* @{
*/

/**
* @brief Class for iterative trapezoidal integration over a line segment.
* @tparam dim - Dimension of the line segment (1, 2, or 3).
*/
template <uint8_t dim>
class IterativeTrapzLineQuadrature: public LineQuadratureBase<dim>
{

    using base = LineQuadratureBase<dim>;

public:

    /**
    * @brief Sets the initial number of sub-segments into which the given line segment is divided.
    * @param[in] starting_num_segments - Number of sub-segments with which to start the iterations.
    */
    void set_starting_num_segments(const uint16_t starting_num_segments)
    {
        if (starting_num_segments > TRAPZ_LINE_MAX_NUM_SEGMENTS)
            throw std::domain_error(
                std::string("IterativeTrapzLineQuadrature::set_starting_num_segments(): number of ") +
                std::string("segments must be less than ") +
                std::to_string(TRAPZ_LINE_MAX_NUM_SEGMENTS) + ".");

        starting_num_segments_ = starting_num_segments;
        base::points_weights_computed_ = false;
        converged_ = false;
        return;
    };


    /**
    * @brief Returns the initial number of sub-segments into which the given line segment is divided.
    * @return Number of sub-segments with which to start the iterations.
    */
    uint16_t starting_num_segments() const { return starting_num_segments_; };


    /**
    * @brief Sets the relative convergence tolerance defining when iterations should stop.
    * @param[in] tol - Tolerance.
    */
    void set_tol(const Float tol)
    {
        tol_ = tol;
        base::points_weights_computed_ = false;
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
        base::points_weights_computed_ = false;
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
    * @param[in] p1 - First point of the line segment.
    * @param[in] p2 - Second point of the line segment.
    * @param[in] eval - Function or class with `operator()` that evaluates the integrand.
    */
    void compute_points_weights(
        ConstEigRef<EigColVecN<Float, dim>> p1,
        ConstEigRef<EigColVecN<Float, dim>> p2,
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
                "IterativeTrapzLineQuadrature::converged(): must call `compute_points_weights()` first.");
        return converged_;
    };


    /**
    * @brief Returns the number of sub-segments for which the iterations converged.
    * @return Number of sub-segments for which the iterations converged; 0 if not converged.
    */
    uint16_t converged_num_segments() const { return converged_num_segments_; };


private:

    bool converged_ = false;
    uint16_t starting_num_segments_ = 1;
    uint16_t converged_num_segments_ = 0;
    uint16_t max_iters_ = LINE_MAX_ORDER;
    Float tol_ = ITER_TRAPZ_LINE_DEFAULT_TOL;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "quadrature/line/iterative_trapz.cpp"
#endif

#endif
