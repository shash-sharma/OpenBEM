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

#ifndef BEM_ITER_TRAPZ_LINE_QUAD_H
#define BEM_ITER_TRAPZ_LINE_QUAD_H

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
    void set_starting_num_segments(const uint16_t starting_num_segments);


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
    { max_iters_ = max_iters; return;
    };


    /**
    * @brief Returns the maximum number of iterations allowed even if not converged.
    * @return Maximum allowed iterations.
    */
    uint16_t max_iters() const { return max_iters_; };


    /**
    * @brief Computes quadrature evaluation points and corresponding weights.
    * @param[in] p1 - First point of the line segment.
    * @param[in] p2 - Second point of the line segment.
    * @param[in] eval - Function or functor to evaluate the integrand.
    * @return Quadrature points and weights.
    */
    QuadratureData<dim> compute(
        ConstEigRef<EigColVecN<Float, dim>> p1,
        ConstEigRef<EigColVecN<Float, dim>> p2,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval = {}
        ) override;


protected:

    uint16_t starting_num_segments_ = 1;
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
