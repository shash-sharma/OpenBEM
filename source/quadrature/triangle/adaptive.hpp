// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Adaptive quadrature over a triangle.
*/

#ifndef BEM_ADAPTIVE_TRI_QUAD_H
#define BEM_ADAPTIVE_TRI_QUAD_H

#include <cassert>
#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/triangle/base.hpp"


namespace bem
{

const uint16_t ADAPTIVE_TRI_DEFAULT_MAX_LEVELS = 200;
const Float ADAPTIVE_TRI_DEFAULT_TOL = 1e-3;

/**
* \ingroup triquad
* @{
*/

/**
* @brief Class for adaptive quadrature over a triangle.
* Reference: O. Ergul, L. Gurel, "The Multilevel Fast Multipole Algorithm (MLFMA) for Solving
* Large-Scale Computational Electromagnetics Problems," book, Wiley-IEEE Press, 2014.
* @tparam dim - Dimension of the triangle (2 or 3).
*/
template <uint8_t dim>
class AdaptiveTriangleQuadrature: public TriangleQuadratureBase<dim>
{

    using base = QuadratureBase<dim>;

public:

    /**
    * @brief Sets the maximum number of recursion levels allowed even if not converged.
    * @param[in] max_levels - Maximum allowed recursion levels.
    */
    void set_max_levels(const uint16_t max_levels)
    { max_levels_ = max_levels; return; };


    /**
    * @brief Returns the maximum number of recursion levels allowed even if not converged.
    * @return Maximum allowed recursion levels.
    */
    uint16_t max_levels() const { return max_levels_; };


    /**
    * @brief Sets the relative convergence tolerance defining when recursion should stop.
    * @param[in] tol - Tolerance.
    */
    void set_tol(const Float tol)
    { tol_ = tol; return; };


    /**
    * @brief Returns the relative convergence tolerance defining when recursion should stop.
    * @return Tolerance.
    */
    Float tol() const { return tol_; };


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


private:

    /**
    * @brief Executes a five-point integration rule along the edges of a given triangle.
    * @param[in, out] qd - Quadrature data.
    * @param[in, out] level - Previous recursion level, gets incremented to the current level.
    * @param[in] tri - Triangle for integration.
    * @param[in] eval - Function or class with `operator()` that evaluates the integrand.
    * @param[in] longest_edge - Index of the longest edge of the triangle.
    * @return Integration result.
    */
    Complex run_recursion(
        QuadratureData<dim>& qd,
        int& level,
        const Triangle<dim>& tri,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval,
        const uint8_t longest_edge
        );


    /**
    * @brief Computes the points for a five-point integration rule along the edges of a given triangle.
    * @param[out] p - Five-point integration points.
    * @param[in] tri - Triangle for integration.
    * @param[in] longest_edge - Index of the longest edge of the triangle.
    */
    static void get_5_points(
        EigMatMN<Float, dim, 5>& p,
        const Triangle<dim>& tri,
        const uint8_t longest_edge
        );


    /**
    * @brief Extracts two subtriangles using the bisector of the longest edge of the given triangle.
    * @param[out] v_subtri1 - vertices for the first sub-`Triangle`.
    * @param[out] v_subtri2 - vertices for the second sub-`Triangle`.
    * @param[in] tri - Triangle for integration.
    * @param[in] longest_edge - Index of the longest edge of the given triangle.
    */
    void get_subtris(
        EigMatMN<Float, dim, 3>& v_subtri1,
        EigMatMN<Float, dim, 3>& v_subtri2,
        const Triangle<dim>& tri,
        const uint8_t longest_edge
        );


    /**
    * @brief Checks for convergence given two successive adaptive integration results.
    * @param[in] val_coarse - Computed integral for the coarser set of points.
    * @param[in] val_fine - Computed integral for the finer set of points.
    * @return `true` if the two values are within the specified tolerance, `false` otherwise.
    */
    bool check_convergence(const Complex val_coarse, const Complex val_fine)
    {
        bool converged = compare_with_tol(val_fine, val_coarse, tol_, 3);

        assert((converged || val_fine != zero) &&
               "AdaptiveTriangleQuadrature::check_convergence(): divide by zero.");
        // converged = converged || check_convergence(val_fine);
        // check_convergence(val_fine);
        // std::cout << val_fine << ", " << val_coarse << ", " << std::endl;
        return converged;
    }


    /**
    * @brief Returns the total number of recursions that were executed.
    * @return Total number of recursions executed.
    */
    uint16_t total_recursions()
    { return total_recursions_; };


    /**
    * @brief Resets the counter for the total number of recursions executed.
    */
    void reset_recursion_count()
    { total_recursions_ = 0; return; };


    uint16_t max_levels_ = ADAPTIVE_TRI_DEFAULT_MAX_LEVELS;
    Float tol_ = ADAPTIVE_TRI_DEFAULT_TOL;

    const EigColVecN<Float, 10> weights10_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    const EigColVecN<Float, 5> weights5_ = {1.0, 1.0, 1.0, 1.0, 2.0};

    EigMatMN<Float, dim, 5> p_main_;
    EigMatMN<Float, dim, 6> p_sub_;
    EigRowVecN<Complex, 5> vals_main_, vals_subtri1_, vals_subtri2_;
    EigRowVecN<Complex, 6> vals_sub_;
    EigRowVec<Complex> vals_;

    EigRowVecN<uint32_t, 2> idx_extra1_, idx_extra2_;
    Complex I_prev_ = 0.0;
    uint16_t total_recursions_ = 0;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "quadrature/triangle/adaptive.cpp"
#endif

#endif
