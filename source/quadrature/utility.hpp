// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Utility functions for quadrature.
*/

#ifndef QUAD_UTILITY_H
#define QUAD_UTILITY_H

#include <vector>
#include <string>

#include "types.hpp"

#ifndef BEM_LINKED
#define BEM_QUAD_UTIL_INLINE inline
#else
#define BEM_QUAD_UTIL_INLINE
#endif


namespace bem
{

/**
* \defgroup quadutil Utility
* \ingroup quad
* @brief Utility and helper functions for quadrature routines.
* @{
*/

/**
* @brief Compares two complex numbers within a given tolerance based on a given rule.
* @param[in] val - Value to check.
* @param[in] val_ref - Reference value.
* @param[in] tol - Tolerance (optional).
* @param[in] mode - Comparison mode:
*   - 1: Compare real and imaginary parts separately (default).
*   - 2: Compare magnitude and phase separately.
*   - 3: Compare the absolute value.
* @return `true` if the two values are within the specified tolerance, `false` otherwise.
*/
BEM_QUAD_UTIL_INLINE
bool compare_with_tol(
    const Complex val,
    const Complex val_ref,
    const Float tol = 1e-3,
    const uint8_t mode = 1
    );


/**
* @brief Data structure defining a quadrature rule.
* @tparam dim - Dimension in which the quadrature rule is defined.
*/
template <uint8_t dim>
struct QuadratureRule
{
    Index num_nodes;
    EigMatNX<Float, dim> nodes;
    EigRowVec<Float> weights;
};


/**
* @brief Loads quadrature rules from the specified json file assumed to be located in the same
* directory as this file.
* @tparam dim - Dimension in which the quadrature rules are defined.
* @param[in] file - json file name.
* @param[in] orders - List of orders to load.
* @return Vector of quadrature rule objects.
*/
template <uint8_t dim>
std::vector<QuadratureRule<dim>> load_rules(
    const std::string file,
    ConstEigRef<EigColVec<Index>> orders
    );

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "quadrature/utility.cpp"
#endif

#endif

