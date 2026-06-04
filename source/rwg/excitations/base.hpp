// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for computing excitation vector coefficients for RWG-based BEM systems.
*/

#ifndef BEM_RWG_EXC_BASE_H
#define BEM_RWG_EXC_BASE_H

#include "types.hpp"
#include "rwg/function_space.hpp"


namespace bem
{
// Forward declarations
template <uint8_t dim> class Triangle;
}


namespace bem::rwg
{

/**
* \addtogroup rwgexc
* @{
*/

/**
* @brief Base class for generating excitation coefficients for RWG-based BEM systems.
* @tparam TestSpace - Testing function space.
*/
template <typename TestSpace>
class ExcitationBase
{

    static_assert(
        std::is_base_of<FunctionSpaceBase<
            TestSpace, TestSpace::dof, TestSpace::dim
            >, TestSpace>::value,
        "ExcitationBase: `TestSpace` must derive from `FunctionSpaceBase`"
        );

public:

    using TestSpaceType = TestSpace;


    /**
    * @brief Computes the excitation coefficients for each degree of freedom associated with
    * the observation triangle.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the local coordinate system of `src_tri`.
    * @return Excitation coefficient matrix, where each row corresponds to each degree of freedom
    * associated with `obs_tri`, and each column corresponds to each excitation when there is more
    * than one excitation (i.e., more than one right-hand side). The number of excitations and various
    * other settings should be provided via the inheriting class's methods.
    */
    virtual EigMatNX<Complex, TestSpace::dof> compute(
        const Complex k,
        const Triangle<3>& obs_tri
        ) = 0;


    /**
    * @brief Returns the number of excitations (right-hand sides) to be generated.
    * @return Number of excitations (right-hand sides).
    */
    virtual Index num_excitations() const = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~ExcitationBase() = default;

};

/**
* @}
*/

}

#endif
