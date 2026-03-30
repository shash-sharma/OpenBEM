// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base classes for BEM operator kernels.
*/

#ifndef BEM_KERNEL_BASE_H
#define BEM_KERNEL_BASE_H

#include <cassert>

#include "types.hpp"
#include "constants.hpp"


namespace bem
{

const Float KERNEL_DEFAULT_TOL = 1.0e-6;

/**
* \addtogroup kern
* @{
*/

/**
* @brief Base class for scalar kernels.
* @tparam dim - Dimension of the kernel (2 or 3).
*/
template <uint8_t dim>
class ScalarKernelBase
{

    static_assert((dim == 2 || dim == 3), "`dim` must be 2 or 3.");

public:

    /**
    * @brief Computes the scalar kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Source position vector.
    * @param[in] k - Complex wavenumber.
    * @return Scalar kernel value.
    */
    virtual Complex kernel(
        ConstEigRef<EigColVecN<Float, dim>> r_obs,
        ConstEigRef<EigColVecN<Float, dim>> r_src,
        const Complex k
        ) const = 0;


    /**
    * @brief Computes the gradient of the scalar kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Source position vector.
    * @param[in] k - Complex wavenumber.
    * @return Components of the gradient of the scalar kernel.
    */
    virtual EigColVecN<Complex, dim> grad_kernel(
        ConstEigRef<EigColVecN<Float, dim>> r_obs,
        ConstEigRef<EigColVecN<Float, dim>> r_src,
        const Complex k
        ) const = 0;

};

/**
* @}
*/

}

#endif
