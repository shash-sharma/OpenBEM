// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling excitation vectors for RWG-based BEM systems.
*/

#ifndef BEM_RWG_EXC_ASSEMBLER_H
#define BEM_RWG_EXC_ASSEMBLER_H

#include "types.hpp"
#include "rwg/assemblers/base.hpp"


namespace bem
{
// Forward declarations
template <typename T> class MatrixBase;
}


namespace bem::rwg
{

// Forward declarations
template <uint8_t obs_num_dof> class ExcitationBase;

/**
* \addtogroup assm
* @{
*/

/**
* @brief Class for generating excitation matrices for RWG-based BEM systems with RWG testing functions.
*/
class EdgeExcitationAssembler: public ExcitationAssemblerBase<3>
{

    using base = ExcitationAssemblerBase<3>;
    using base::base;

public:

    /**
    * @brief Assembles the excitation matrix for a given excitation object and observation triangle mesh.
    * @param[out] mat - Matrix to store the assembled excitation coefficients, with columns corresponding
    * to each right-hand side, and rows corresponding to observation mesh edges.
    * @param[in] exc - Excitation object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        ExcitationBase<3>& exc,
        const Complex k
        ) override;

};


/**
* @brief Class for generating excitation matrices for RWG-based BEM systems with pulse testing functions.
*/
class FaceExcitationAssembler: public ExcitationAssemblerBase<1>
{

    using base = ExcitationAssemblerBase<1>;
    using base::base;

public:

    /**
    * @brief Assembles the excitation matrix for a given excitation object and observation triangle mesh.
    * @param[out] mat - Matrix to store the assembled excitation coefficients, with columns corresponding
    * to each right-hand side, and rows corresponding to observation mesh faces.
    * @param[in] exc - Excitation object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        ExcitationBase<1>& exc,
        const Complex k
        ) override;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/excitation_matrix.cpp"
#endif

#endif
