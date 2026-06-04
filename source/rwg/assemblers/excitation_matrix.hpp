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
* @brief Class for generating excitation matrices for RWG-based systems.
* @tparam TestSpace - Testing function space.
*/
template <typename TestSpace>
class ExcitationAssembler: public ExcitationAssemblerBase<TestSpace>
{

    using base = ExcitationAssemblerBase<TestSpace>;

public:

    /**
    * @brief Constructs an `ExcitationAssembler` for a given mesh on given test elements.
    * @param[in] mesh - Triangle mesh for which the excitation matrix is to be assembled.
    * @param[in] elems - Triangle index pairs on which to test the incident field.
    */
    ExcitationAssembler(
        const TriangleMesh<3>& mesh,
        EigRowVec<Index> elems = EigRowVec<Index>::Zero(1, 0)
        ):
        mesh_(mesh),
        elems_(elems)
    {
        if (elems_.cols() == 0)
            elems_ = EigRowVec<Index>::LinSpaced(mesh_.num_elems(), 0, mesh_.num_elems() - 1);
        return;
    };


    /**
    * @brief Assembles the excitation matrix for a given excitation object and observation triangle mesh.
    * @param[out] mat - Matrix to store the assembled excitation coefficients, with columns corresponding
    * to each right-hand side, and rows corresponding to observation mesh edges.
    * @param[in] exc - Excitation object that computes the coefficients to assemble into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        ExcitationBase<TestSpace::dof>& exc,
        const Complex k
        ) override;


protected:

    const TriangleMesh<3>& mesh_;
    EigRowVec<Index> elems_;

};

/**
* @}
*/

}

#include "rwg/assemblers/excitation_matrix.tpp"

#endif
