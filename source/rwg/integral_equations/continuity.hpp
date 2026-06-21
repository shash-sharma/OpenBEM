// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class for assembling discrete operators for the surface continuity equation.
*/

#ifndef BEM_RWG_INTEQ_CONT_H
#define BEM_RWG_INTEQ_CONT_H

#include "types.hpp"
#include "constants.hpp"
#include "materials.hpp"

#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"

#include "rwg/integral_equations/base.hpp"
#include "rwg/operators/incidence.hpp"
#include "rwg/assemblers/operator_assembler.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgie
* @{
*/

/**
* @brief Class defining the RWG-based surface continuity equation.
* @tparam MatrixType - Matrix type used for all operators, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenMatrix<Complex>>
class Continuity: public IntegralEquationBase<MatrixType>
{

    using base = IntegralEquationBase<MatrixType>;
    using base::base;

public:

    /**
    * @brief Returns the operator matrix associated with the electric surface current density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @return Operator matrix.
    */
    MatrixType j_matrix(
        const Float f,
        const Material& material
        )
    {
        DivergenceOp op_D;
        MatrixType D;
        assembler_.assemble(D, op_D, 0);
        return D;
    }


    /**
    * @brief Returns the operator matrix associated with the electric surface charge density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @return Operator matrix.
    */
    MatrixType rho_matrix(
        const Float f,
        const Material& material
        )
    {
        MatrixType I (base::obs_mesh_.num_elems(), base::src_mesh_.num_elems());
        I.set_identity();
        I.scale(J * two_pi * f);
        return I;
    }


protected:

    OperatorAssembler assembler_ = OperatorAssembler (base::obs_mesh_, base::src_mesh_, base::elem_pairs_);

};

/**
* @}
*/

}

#endif
