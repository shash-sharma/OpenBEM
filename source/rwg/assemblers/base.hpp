// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base classes for assembling RWG-based BEM matrices.
*/

#ifndef BEM_RWG_ASSEMBLER_BASE_H
#define BEM_RWG_ASSEMBLER_BASE_H

#include <vector>
#include <memory>
#include <stdexcept>

#include "types.hpp"
#include "matrix/base.hpp"
#include "rwg/operators/base.hpp"
#include "rwg/excitations/base.hpp"
#include "rwg/projectors/base.hpp"
#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/strategic.hpp"
#include "rwg/assemblers/indexing.hpp"


namespace bem::rwg
{

const Index EDGE_ELEM_RATIO = 2;

/**
* \addtogroup assm
* @{
*/

/**
* @brief Base class for generating BEM operator matrices.
*/
class OperatorAssemblerBase
{
public:

    /**
    * @brief Assembles the operator matrix for a given operator object.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] op - Operator object that computes the coefficients to assemble into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    virtual void assemble(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        const Complex k
        ) = 0;

    
    /**
    * @brief Assembles a set of operator matrices for given operators.
    * @param[out] mats - Matrix pointers to store the assembled operator coefficients, with columns
    * corresponding to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] ops - Operator object pointers that compute the coefficients to assemble into `mats`.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_integrator - Integration object for the observation triangle for all operators.
    */
    virtual void assemble(
        std::vector<std::shared_ptr<MatrixBase<Complex>>>& mats,
        const std::vector<std::shared_ptr<OperatorBase>>& ops,
        const Complex k,
        ObsIntegratorBase& obs_integrator
        )
    { throw std::runtime_error("OperatorAssemblerBase::assemble(): Not implemented."); };


    /**
    * @brief Rescopes this assembler to a new index set.
    * @param[in] index_set - New block index definition.
    */
    virtual void set_indices(const IndexSet& index_set) {};


    /**
    * @brief Virtual destructor.
    */
    virtual ~OperatorAssemblerBase() = default;

};


/**
* @brief Base class for generating excitation matrices for RWG-based BEM systems.
*/
class ExcitationAssemblerBase
{
public:

    /**
    * @brief Assembles the excitation matrix for a given excitation object and observation triangle mesh.
    * @param[out] mat - Matrix to store the assembled excitation coefficients, with columns corresponding
    * to each right-hand side, and rows corresponding to observation mesh degrees of freedom.
    * @param[in] exc - Excitation object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    virtual void assemble(
        MatrixBase<Complex>& mat,
        ExcitationBase& exc,
        const Complex k
        ) = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~ExcitationAssemblerBase() = default;

};


/**
* @brief Base class for generating RWG-based BEM projector matrices.
* @tparam obs_dim - Dimension of the projected fields.
*/
template <uint8_t obs_dim>
class ProjectorAssemblerBase
{

    static_assert((obs_dim > 0), "`obs_dim` must be greater than 0.");

public:

    /**
    * @brief Assembles the projector matrix for a given projector object, source mesh, and observation points.
    * @param[out] mat - Matrix to store the assembled projector coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation points.
    * @param[in] op - Projector object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    virtual void assemble(
        MatrixBase<Complex>& mat,
        ProjectorBase& op,
        const Complex k
        ) = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~ProjectorAssemblerBase() = default;

};

/**
* @}
*/

}

#endif
