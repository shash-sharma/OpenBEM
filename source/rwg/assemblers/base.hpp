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

#include "types.hpp"
#include "geometry/point_cloud.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/primitives/triangle.hpp"
#include "matrix/base.hpp"
#include "rwg/function_space.hpp"
#include "rwg/operators/base.hpp"
#include "rwg/assemblers/index_generator.hpp"
#include "rwg/excitations/base.hpp"
#include "rwg/projectors/base.hpp"


namespace bem::rwg
{

const Index EDGE_ELEM_RATIO = 2;

/**
* \addtogroup assm
* @{
*/

/**
* @brief Base class for generating BEM operator matrices.
* @tparam TestSpace - Testing function space.
* @tparam ExpansionSpace - Expansion function space.
*/
template <typename TestSpace, typename ExpansionSpace>
class OperatorAssemblerBase
{

    static_assert(
        std::is_base_of<FunctionSpaceBase<
            TestSpace, TestSpace::dof, TestSpace::dim
            >, TestSpace>::value,
        "OperatorAssemblerBase: `TestSpace` must derive from `FunctionSpaceBase`"
        );

    static_assert(
        std::is_base_of<FunctionSpaceBase<
            ExpansionSpace, ExpansionSpace::dof, ExpansionSpace::dim
            >, ExpansionSpace>::value,
        "OperatorAssemblerBase: `ExpansionSpace` must derive from `FunctionSpaceBase`"
        );

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
        const OperatorBase<TestSpace, ExpansionSpace>& op,
        const Complex k
        ) = 0;


    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    */
    virtual void prep_matrix(MatrixBase<Complex>& mat) {};


    /**
    * @brief Fills operator values in the matrix based on source and observation meshes and degrees of freedom.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    virtual void fill_matrix(
        MatrixBase<Complex>& mat,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMatMN<Complex, TestSpace::dof, ExpansionSpace::dof>> values
        ) {};


    /**
    * @brief Virtual destructor.
    */
    virtual ~OperatorAssemblerBase() = default;

};


/**
* @brief Base class for generating excitation matrices for RWG-based BEM systems.
* @tparam TestSpace - Testing function space.
*/
template <typename TestSpace>
class ExcitationAssemblerBase
{

    static_assert(
        std::is_base_of<FunctionSpaceBase<
            TestSpace, TestSpace::dof, TestSpace::dim
            >, TestSpace>::value,
        "ExcitationAssemblerBase: `TestSpace` must derive from `FunctionSpaceBase`"
        );

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
        ExcitationBase<TestSpace>& exc,
        const Complex k
        ) = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~ExcitationAssemblerBase() = default;

};


/**
* @brief Base class for generating RWG-based BEM projector matrices.
* @tparam ExpansionSpace - Expansion function space.
* @tparam obs_dim - Dimension of the projected fields.
*/
template <typename ExpansionSpace, uint8_t obs_dim>
class ProjectorAssemblerBase
{

    static_assert(
        std::is_base_of<FunctionSpaceBase<
            ExpansionSpace, ExpansionSpace::dof, ExpansionSpace::dim
            >, ExpansionSpace>::value,
        "ProjectorAssemblerBase: `ExpansionSpace` must derive from `FunctionSpaceBase`"
        );

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
        ProjectorBase<ExpansionSpace>& op,
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
