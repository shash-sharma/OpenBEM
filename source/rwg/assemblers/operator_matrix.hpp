// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling RWG-based BEM operator matrices.
*/

#ifndef BEM_RWG_OP_ASSEMBLER_H
#define BEM_RWG_OP_ASSEMBLER_H

#include "types.hpp"
#include "rwg/function_space.hpp"
#include "rwg/assemblers/base.hpp"


namespace bem
{
// Forward declarations
template <typename T> class MatrixBase;
}


namespace bem::rwg
{

/**
* \addtogroup assm
* @{
*/

/**
* @brief Class for generating operator matrices for RWG observation and source functions.
* @tparam TestSpace - Testing function space.
* @tparam ExpansionSpace - Expansion function space.
*/
template <typename TestSpace, typename ExpansionSpace>
class OperatorAssembler: public OperatorAssemblerBase<TestSpace, ExpansionSpace>
{

    using base = OperatorAssemblerBase<TestSpace, ExpansionSpace>;

public:

    /**
    * @brief Constructs an `OperatorAssembler` for given observation and source meshes.
    * @param[in] obs_mesh - Observation triangle mesh for which the operator matrix is to be assembled.
    * @param[in] src_mesh - Source triangle mesh for which the operator matrix is to be assembled.
    * @param[in] elem_pairs - Observation (first row) and source (second row) triangle index pairs
    * for which the operator matrix is to be assembled (optional).
    */
    OperatorAssembler(
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh,
        const EigMatNX<Index, 2> elem_pairs = EigMatNX<Index, 2>::Zero(2, 0)
        ):
            obs_mesh_(obs_mesh),
            src_mesh_(src_mesh),
            elem_pairs_(elem_pairs)
    {
        if (elem_pairs_.cols() == 0)
            elem_pairs_ = IndexGenerator::elem_pairs(obs_mesh_, src_mesh_);
        return;
    };


    /**
    * @brief Constructs an `OperatorAssembler` for a given mesh.
    * @param[in] mesh - Triangle mesh for which the operator matrix is to be assembled.
    * @param[in] elem_pairs - Observation (first row) and source (second row) triangle index pairs
    * for which the operator matrix is to be assembled (optional).
    */
    OperatorAssembler(
        const TriangleMesh<3>& mesh,
        const EigMatNX<Index, 2> elem_pairs = EigMatNX<Index, 2>::Zero(2, 0)
        ): OperatorAssembler(mesh, mesh, elem_pairs) {};


    /**
    * @brief Assembles the operator matrix for a given operator object.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] op - Operator object that computes the coefficients to assemble into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        const OperatorBase<TestSpace, ExpansionSpace>& op,
        const Complex k
        ) override;


    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation edges.
    */
    void prep_matrix(MatrixBase<Complex>& mat) override;


    /**
    * @brief Fills operator values in the matrix for edge-based RWG observation and source functions.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation edges.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    void fill_matrix(
        MatrixBase<Complex>& mat,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMatMN<Complex, TestSpace::dof, ExpansionSpace::dof>> values
        ) override;


protected:

    const TriangleMesh<3>& obs_mesh_;
    const TriangleMesh<3>& src_mesh_;
    EigMatNX<Index, 2> elem_pairs_;

};





// /**
// * @brief Class for generating the full set of vector operator matrices for RWG observation and source functions.
// */
// class VectorOperatorsAssembler: public OperatorAssemblerBase<3, 12>
// {

//     using base = OperatorAssemblerBase<3, 12>;
//     using base::base;

// public:

//     /**
//     * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
//     * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
//     * to source edges, and rows corresponding to observation edges. The four vector operator matrices
//     * are stacked along the horizontal direction.
//     */
//     void prep_matrix(MatrixBase<Complex>& mat) override;


//     /**
//     * @brief Fills operator values in the matrix for edge-based RWG observation and source functions.
//     * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
//     * to source edges, and rows corresponding to observation edges, for all operators stacked along
//     * the vertical direction.
//     * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
//     * @param[in] values - Operator values for each pair of observation and source degrees of freedom,
//     * for all operators stacked along the horizontal direction.
//     */
//     void fill_matrix(
//         MatrixBase<Complex>& mat,
//         ConstEigRef<EigColVecN<Index, 2>> elem_pair,
//         ConstEigRef<EigMatMN<Complex, 3, 12>> values
//         ) override;

// };

/**
* @}
*/

}

#include "rwg/assemblers/operator_matrix.tpp"

#endif
