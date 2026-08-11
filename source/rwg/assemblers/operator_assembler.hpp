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

#include <memory>
#include <vector>

#include "types.hpp"
#include "matrix/base.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/strategic.hpp"
#include "rwg/function_space.hpp"
#include "rwg/assemblers/base.hpp"
#include "rwg/assemblers/indexing.hpp"


namespace bem::rwg
{

/**
* \addtogroup assm
* @{
*/

/**
* @brief Class for generating operator matrices for RWG observation and source functions.
*/
class OperatorAssembler: public OperatorAssemblerBase
{

    using base = OperatorAssemblerBase;

public:

    /**
    * @brief Constructs an `OperatorAssembler` for given observation and source meshes.
    * @param[in] obs_mesh - Observation triangle mesh for which the operator matrix is to be assembled.
    * @param[in] src_mesh - Source triangle mesh for which the operator matrix is to be assembled.
    * @param[in] face_pairs - Observation (first row) and source (second row) triangle index pairs
    * for which the operator matrix is to be assembled (optional).
    */
    OperatorAssembler(
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh,
        const EigMatNX<Index, 2> face_pairs = EigMatNX<Index, 2>::Zero(2, 0)
        ):
            obs_mesh_(obs_mesh),
            src_mesh_(src_mesh),
            face_pairs_(face_pairs)
    {
        if (face_pairs_.cols() == 0)
            face_pairs_ = IndexGenerator::face_pairs(obs_mesh_, src_mesh_);
        return;
    };


    /**
    * @brief Constructs an `OperatorAssembler` for a given mesh.
    * @param[in] mesh - Triangle mesh for which the operator matrix is to be assembled.
    * @param[in] face_pairs - Observation (first row) and source (second row) triangle index pairs
    * for which the operator matrix is to be assembled (optional).
    */
    OperatorAssembler(
        const TriangleMesh<3>& mesh,
        const EigMatNX<Index, 2> face_pairs = EigMatNX<Index, 2>::Zero(2, 0)
        ): OperatorAssembler(mesh, mesh, face_pairs) {};


    /**
    * @brief Assembles the operator matrix for a given operator object.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] op - Operator object that computes the coefficients to assemble into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        const Complex k
        ) override;


    /**
    * @brief Assembles a set of operator matrices for given operators, avoiding redundant kernel evaluations.
    * @param[out] mats - Matrix pointers to store the assembled operator coefficients, with columns
    * corresponding to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] ops - Operator object pointers that compute the coefficients to assemble into `mats`.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_integrator - Integration object for the observation triangle for all operators.
    */
    void assemble(
        std::vector<std::shared_ptr<MatrixBase<Complex>>>& mats,
        const std::vector<std::shared_ptr<OperatorBase>>& ops,
        const Complex k,
        ObsIntegratorBase& obs_integrator
        ) override;


    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation edges.
    * @param[in] op - Operator object that computes the coefficients to be assembled into `mat`.
    */
    void prep_matrix(
        MatrixBase<Complex>& mat,
        const OperatorBase& op
        );


    /**
    * @brief Fills operator values in the matrix.
    * @param[out] mat - Matrix to store the assembled operator coefficients.
    * @param[in] op - Operator object that computes the coefficients to be assembled into `mat`.
    * @param[in] face_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    void fill_matrix(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        ConstEigRef<EigColVecN<Index, 2>> face_pair,
        ConstEigRef<EigMat<Complex>> values
        );


protected:

    const TriangleMesh<3>& obs_mesh_;
    const TriangleMesh<3>& src_mesh_;
    EigMatNX<Index, 2> face_pairs_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/operator_assembler.cpp"
#endif

#endif
