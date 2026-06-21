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
        const OperatorBase& op,
        const Complex k
        ) override;


    /**
    * @brief Assembles operator matrices for given operator objects.
    * @param[out] mats - Matrices to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] ops - Operator objects that compute the coefficients to assemble into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    template <typename MatrixType, typename ObsIntegratorType = ObsStrategic<>, typename... Ops>
    void assemble(
        std::vector<MatrixType>& mats,
        const Complex k,
        const ObsIntegratorType obs_integrator = ObsStrategic<>()
        );


protected:

    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation edges.
    */
    void prep_matrix(
        MatrixBase<Complex>& mat,
        const OperatorBase& op
        );


    /**
    * @brief Fills operator values in the matrix for edge-based RWG observation and source functions.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation edges.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    void fill_matrix(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMat<Complex>> values
        );


    const TriangleMesh<3>& obs_mesh_;
    const TriangleMesh<3>& src_mesh_;
    EigMatNX<Index, 2> elem_pairs_;

};


template <typename MatrixType, typename ObsIntegratorType, typename... Ops>
void OperatorAssembler::assemble(
    std::vector<MatrixType>& mats,
    const Complex k,
    ObsIntegratorType obs_integrator
    )
{

    static_assert((std::is_base_of_v<OperatorBase, Ops> && ...));

    std::vector<std::unique_ptr<OperatorBase>> ops;
    (ops.push_back(std::make_unique<Ops>()), ...);

    mats.resize(ops.size());
    for (Index ii = 0; ii < ops.size(); ++ii)
        prep_matrix(mats[ii], *ops[ii]);

#pragma omp parallel firstprivate(obs_integrator)
    {

        std::vector<std::unique_ptr<OperatorBase>> ops;
        (ops.push_back(std::make_unique<Ops>()), ...);

#pragma omp for
        for (Index ii = 0; ii < elem_pairs_.cols(); ++ii)
        {
            Triangle<3> obs_tri = obs_mesh_.elem_primitive(elem_pairs_(0, ii));
            Triangle<3> src_tri = src_mesh_.elem_primitive(elem_pairs_(1, ii));

            Triangle<3> obs_tri_local;
            Triangle<2> src_tri_local;
            OperatorBase::transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

            obs_integrator.set_compute_terms(true, true, true, true);
            const ObsResult obs_result = obs_integrator.integrate(k, obs_tri_local, src_tri_local);

            for (Index jj = 0; jj < ops.size(); ++jj)
            {
                EigMat<Complex> values = ops[jj]->assemble(
                    k, obs_tri_local, src_tri_local.to_3d(), obs_result
                    );

#pragma omp critical
                fill_matrix(mats[jj], *ops[jj], elem_pairs_.col(ii), values);
            }
        }

    }

    for (auto& mat: mats)
        mat.assemble();

    return;

}

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/operator_assembler.cpp"
#endif

#endif
