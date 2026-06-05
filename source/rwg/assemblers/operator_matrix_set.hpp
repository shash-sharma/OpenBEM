// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling sets of RWG-based BEM operator matrices.
*/

#ifndef BEM_RWG_OP_SET_ASSEMBLER_H
#define BEM_RWG_OP_SET_ASSEMBLER_H

#include <vector>
#include <tuple>

#include "types.hpp"
#include "rwg/operators/operator_set.hpp"
#include "rwg/assemblers/operator_matrix.hpp"


namespace bem::rwg
{

/**
* \addtogroup assm
* @{
*/

/**
* @brief Class for generating operator matrices for RWG observation and source functions.
* @tparam OperatorSet - Type for the set of operators to be assembled.
* @tparam MatrixType - Assembled matrix type, must inherit from `MatrixBase<Complex>`.
*/
template <typename MatrixType, typename ObsIntegratorType, typename... Ops>
class OperatorSetAssembler
{

    static_assert(
        std::is_base_of<MatrixBase<Complex>, MatrixType>::value,
        "OperatorSetAssembler: `MatrixType` must derive from `MatrixBase<Complex>`"
        );

    static_assert(
        std::is_base_of<ObsIntegratorBase, ObsIntegratorType>::value,
        "OperatorSet: `ObsIntegratorType` must derive from `ObsIntegratorBase`"
        );

public:

    /**
    * @brief Constructs an `OperatorSetAssembler` for given observation and source meshes.
    * @param[in] obs_mesh - Observation triangle mesh for which the operator matrix is to be assembled.
    * @param[in] src_mesh - Source triangle mesh for which the operator matrix is to be assembled.
    * @param[in] elem_pairs - Observation (first row) and source (second row) triangle index pairs
    * for which the operator matrices are to be assembled (optional).
    */
    OperatorSetAssembler(
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh,
        const EigMatNX<Index, 2> elem_pairs = EigMatNX<Index, 2>::Zero(2, 0),
        const ObsIntegratorType obs_integrator = ObsStrategic<>()
        ):
            obs_mesh_(obs_mesh),
            src_mesh_(src_mesh),
            elem_pairs_(elem_pairs),
            obs_integrator_(obs_integrator),
            ops_(Ops{}...)
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
    OperatorSetAssembler(
        const TriangleMesh<3>& mesh,
        const EigMatNX<Index, 2> elem_pairs = EigMatNX<Index, 2>::Zero(2, 0),
        const ObsIntegratorType obs_integrator = ObsStrategic<>()
        ): OperatorSetAssembler(mesh, mesh, elem_pairs, obs_integrator) {};


    /**
    * @brief Assembles the operator matrix for a given operator object.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] op - Operator object that computes the coefficients to assemble into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        std::vector<MatrixType>& mats,
        // const OperatorSet& op_set,
        const Complex k
        )
    {


        mats.clear();

        // using Assemblers = std::tuple<
        //     OperatorAssembler<
        //         ops_::TestSpaceType,
        //         ops_::ExpansionSpaceType
        //         > (obs_mesh_, src_mesh_, elem_pairs_)...
        //     >;

        // using Assemblers = std::tuple<
        //     // EigMatMN<Complex, Ops::TestSpaceType::dof, Ops::ExpansionSpaceType::dof>...
        //     OperatorAssembler<
        //         typename Ops::TestSpaceType,
        //         typename Ops::ExpansionSpaceType
        //         > (obs_mesh_, src_mesh_, elem_pairs_)...
        //     >;

        // std::apply(
        //     [&] (auto&&... Op)
        //     {([&]
        //     {

        //         OperatorAssembler<Op::TestSpaceType, Op::ExpansionSpaceType> assm (
        //             obs_mesh_, src_mesh_, elem_pairs_
        //             );

        //         mats.push_back(MatrixType());
        //         assm.prep_matrix(mats.back());

        //     }(), ...); },
        //     Ops
        //     // Assemblers
        //     );

#pragma omp parallel
        {

            OperatorSet<Ops...> op_set (obs_integrator_);

#pragma omp for
            for (Index ii = 0; ii < elem_pairs_.cols(); ++ii)
            {
                //             Triangle<3> obs_tri = obs_mesh_.elem_primitive(elem_pairs_(0, ii));
                //             Triangle<3> src_tri = src_mesh_.elem_primitive(elem_pairs_(1, ii));

                //             typename OperatorSet::OperatorValuesType op_vals = op_set.compute(k, obs_tri, src_tri);

                //             Index jj = 0;
                //             std::apply(
                //                 [&] (auto&&... values)
                //                 {
                //                     std::apply(
                //                         [&] (auto&&... assm)
                //                         {([&]
                //                         {

                // #pragma omp critical
                //                             assm.fill_matrix(mats[jj++], elem_pairs_.col(ii), values);

                //                         }(), ...); },
                //                         Assemblers
                //                         );
                //                 },
                //                 op_vals
                //                 );

            }
        }

        return;

    };


protected:

    const TriangleMesh<3>& obs_mesh_;
    const TriangleMesh<3>& src_mesh_;
    EigMatNX<Index, 2> elem_pairs_;
    ObsIntegratorType obs_integrator_;
    std::tuple<Ops...> ops_;

};

/**
* @}
*/

}

// #include "rwg/assemblers/operator_matrix.tpp"

#endif
