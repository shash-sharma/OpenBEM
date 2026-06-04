// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class to generate a given set of operators.
*/

#ifndef BEM_RWG_OPS_OPSET_H
#define BEM_RWG_OPS_OPSET_H

#include <stdexcept>
#include <tuple>
#include <array>

#include "types.hpp"

#include "geometry/primitives/triangle.hpp"

#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/strategic.hpp"
#include "rwg/operators/base.hpp"
#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgops
* @{
*/

/**
* @brief Class for computing the full set of vector RWG operators.
* @tparam ObsIntegratorType - Type of the observation triangle integrator, must derive from `ObsIntegratorBase`.
*/
template <typename... Ops, typename ObsIntegratorType = ObsStrategic<>>
class OperatorSet
{

    static_assert(
        std::is_base_of<ObsIntegratorBase, ObsIntegratorType>::value,
        "VectorRwgOps: `ObsIntegratorType` must derive from `ObsIntegratorBase`"
        );

public:

    /**
    * @brief Constructs an `OperatorSet` object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    OperatorSet(
        const ObsIntegratorType obs_integrator = ObsStrategic<>()
        ):
        obs_integrator_(obs_integrator) {};


    /**
    * @brief Computes the operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom, for each operator.
    * @details Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom. The operators are stacked column by column; the first
    * three columns correspond to the first operator, the next three to the next operator, and so
    * on, because an operator is assumed to have at most 3 observation and 3 source degrees of
    * freedom. If it has fewer degrees of freedom, the remaining entries of its 3x3 block will be
    * zeros.
    */
    EigMatMN<Complex, 3, 3 * num_ops> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        )
    {

        Triangle<3> obs_tri_local;
        Triangle<2> src_tri_local;
        transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

        obs_integrator_.set_compute_terms(true, true, true, true);
        const ObsResult obs_result = obs_integrator_.integrate(
            k, obs_tri_local, src_tri_local
            );

        std::array<Index, num_ops> idxs;
        std::iota(idxs.begin(), idxs.end(), 0);

        EigMatMN<Complex, 3, 3 * num_ops> result;

        std::apply(
        [&] (auto&&... op)
        {
            std::apply(
                [&] (auto&&... idx)
                {([&]
                {

                    result.block(
                        0, 3 * idx, op::TestSpaceType::dof, op::ExpansionSpaceType::dof
                        ) = op.assemble(
                            k, obs_tri_local, src_tri_local.to_3d(), obs_result
                            );

                }(), ...); },
                idxs
                );
        },
        ops_
        );

        return result;

    };


    // ops_(std::move(ops)...)
    // constexpr size_t num_ops =
    // sizeof...(Ops);


protected:

    ObsIntegratorType obs_integrator_;
    std::tuple<Ops...> ops_;
    static constexpr size_t num_ops = std::tuple_size<decltype(ops_)>::value>;

    VectorSingleLayerOp<ObsIntegratorType> vector_single_layer_;
    RotVectorSingleLayerOp<ObsIntegratorType> rot_vector_single_layer_;
    VectorDoubleLayerPvOp<ObsIntegratorType> vector_double_layer_pv_;
    RotVectorDoubleLayerPvOp<ObsIntegratorType> rot_vector_double_layer_pv_;
    RotGradScalarSingleLayerOp<ObsIntegratorType> rot_grad_scalar_single_layer_;
    ScalarSingleLayerOp<ObsIntegratorType> scalar_single_layer_;

};

/**
* @}
*/

}

#include "rwg/operators/vector_ops.tpp"

#endif
