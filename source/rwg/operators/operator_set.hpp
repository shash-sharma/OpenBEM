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
#include <iostream>

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
    * @details
    * Rows of the output matrix correspond to observation edges, and columns correspond to
    * source edges. The operators are stacked column by column; the first three columns correspond to the first
    * operator, the next three to the next operator, and so on.
    */
    EigMatMN<Complex, 3, 12> compute(
        Ops... ops,
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) override;


    // ops_(std::move(ops)...)
    // std::tuple<Ops...> ops_;
    // constexpr size_t num_ops =
    // sizeof...(Ops);


protected:

    ObsIntegratorType obs_integrator_;

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
