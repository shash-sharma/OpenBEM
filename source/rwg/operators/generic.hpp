// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Convenience wrapper classes to generate any operator based on a given integer flag.
*/

#ifndef BEM_RWG_OPS_GENERIC_H
#define BEM_RWG_OPS_GENERIC_H

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
#include "rwg/operators/incidence.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgops
* @{
*/

/**
* @brief Enum class defining operator names.
*/
enum class OperatorName
{
    VECTOR_SINGLE_LAYER,
    ROT_VECTOR_SINGLE_LAYER,
    VECTOR_DOUBLE_LAYER_PV,
    ROT_VECTOR_DOUBLE_LAYER_PV,
    SCALAR_SINGLE_LAYER,
    VECTOR_HYPERSINGULAR,
    ROT_VECTOR_HYPERSINGULAR,
    RWG_RWG,
    ROT_RWG_RWG,
    PULSE_PULSE,
    DIVRWG,
    DEFAULT
};

/**
* @brief Overloaded output stream operator for `OperatorName`.
* @param[in, out] os - Output stream.
* @param[in] op_name - Operator name to be printed.
* @return Output stream with the operator name description.
*/
std::ostream& operator<<(std::ostream& os, OperatorName op_name)
{

    switch (op_name)
    {
        case OperatorName::VECTOR_SINGLE_LAYER:
            os << "< RWG | G . RWG >";
            break;
        case OperatorName::ROT_VECTOR_SINGLE_LAYER:
            os << "< nxRWG | G . RWG >";
            break;
        case OperatorName::VECTOR_DOUBLE_LAYER_PV:
            os << "< RWG | gradG x RWG > (pv)";
            break;
        case OperatorName::ROT_VECTOR_DOUBLE_LAYER_PV:
            os << "< nxRWG | gradG x RWG > (pv)";
            break;
        case OperatorName::SCALAR_SINGLE_LAYER:
            os << "< pulse | G . pulse >";
            break;
        case OperatorName::VECTOR_HYPERSINGULAR:
            os << "< RWG | G + (1/k^2) hessG . RWG >";
            break;
        case OperatorName::ROT_VECTOR_HYPERSINGULAR:
            os << "< nxRWG | G + (1/k^2) hessG . RWG >";
            break;
        case OperatorName::RWG_RWG:
            os << "< RWG | RWG >";
            break;
        case OperatorName::ROT_RWG_RWG:
            os << "< nxRWG | RWG >";
            break;
        case OperatorName::DIVRWG:
            os << "< divRWG >";
            break;
        default:
            throw std::invalid_argument(
                "OperatorName: `op_name` is invalid or not implemented."
                );
            break;
    }

    return os;

}


/**
* @brief Class for computing RWG operators based on a given `OperatorName`.
*/
template <typename ObsIntegratorType = ObsStrategic<>>
class GenericRwgOp: public OperatorBase<3, 3>
{

    static_assert(
        std::is_base_of<ObsIntegratorBase, ObsIntegratorType>::value,
        "GenericRwgOp: `ObsIntegratorType` must derive from `ObsIntegratorBase`"
        );

public:

    /**
    * @brief Constructs a `GenericRwgOp` for computing operator `op_name` with a specified integration object.
    * @param[in] op_name - Name of the operator to be computed.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    GenericRwgOp(const OperatorName op_name, const ObsIntegratorType obs_integrator = ObsStrategic<>()):
        op_name_(op_name),
        vector_single_layer_(obs_integrator),
        rot_vector_single_layer_(obs_integrator),
        vector_double_layer_pv_(obs_integrator),
        rot_vector_double_layer_pv_(obs_integrator),
        rot_grad_scalar_single_layer_(obs_integrator),
        vector_hypersingular_(obs_integrator),
        rot_vector_hypersingular_(obs_integrator) {};


    /**
    * @brief Computes the operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation edges, and columns correspond to
    * source edges.
    */
    EigMat<Complex> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase<3, 3>> clone() const override
    { return std::make_unique<GenericRwgOp<ObsIntegratorType>> (*this); };


private:

    OperatorName op_name_;
    VectorSingleLayerOp<ObsIntegratorType> vector_single_layer_;
    RotVectorSingleLayerOp<ObsIntegratorType> rot_vector_single_layer_;
    VectorDoubleLayerPvOp<ObsIntegratorType> vector_double_layer_pv_;
    RotVectorDoubleLayerPvOp<ObsIntegratorType> rot_vector_double_layer_pv_;
    RotGradScalarSingleLayerOp<ObsIntegratorType> rot_grad_scalar_single_layer_;
    VectorHypersingularOp<ObsIntegratorType> vector_hypersingular_;
    RotVectorHypersingularOp<ObsIntegratorType> rot_vector_hypersingular_;
    RwgRwgOp<> rwg_rwg_;
    RotRwgRwgOp<> rot_rwg_rwg_;

};


/**
* @brief Class for computing pulse function operators based on a given `OperatorName`.
*/
template <typename ObsIntegratorType = ObsQuadrature<>>
class GenericPulseOp: public OperatorBase<1, 1>
{

    static_assert(
        std::is_base_of<ObsIntegratorBase, ObsIntegratorType>::value,
        "GenericPulseOp: `ObsIntegratorType` must derive from `ObsIntegratorBase`"
        );

public:

    /**
    * @brief Constructs a `GenericPulseOp` for computing operator `op_name` with a specified integration object.
    * @param[in] op_name - Name of the operator to be computed.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    GenericPulseOp(const OperatorName op_name, const ObsIntegratorType obs_integrator = ObsStrategic<>()):
        op_name_(op_name), scalar_single_layer_(obs_integrator) {};


    /**
    * @brief Computes the operator value for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator value.
    */
    EigMat<Complex> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase<1, 1>> clone() const override
    { return std::make_unique<GenericPulseOp<ObsIntegratorType>> (*this); };


private:

    OperatorName op_name_;
    ScalarSingleLayerOp<ObsIntegratorType> scalar_single_layer_;

};

/**
* @}
*/

}

#include "rwg/operators/generic.tpp"

#endif
