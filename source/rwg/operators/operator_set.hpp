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

#include <tuple>

#include "types.hpp"

#include "geometry/primitives/triangle.hpp"
#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/strategic.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgops
* @{
*/

/**
* @brief Class for computing a given set of operators.
* @tparam ObsIntegratorType - Type of the observation triangle integrator, must derive from `ObsIntegratorBase`.
*/
template <typename ObsIntegratorType, typename... Ops>
class OperatorSet
{

    static_assert(
        std::is_base_of<ObsIntegratorBase, ObsIntegratorType>::value,
        "OperatorSet: `ObsIntegratorType` must derive from `ObsIntegratorBase`"
        );

public:

    static constexpr std::size_t num_ops = sizeof...(Ops);
    static std::tuple<Ops...> ops = std::make_tuple ();

    using OperatorsType = std::tuple<Ops...>;
    using OperatorValuesType = std::tuple<
        EigMatMN<Complex, Ops::TestSpaceType::dof, Ops::ExpansionSpaceType::dof>...
        >;


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
    */
    OperatorValuesType compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        )
    {

        Triangle<3> obs_tri_local;
        Triangle<2> src_tri_local;
        OperatorBase<Rwg, Rwg>::transform_coordinates(
            obs_tri_local, src_tri_local, obs_tri, src_tri
            );

        obs_integrator_.set_compute_terms(true, true, true, true);
        const ObsResult obs_result = obs_integrator_.integrate(
            k, obs_tri_local, src_tri_local
            );

        OperatorValuesType op_vals;

        std::apply(
        [&] (auto&&... op)
        {
            std::apply(
                [&] (auto&&... op_val)
                {([&]
                {

                    op_val = op.assemble(
                        k, obs_tri_local, src_tri_local.to_3d(), obs_result
                        );

                }(), ...); },
                op_vals
                );
        },
        ops
        );

        return op_vals;

    };


protected:

    ObsIntegratorType obs_integrator_;
    // std::tuple<Ops...> ops_;

};

/**
* @}
*/

}

// #include "rwg/operators/vector_ops.tpp"

#endif
