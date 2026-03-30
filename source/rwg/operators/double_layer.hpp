// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based double-layer potential BEM operators.
*/

#ifndef BEM_RWG_OPS_DOUBLE_LAYER_H
#define BEM_RWG_OPS_DOUBLE_LAYER_H

#include <type_traits>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"

#include "rwg/operators/base.hpp"

#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/strategic.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgops
* @{
*/

/**
* @brief Class for computing the vector double-layer potential operator in a principal value sense.
* @tparam ObsIntegratorType - Type of the observation triangle integrator, must derive from `ObsIntegratorBase`.
* @details
* Computes
* \f[
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{f_m}(\vec{r})\cdot
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{f_n}(\vec{r}\,')
* \f]
* in a principal value sense, for each pair \f$ (m, n) \f$ of observation and source triangle edges, where
* \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, and \f$ \vec{f_i}(\vec{r}) \f$ is the RWG
* function associated with edge \f$ i \f$.
* Rows of the output matrix correspond to observation edges, and columns
* correspond to source edges.
*/
template <typename ObsIntegratorType = ObsStrategic<>>
class VectorDoubleLayerPvOp: public OperatorBase<3, 3>
{

    static_assert(
        std::is_base_of<ObsIntegratorBase, ObsIntegratorType>::value,
        "VectorDoubleLayerPvOp: `ObsIntegratorType` must derive from `ObsIntegratorBase`"
        );

public:

    /**
    * @brief Constructs a `VectorDoubleLayerPvOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    VectorDoubleLayerPvOp(const ObsIntegratorType obs_integrator = ObsStrategic<>()):
        obs_integrator_(obs_integrator) {};


    /**
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
    */
    EigMatMN<Complex, 3, 3> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) override;


    /**
    * @brief Assembles the computed integrals into the final operator values.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the source's local coordinate system.
    * @param[in] src_tri - Source triangle in its local coordinate system.
    * @param[in] obs_result - Integration result.
    * @return Operator values for each pair of observation and source triangle edges.
    */
    EigMatMN<Complex, 3, 3> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        );


protected:

    ObsIntegratorType obs_integrator_;

};


/**
* @brief Class for computing the rotationally-tested vector double-layer potential operator in a principal value sense.
* @tparam ObsIntegratorType - Type of the observation triangle integrator, must derive from `ObsIntegratorBase`.
* @details
* Computes
* \f[
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,(\nhat\times\vec{f_m}(\vec{r}))\cdot
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{f_n}(\vec{r}\,')
* \f]
* in a principal value sense, for each pair \f$ (m, n) \f$ of observation and source triangle edges, where
* \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, and \f$ \vec{f_i}(\vec{r}) \f$ is the RWG
* function associated with edge \f$ i \f$.
* Rows of the output matrix correspond to observation edges, and columns
* correspond to source edges.
*/
template <typename ObsIntegratorType = ObsStrategic<>>
class RotVectorDoubleLayerPvOp: public OperatorBase<3, 3>
{

    static_assert(
        std::is_base_of<ObsIntegratorBase, ObsIntegratorType>::value,
        "RotVectorDoubleLayerPvOp: `ObsIntegratorType` must derive from `ObsIntegratorBase`"
        );

public:

    /**
    * @brief Constructs a `RotVectorDoubleLayerPvOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    RotVectorDoubleLayerPvOp(const ObsIntegratorType obs_integrator = ObsStrategic<>()):
        obs_integrator_(obs_integrator) {};


    /**
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
    */
    EigMatMN<Complex, 3, 3> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) override;


    /**
    * @brief Assembles the computed integrals into the final operator values.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the source's local coordinate system.
    * @param[in] src_tri - Source triangle in its local coordinate system.
    * @param[in] obs_result - Integration result.
    * @return Operator values for each pair of observation and source triangle edges.
    */
    EigMatMN<Complex, 3, 3> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        );


protected:

    ObsIntegratorType obs_integrator_;

};

/**
* @}
*/

}

#include "rwg/operators/double_layer.tpp"

#endif
