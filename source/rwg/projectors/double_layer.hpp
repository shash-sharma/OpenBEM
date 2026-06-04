// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based double-layer potential BEM projectors.
*/

#ifndef BEM_RWG_PROJ_DOUBLE_LAYER_H
#define BEM_RWG_PROJ_DOUBLE_LAYER_H

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"
#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/obs/quadrature.hpp"
#include "rwg/projectors/base.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgproj
* @{
*/

/**
* @brief Class for computing the vector double-layer potential projector.
*/
template <typename SrcIntegratorType = SrcStrategic<>>
class VectorDoubleLayerProj: public ProjectorBase
{

    static_assert(
        std::is_base_of<SrcIntegratorBase, SrcIntegratorType>::value,
        "VectorDoubleLayerProj: `SrcIntegratorType` must derive from `SrcIntegratorBase`"
        );

public:

    /**
    * @brief Constructs a `VectorDoubleLayerProj` object with a specified integration object.
    * @param[in] src_integrator - Integration object for the source triangle (optional).
    */

    VectorDoubleLayerProj(const SrcIntegratorType src_integrator = SrcStrategic<>()):
        src_integrator_(src_integrator) {};


    /**
    * @brief Returns the number of degrees of freedom per triangle for the expansion function space.
    * @return Number of source degrees of freedom per triangle.
    */
    uint8_t src_dof() const override { return 3; };


    /**
    * @brief Computes the vector double-layer projector.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_points - Observation coordinates on which to project the generated field.
    * @param[in] src_tri - Source triangle.
    * @return Projected field components at each observation point, for each source degree of freedom.
    * @details
    * Computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{f_n}(\vec{r}\,')
    * \f]
    * for each source triangle edge \f$ n \f$, where \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel,
    * and \f$ \vec{f_n}(\vec{r}) \f$ is the RWG function associated with edge \f$ n \f$.
    * Rows of the output matrix are ordered by field components and then by observation points, and columns
    * correspond to source edges. I.e., each column of the output looks like
    * \f$ (F_{x1}, F_{y1}, F_{z1}), (F_{x2}, F_{y2}, F_{z2}), \ldots \f$, where
    * \f$ (F_{xi}, F_{yi}, F_{zi}) \f$ are the components of the projected field \f$ \vec{F} \f$
    * defined at the observation point \f$ (x_i, y_i, z_i) \f$.
    */
    EigMat<Complex> compute(
        const Complex k,
        ConstEigRef<EigMatNX<Float, 3>> obs_points,
        const Triangle<3>& src_tri
        ) override;


private:

    SrcIntegratorType src_integrator_;

};

/**
* @}
*/

}

#include "rwg/projectors/double_layer.tpp"

#endif
