// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Strategic auto-integration over the source triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_STRATEGIC_H
#define BEM_RWG_OPINT_SRC_STRATEGIC_H

#include "types.hpp"

#include "geometry/primitives/triangle.hpp"
#include "quadrature/triangle/base.hpp"
#include "quadrature/triangle/gauss.hpp"
#include "quadrature/line/base.hpp"
#include "quadrature/line/gauss.hpp"

#include "kernels/hgf.hpp"
#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/src/quadrature.hpp"
#include "rwg/integrators/src/singularity.hpp"
#include "rwg/integrators/src/line.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgopintsrc
* @{
*/

/**
* @brief Data structure defining settings for strategic integration over the source triangle.
*/
struct SrcIntegrationSettings
{
    /** Source triangle quadrature order for well separated interactions. */
    Float tri_order_far = 4;

    /** Source triangle quadrature order for nearby interactions when singularity treatment is applied. */
    Float tri_order_near = 8;

    /** Source triangle line integration order. */
    Float line_order = 10;

    /** Electrical distance below which singularity treatment is applied (default 0.1 if negative). */
    Float threshold_wvl_singularity = -1;

    /** Physical distance below which singularity treatment is applied (disabled if negative). */
    Float threshold_dist_singularity = -1;

    /** Electrical length of src_tri edge above which line integration is used (default 1 if negative). */
    Float threshold_length_line_int = -1;

    /** Number of skin depths beyond which interactions should be ignored (disabled if negative). */
    Float threshold_skin_depths = 10;
};


/**
* @brief Class integration over the source triangle for RWG-based BEM operators. The method of integration
* is chosen automatically and strategically based on mesh parameters, materials, and frequency.
* @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive from
* `TriangleQuadratureBase<2>`.
* @tparam LineQuadratureType - Type of the line quadrature object, which must derive from
* `LineQuadratureBase<1>`.
*/
template <typename TriangleQuadratureType = GaussTriangleQuadrature<2>, typename LineQuadratureType = GaussLineQuadrature<1>>
class SrcStrategic: public SrcIntegratorBase
{

    using base = SrcIntegratorBase;
    static_assert(
        std::is_base_of<TriangleQuadratureBase<2>, TriangleQuadratureType>::value,
        "SrcStrategic: `TriangleQuadratureType` must derive from `TriangleQuadratureBase<2>`"
        );
    static_assert(
        std::is_base_of<LineQuadratureBase<1>, LineQuadratureType>::value,
        "SrcStrategic: `LineQuadratureType` must derive from `LineQuadratureBase<1>`"
        );

public:

    /**
    * @brief Constructs a `SrcStrategic` integrator with specified line and triangle quadrature objects.
    * @param[in] settings - Integration settings for singularity treatment and line integration (optional).
    * @param[in] tri_quad - Triangle quadrature object to use for integration (optional).
    * @param[in] line_quad - Line quadrature object to use for integration (optional).
    */
    SrcStrategic(
        const SrcIntegrationSettings settings = SrcIntegrationSettings(),
        const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<2>(),
        const LineQuadratureType line_quad = GaussLineQuadrature<1>()
        ):
            settings_(settings),
            src_hgf_(tri_quad, HGF()),
            src_shgf_(tri_quad, SingularitySubtractedHGF()),
            src_sthgf_(tri_quad, SingularitySubtractedTaylorHGF()),
            src_line_(line_quad)
    {
        src_line_.quadrature_object().set_order(settings_.line_order);
        src_sthgf_.quadrature_object().set_order(settings_.tri_order_near);
        src_shgf_.quadrature_object().set_order(settings_.tri_order_near);
        src_hgf_.quadrature_object().set_order(settings_.tri_order_far);
        return;
    };


    /**
    * @brief Computes the integral over the source triangle.
    * @param[in] k - Complex wavenumber.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] r_obs - Observation points in the local coordinate system of `src_tri`.
    * @return Integration result.
    */
    SrcResult integrate(
        const Complex k,
        const Triangle<2>& src_tri,
        ConstEigRef<EigMatNX<Float, 3>> r_obs
        ) override;


private:

    SrcIntegrationSettings settings_;

    SrcQuadrature<TriangleQuadratureType, HGF> src_hgf_;
    SrcSingularity<TriangleQuadratureType, SingularitySubtractedHGF> src_shgf_;
    SrcSingularity<TriangleQuadratureType, SingularitySubtractedTaylorHGF> src_sthgf_;
    SrcLineIntegrator<LineQuadratureType> src_line_;

};

/**
* @}
*/

}

#include "rwg/integrators/src/strategic.tpp"

#endif
