// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Strategic auto-integration over the observation and source triangles for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_OBS_STRATEGIC_H
#define BEM_RWG_OPINT_OBS_STRATEGIC_H

#include <memory>

#include "types.hpp"

#include "geometry/primitives/triangle.hpp"
#include "quadrature/triangle/base.hpp"
#include "quadrature/triangle/gauss.hpp"
#include "quadrature/line/base.hpp"
#include "quadrature/line/gauss.hpp"

#include "kernels/hgf.hpp"
#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/quadrature.hpp"
#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/src/quadrature.hpp"
#include "rwg/integrators/src/singularity.hpp"
#include "rwg/integrators/src/line.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgopintobs
* @{
*/

/**
* @brief Data structure defining settings for strategic integration over the observation and source triangles.
*/
struct IntegrationSettings
{

    /** Observation triangle quadrature order for well separated interactions. */
    Float obs_quad_order_far = 4;

    /** Observation triangle quadrature order for nearby interactions when singularity treatment is applied. */
    Float obs_quad_order_near = 8;

    /** Source triangle quadrature order for well separated interactions. */
    Float src_quad_order_far = 4;

    /** Source triangle quadrature order for nearby interactions when singularity treatment is applied. */
    Float src_quad_order_near = 8;

    /** Source triangle line integration order. */
    Float src_line_order = 10;

    /** Physical distance below which singularity treatment is applied (disabled if negative). */
    Float threshold_dist_singularity = -1;

    /** Electrical length of src_tri edge above which line integration is used (default 1 if negative). */
    Float threshold_length_line_int = -1;

    /** Number of skin depths beyond which interactions should be ignored (disabled if negative). */
    Float threshold_skin_depths = 10;

};


/**
* @brief Strategic integration over the observation triangle for RWG-based BEM operators. The method of
* integration is chosen automatically and strategically based on mesh parameters, materials, and frequency.
*/
class ObsStrategic: public ObsIntegratorBase
{

    using base = ObsIntegratorBase;

public:

    /**
    * @brief Constructs an `ObsStrategic` integrator with specified line and triangle quadrature objects.
    */
    ObsStrategic(const IntegrationSettings settings = IntegrationSettings())
    { set(settings); return; }


    /**
    * @brief Sets specified line and triangle quadrature object types.
    * @tparam ObsTriangleQuadratureType - Type of the observation triangle quadrature object, which must derive
    * from `TriangleQuadratureBase<2>`.
    * @tparam SrcTriangleQuadratureType - Type of the source triangle quadrature object, which must derive from
    * `TriangleQuadratureBase<2>`.
    * @tparam LineQuadratureType - Type of the line quadrature object, which must derive from
    * `LineQuadratureBase<1>`.
    * @param[in] settings - Integration settings for singularity treatment and line integration (optional).
    */
    template <
        typename ObsTriangleQuadratureType = GaussTriangleQuadrature<3>,
        typename SrcTriangleQuadratureType = GaussTriangleQuadrature<2>,
        typename LineQuadratureType = GaussLineQuadrature<1>
        >
    void set(const IntegrationSettings settings = IntegrationSettings())
    {
        settings_ = settings;

        ObsTriangleQuadratureType obs_tri_quad;
        SrcTriangleQuadratureType src_tri_quad;

        SrcTriangleQuadratureType src_tri_quad_near = src_tri_quad;
        src_tri_quad_near.set_order(settings_.src_quad_order_near);

        SrcTriangleQuadratureType src_tri_quad_far = src_tri_quad;
        src_tri_quad_far.set_order(settings_.src_quad_order_far);

        ObsTriangleQuadratureType obs_tri_quad_near = obs_tri_quad;
        obs_tri_quad_near.set_order(settings_.obs_quad_order_near);

        ObsTriangleQuadratureType obs_tri_quad_far = obs_tri_quad;
        obs_tri_quad_far.set_order(settings_.obs_quad_order_far);

        src_hgf_.set(src_tri_quad_far, HGF());
        src_shgf_.set(src_tri_quad_near, SingularitySubtractedHGF());
        src_sthgf_.set(src_tri_quad_near, SingularitySubtractedTaylorHGF());
        src_line_.set(LineQuadratureType (settings_.src_line_order));

        hgf_.set(obs_tri_quad_far, src_hgf_);
        shgf_.set(obs_tri_quad_near, src_shgf_);
        sthgf_.set(obs_tri_quad_near, src_sthgf_);
        line_.set(obs_tri_quad_far, src_line_);

        return;
    };


    /**
    * @brief Computes the integral over the source triangle.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the local coordinate system of `src_tri`.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] g_term - Whether to compute the scalar kernel term (optional).
    * @param[in] rs_g_terms - Whether to compute the vector kernel termss (optional).
    * @param[in] grad_g_terms - Whether to compute the gradient kernel terms (optional).
    * @param[in] rot_grad_g_terms - Whether to compute the rotated gradient kernel terms (optional).
    * @return Integration result.
    * @details
    * If `g_term` is true, computes
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * If `rs_g_terms` is true, computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\vec{r}\,'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * If `grad_g_terms` is true, computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{r}\,'
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * If `rot_grad_g_terms` is true, computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\hat{n}\times\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{r}\,'
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    */
    ObsResult integrate(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<2>& src_tri,
        const bool g_term = true,
        const bool rs_g_terms = true,
        const bool grad_g_terms = true,
        const bool rot_grad_g_terms = true
        ) override;


    /**
    * @brief Returns a unique pointer to a copy of this object.
    * @return Unique pointer to a copy of this object.
    */
    std::unique_ptr<ObsIntegratorBase> clone() const override
    {
        return std::make_unique<ObsStrategic> (*this);
    };


private:

    IntegrationSettings settings_;

    SrcQuadrature src_hgf_;
    SrcSingularity src_shgf_;
    SrcSingularity src_sthgf_;
    SrcLineIntegrator src_line_;

    ObsQuadrature hgf_;
    ObsQuadrature shgf_;
    ObsQuadrature sthgf_;
    ObsQuadrature line_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/integrators/obs/strategic.cpp"
#endif

#endif
