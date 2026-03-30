// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Quadrature over the observation triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_OBS_QUAD_H
#define BEM_RWG_OPINT_OBS_QUAD_H

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/triangle/base.hpp"
#include "quadrature/triangle/gauss.hpp"

#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/src/strategic.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgopintobs
* @{
*/

/**
* @brief Class for quadrature over the observation triangle for RWG-based BEM operators.
* Reference: O. Ergul, L. Gurel, "The Multilevel Fast Multipole Algorithm (MLFMA) for Solving
* Large-Scale Computational Electromagnetics Problems," book, Wiley-IEEE Press, 2014.
* @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive from
* `TriangleQuadratureBase<3>`.
* @tparam SrcIntegratorType - Object for integrating over the source triangle, which must derive from
* `SrcIntegratorBase`.
*/
template <typename TriangleQuadratureType = GaussTriangleQuadrature<3>, typename SrcIntegratorType = SrcStrategic<>>
class ObsQuadrature: public ObsIntegratorBase
{

    using base = ObsIntegratorBase;
    static_assert(
        std::is_base_of<TriangleQuadratureBase<3>, TriangleQuadratureType>::value,
        "ObsQuadrature: `TriangleQuadratureType` must derive from `TriangleQuadratureBase<3>`"
        );
    static_assert(
        std::is_base_of<SrcIntegratorBase, SrcIntegratorType>::value,
        "ObsQuadrature: `SrcIntegratorType` must derive from `SrcIntegratorBase`"
        );

public:

    /**
    * @brief Constructs an `ObsQuadrature` with a specified triangle quadrature object.
    * @param[in] tri_quad - Triangle quadrature object to use for integration.
    * @param[in] src_integrator - Object for integrating over the source triangle.
    */
    ObsQuadrature(
        const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<3>(),
        const SrcIntegratorType src_integrator = SrcStrategic<>()
        ): tri_quad_(tri_quad), src_integrator_(src_integrator) {};


    /**
    * @brief Computes the integral over the observation triangle.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the local coordinate system of `src_tri`.
    * @param[in] src_tri - Source triangle in 2D space.
    * @return Integration result.
    */
    ObsResult integrate(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<2>& src_tri
        ) override;


    /**
    * @brief Provides read-only access to the triangle quadrature object for inspection.
    * @return Read-only reference to the triangle quadrature object.
    */
    const TriangleQuadratureType& quadrature_object() const
    { return tri_quad_; };


    /**
    * @brief Provides writable access to the triangle quadrature object.
    * @return Writable reference to the triangle quadrature object.
    */
    TriangleQuadratureType& quadrature_object()
    { return tri_quad_; };


    /**
    * @brief Provides read-only access to the source integrator for inspection.
    * @return Read-only reference to the source integrator object.
    */
    const SrcIntegratorType& src_integrator() const
    { return src_integrator_; };


    /**
    * @brief Provides writable access to the source integrator object.
    * @return Writable reference to the source integrator object.
    */
    SrcIntegratorType& src_integrator()
    { return src_integrator_; };


private:

    /**
    * @brief Computes
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * @param[in] src_result - Result of the integration over the source triangle.
    * @return Integration result.
    */
    Complex g_term(const SrcResult& src_result);


    /**
    * @brief Computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\vec{r}\,'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * @param[in] src_result - Result of the integration over the source triangle.
    * @return Integration result.
    */
    EigRowVecN<Complex, 12> rs_g_terms(const SrcResult& src_result);


    /**
    * @brief Computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{r}\,'
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * @param[in] src_result - Result of the integration over the source triangle.
    * @return Integration result.
    */
    EigRowVecN<Complex, 9> grad_g_terms(const SrcResult& src_result);


    /**
    * @brief Computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\hat{n}\times\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{r}\,'
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * @param[in] src_result - Result of the integration over the source triangle.
    * @return Integration result.
    */
    EigRowVecN<Complex, 15> rot_grad_g_terms(const SrcResult& src_result);


    TriangleQuadratureType tri_quad_;
    SrcIntegratorType src_integrator_;

};

/**
* @}
*/

}

#include "rwg/integrators/obs/quadrature.tpp"

#endif
