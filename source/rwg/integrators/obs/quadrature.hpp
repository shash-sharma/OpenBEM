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

#include <memory>

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
*/
class ObsQuadrature: public ObsIntegratorBase
{

    using base = ObsIntegratorBase;

public:

    /**
    * @brief Constructs an `ObsQuadrature` with a specified triangle quadrature object.
    * @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive from
    * `TriangleQuadratureBase<3>`.
    * @tparam SrcIntegratorType - Object for integrating over the source triangle, which must derive from
    * `SrcIntegratorBase`.
    * @param[in] tri_quad - Triangle quadrature object to use for integration.
    * @param[in] src_integrator - Object for integrating over the source triangle.
    */
    template <
        typename TriangleQuadratureType = GaussTriangleQuadrature<3>,
        typename SrcIntegratorType = SrcStrategic
        >
    ObsQuadrature(
        const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<3>(),
        const SrcIntegratorType src_integrator = SrcStrategic()
        ) { set(tri_quad, src_integrator); return; };


    /**
    * @brief Sets the triangle quadrature object and kernel object.
    * @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive
    * from `TriangleQuadratureBase<3>`.
    * @tparam SrcIntegratorType - Object for integrating over the source triangle, which must derive from
    * `SrcIntegratorBase`.
    * @param[in] tri_quad - Triangle quadrature object to use for integration.
    * @param[in] src_integrator - Object for integrating over the source triangle.
    */
    template <typename TriangleQuadratureType, typename SrcIntegratorType>
    void set(const TriangleQuadratureType& tri_quad, const SrcIntegratorType& src_integrator)
    {
        tri_quad_ = std::make_shared<TriangleQuadratureType> (std::move(tri_quad));
        src_integrator_ = std::make_shared<SrcIntegratorType> (std::move(src_integrator));
        return;
    };


    /**
    * @brief Computes the integral over the observation triangle.
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
    { return std::make_unique<ObsQuadrature> (*this); };


protected:

    /**
    * @brief Computes
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * @param[in] src_result - Result of the integration over the source triangle.
    * @param[in] qd - Observation triangle quadrature data.
    * @return Integration result.
    */
    Complex compute_g_term(
        const SrcResult& src_result, const QuadratureData<3>& qd
        );


    /**
    * @brief Computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\vec{r}\,'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * @param[in] src_result - Result of the integration over the source triangle.
    * @param[in] qd - Observation triangle quadrature data.
    * @return Integration result.
    */
    EigRowVecN<Complex, 12> compute_rs_g_terms(
        const SrcResult& src_result, const QuadratureData<3>& qd
        );


    /**
    * @brief Computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{r}\,'
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * @param[in] src_result - Result of the integration over the source triangle.
    * @param[in] qd - Observation triangle quadrature data.
    * @return Integration result.
    */
    EigRowVecN<Complex, 9> compute_grad_g_terms(
        const SrcResult& src_result, const QuadratureData<3>& qd
        );


    /**
    * @brief Computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\hat{n}\times\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{r}\,'
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * @param[in] src_result - Result of the integration over the source triangle.
    * @param[in] qd - Observation triangle quadrature data.
    * @return Integration result.
    */
    EigRowVecN<Complex, 15> compute_rot_grad_g_terms(
        const SrcResult& src_result, const QuadratureData<3>& qd);


    std::shared_ptr<TriangleQuadratureBase<3>> tri_quad_;
    std::shared_ptr<SrcIntegratorBase> src_integrator_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/integrators/obs/quadrature.cpp"
#endif

#endif
