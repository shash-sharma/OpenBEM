// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Line integration over the source triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_LINE_H
#define BEM_RWG_OPINT_SRC_LINE_H

#include <memory>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/line/base.hpp"
#include "quadrature/line/gauss.hpp"
#include "quadrature/line/trapz.hpp"
#include "rwg/integrators/src/base.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgopintsrc
* @{
*/

/**
* @brief Class for line integration over the source triangle for RWG-based BEM operators.
* References:
* - [1] T. Xia et al., "An Integral Equation Modeling of Lossy Conductors With the Enhanced
* Augmented Electric Field Integral Equation," in IEEE Transactions on Antennas and Propagation,
* vol. 65, no. 8, pp. 4181-4190, Aug. 2017, doi: 10.1109/TAP.2017.2718587.
* - [2] Z. G. Qian, W. C. Chew and R. Suaya, "Generalized Impedance Boundary Condition for Conductor
* Modeling in Surface Integral Equation," in IEEE Transactions on Microwave Theory and Techniques,
* vol. 55, no. 11, pp. 2354-2364, Nov. 2007, doi: 10.1109/TMTT.2007.908678.
*/
class SrcLineIntegrator: public SrcIntegratorBase
{

    using base = SrcIntegratorBase;

public:

    /**
    * @brief Constructs a `SrcLineIntegrator` with a specified line quadrature object.
    * @tparam LineQuadratureType - Type of the line quadrature object, which must derive from
    * `LineQuadratureBase<1>`.
    * @param[in] line_quad - Line quadrature object to use for integration (optional).
    */
    template <typename LineQuadratureType = GaussLineQuadrature<1>>
    SrcLineIntegrator(const LineQuadratureType line_quad = GaussLineQuadrature<1>())
    { set(line_quad); return; };


    /**
    * @brief Sets the line quadrature object.
    * @tparam LineQuadratureType - Type of the line quadrature object, which must derive from
    * `LineQuadratureBase<1>`.
    * @param[in] line_quad - Line quadrature object to use for integration.
    */
    template <typename LineQuadratureType>
    void set(const LineQuadratureType& line_quad)
    {
        line_quad_ = std::make_shared<LineQuadratureType> (line_quad);
        return;
    };


    /**
    * @brief Computes the integral over the source triangle.
    * @param[in] k - Complex wavenumber.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] r_obs - Observation points in the local coordinate system of `src_tri`.
    * @param[in] g_terms - Whether to compute kernel terms (optional).
    * @param[in] grad_g_terms - Whether to compute kernel gradient terms (optional).
    * @return Integration result.
    * @details
    * If `g_terms` is true, the function computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * and
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\vec{r}\,'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * If `grad_g_terms` is true, the function computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \f]
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,x'\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \f]
    * and
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,y'\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$, and for local source
    * triangle coordinates \f$\vec{r}\,' = [x', y']^T\f$.
    */
    SrcResult integrate(
        const Complex k,
        const Triangle<2>& src_tri,
        ConstEigRef<EigMatNX<Float, 3>> r_obs,
        const bool g_terms = true,
        const bool grad_g_terms = true
        ) override;


protected:

    std::shared_ptr<LineQuadratureBase<1>> line_quad_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/integrators/src/line.cpp"
#endif

#endif
