// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Quadrature over the source triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_QUAD_H
#define BEM_RWG_OPINT_SRC_QUAD_H

#include <memory>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/triangle/base.hpp"
#include "quadrature/triangle/gauss.hpp"
#include "kernels/hgf.hpp"
#include "rwg/integrators/src/base.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgopintsrc
* @{
*/

/**
* @brief Class for quadrature over the source triangle for RWG-based BEM operators.
* Reference:
* - [1] O. Ergul, L. Gurel, "The Multilevel Fast Multipole Algorithm (MLFMA) for Solving Large-Scale
* Computational Electromagnetics Problems," book, Wiley-IEEE Press, 2014.
*/
class SrcQuadrature: public SrcIntegratorBase
{

    using base = SrcIntegratorBase;

public:

    /**
    * @brief Constructs a `SrcQuadrature` with a specified triangle quadrature object.
    * @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive
    * from `TriangleQuadratureBase<2>`.
    * @tparam ScalarKernelType - Object for computing the scalar kernel, which must derive from
    * `ScalarKernelBase<3>`.
    * @param[in] tri_quad - Triangle quadrature object to use for integration (optional).
    * @param[in] kernel - Object for computing the scalar kernel (optional).
    */
    template <
        typename TriangleQuadratureType = GaussTriangleQuadrature<2>,
        typename ScalarKernelType = HGF
        >
    SrcQuadrature(
        const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<2>(),
        const ScalarKernelType kernel = HGF()
        ) { set(tri_quad, kernel); return; };


    /**
    * @brief Sets the triangle quadrature object and kernel object.
    * @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive
    * from `TriangleQuadratureBase<2>`.
    * @tparam ScalarKernelType - Object for computing the scalar kernel, which must derive from
    * `ScalarKernelBase<3>`.
    * @param[in] tri_quad - Triangle quadrature object to use for integration.
    * @param[in] kernel - Object for computing the scalar kernel.
    */
    template <typename TriangleQuadratureType, typename ScalarKernelType>
    void set(
        const TriangleQuadratureType& tri_quad,
        const ScalarKernelType& kernel
        )
    {
        tri_quad_ = std::make_shared<TriangleQuadratureType> (tri_quad);
        kernel_ = std::make_shared<ScalarKernelType> (kernel);
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

    std::shared_ptr<TriangleQuadratureBase<2>> tri_quad_;
    std::shared_ptr<ScalarKernelBase<3>> kernel_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/integrators/src/quadrature.cpp"
#endif

#endif
