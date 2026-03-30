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
* @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive from
* `TriangleQuadratureBase<2>`.
* @tparam ScalarKernelType - Object for computing the scalar kernel, which must derive from
* `ScalarKernelBase<3>`.
*/
template <typename TriangleQuadratureType = GaussTriangleQuadrature<2>, typename ScalarKernelType = HGF>
class SrcQuadrature: public SrcIntegratorBase
{

    using base = SrcIntegratorBase;
    static_assert(
        std::is_base_of<TriangleQuadratureBase<2>, TriangleQuadratureType>::value,
        "SrcQuadrature: `TriangleQuadratureType` must derive from `TriangleQuadratureBase<2>`"
        );
    static_assert(
        std::is_base_of<ScalarKernelBase<3>, ScalarKernelType>::value,
        "SrcQuadrature: `ScalarKernelType` must derive from `ScalarKernelBase<3>`"
        );

public:

    /**
    * @brief Constructs a `SrcQuadrature` with a specified triangle quadrature object.
    * @param[in] tri_quad - Triangle quadrature object to use for integration (optional).
    * @param[in] kernel - Object for computing the scalar kernel (optional).
    */
    SrcQuadrature(
        const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<2>(),
        const ScalarKernelType kernel = HGF()
        ): tri_quad_(tri_quad), kernel_(kernel) {};


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


private:

    TriangleQuadratureType tri_quad_;
    ScalarKernelType kernel_;

};

/**
* @}
*/

}

#include "rwg/integrators/src/quadrature.tpp"

#endif
