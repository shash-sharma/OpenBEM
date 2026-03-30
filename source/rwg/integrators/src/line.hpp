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

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/line/base.hpp"
#include "quadrature/line/gauss.hpp"
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
template <typename LineQuadratureType = GaussLineQuadrature<1>>
class SrcLineIntegrator: public SrcIntegratorBase
{

    using base = SrcIntegratorBase;
    static_assert(
        std::is_base_of<LineQuadratureBase<1>, LineQuadratureType>::value,
        "SrcLineIntegrator: `LineQuadratureType` must derive from `LineQuadratureBase<1>`"
        );

public:

    /**
    * @brief Constructs a `SrcLineIntegrator` with a specified line quadrature object.
    * @tparam LineQuadratureType - Type of the line quadrature object, which must derive from
    * `LineQuadratureBase<1>`.
    * @param[in] line_quad - Line quadrature object to use for integration (optional).
    */
    SrcLineIntegrator(const LineQuadratureType line_quad = GaussLineQuadrature<1>()):
        line_quad_(line_quad) {};


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
    * @brief Provides read-only access to the line quadrature object for inspection.
    * @return Read-only reference to the line quadrature object.
    */
    const LineQuadratureType& quadrature_object() const
    { return line_quad_; };


    /**
    * @brief Provides writable access to the line quadrature object.
    * @return Writable reference to the line quadrature object.
    */
    LineQuadratureType& quadrature_object()
    { return line_quad_; };


private:

    LineQuadratureType line_quad_;

};

/**
* @}
*/

}

#include "rwg/integrators/src/line.tpp"

#endif
