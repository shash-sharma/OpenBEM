// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based Gram matrix operators.
*/

#ifndef BEM_RWG_OPS_GRAM_H
#define BEM_RWG_OPS_GRAM_H

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/triangle/gauss.hpp"
#include "rwg/function_space.hpp"
#include "rwg/operators/base.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgops
* @{
*/

/**
* @brief Class for computing the RWG identity operator.
* @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive from
* `TriangleQuadratureBase<3>`.
* @details
* Computes
* \f[
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{f_m}(\vec{r})\cdot
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\vec{f_n}(\vec{r}\,')
* \f]
* for each pair \f$ (m, n) \f$ of observation and source triangle edges, where
* \f$ \vec{f_i}(\vec{r}) \f$ is the RWG function associated with edge \f$ i \f$. The
* result is nonzero only when \f$ \text{obs\_tri} \f$ and \f$ \text{src\_tri} \f$ overlap.
* Rows of the output matrix correspond to observation edges, and columns correspond
* to source edges.
*/
template <typename TriangleQuadratureType = GaussTriangleQuadrature<3>>
class VectorIdentityOp: public OperatorBase
{

    static_assert(
        std::is_base_of<TriangleQuadratureBase<3>, TriangleQuadratureType>::value,
        "VectorIdentityOp: `TriangleQuadratureType` must derive from `TriangleQuadratureBase<3>`"
        );

public:

    /**
    * @brief Constructs a `VectorIdentityOp` object with a specified quadrature object for integration.
    * @param[in] tri_quad - Triangle quadrature object to use for integration.
    */
    VectorIdentityOp(const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<3>()):
        tri_quad_(tri_quad) {};


    /**
    * @brief Returns the number of degrees of freedom per triangle for the testing function space.
    * @return Number of observation degrees of freedom per triangle.
    */
    uint8_t obs_dof() const override { return 3; };


    /**
    * @brief Returns the number of degrees of freedom per triangle for the expansion function space.
    * @return Number of source degrees of freedom per triangle.
    */
    uint8_t src_dof() const override { return 3; };


    /**
    * @brief Computes the RWG identity operator.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source triangle edges.
    */
    EigMat<Complex> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<VectorIdentityOp<TriangleQuadratureType>> (*this); };


private:

    TriangleQuadratureType tri_quad_;

};


/**
* @brief Class for computing the rotationally-tested RWG identity operator.
* @tparam TriangleQuadratureType - Type of the triangle quadrature object, which must derive from
* `TriangleQuadratureBase<3>`.
* @details
* Computes
* \f[
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\hat{n}\times\vec{f_m}(\vec{r})\cdot
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\vec{f_n}(\vec{r}\,')
* \f]
* for each pair \f$ (m, n) \f$ of observation and source triangle edges, where
* \f$ \vec{f_i}(\vec{r}) \f$ is the RWG function associated with edge \f$ i \f$, and
* \f$ \hat{n} \f$ is the unit normal vector associated with `obs_tri`. The result
* is nonzero only when \f$ \text{obs\_tri} \f$ and \f$ \text{src\_tri} \f$ overlap.
* Rows of the output matrix correspond to observation edges, and columns correspond
* to source edges.
*/
template <typename TriangleQuadratureType = GaussTriangleQuadrature<3>>
class RotVectorIdentityOp: public OperatorBase
{

    static_assert(
        std::is_base_of<TriangleQuadratureBase<3>, TriangleQuadratureType>::value,
        "RotVectorIdentityOp: `TriangleQuadratureType` must derive from `TriangleQuadratureBase<3>`"
        );

public:

    /**
    * @brief Constructs a `RotVectorIdentityOp` object with a specified quadrature object for integration.
    * @param[in] tri_quad - Triangle quadrature object to use for integration.
    */
    RotVectorIdentityOp(const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<3>()):
        tri_quad_(tri_quad) {};


    /**
    * @brief Returns the number of degrees of freedom per triangle for the testing function space.
    * @return Number of observation degrees of freedom per triangle.
    */
    uint8_t obs_dof() const override { return 3; };


    /**
    * @brief Returns the number of degrees of freedom per triangle for the expansion function space.
    * @return Number of source degrees of freedom per triangle.
    */
    uint8_t src_dof() const override { return 3; };


    /**
    * @brief Computes the rotationally-tested RWG identity operator.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source triangle edges.
    */
    EigMat<Complex> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<RotVectorIdentityOp<TriangleQuadratureType>> (*this); };


private:

    TriangleQuadratureType tri_quad_;

};

/**
* @}
*/

}

#include "rwg/operators/gram.tpp"

#endif
