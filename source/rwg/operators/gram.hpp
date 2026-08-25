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

#include <memory>

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
class VectorIdentityOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `VectorIdentityOp` object with a specified quadrature object for integration.
    * @tparam TriangleQuadratureType - Type of the triangle quadrature object, derived from
    * `TriangleQuadratureBase<3>`.
    * @param[in] tri_quad - Triangle quadrature object to use for integration.
    */
    template <typename TriangleQuadratureType = GaussTriangleQuadrature<3>>
    VectorIdentityOp(const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<3>()):
        tri_quad_(std::make_shared<TriangleQuadratureType> (tri_quad)) {};


    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    OperatorDof obs_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Returns the observation-triangle integrals required.
    * @return Integrals required, none of which this operator uses.
    */
    ObsTerms obs_terms() const override { return { false, false, false, false }; };


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
        ) const override;


    /**
    * @brief Assembles the computed integrals into the final operator values.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the source's local coordinate system.
    * @param[in] src_tri - Source triangle in its local coordinate system.
    * @param[in] obs_result - Integration result.
    * @return Operator values for each observation triangle edge and source triangle face.
    */
    EigMat<Complex> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        ) const override { return compute(k, obs_tri, src_tri); };


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<VectorIdentityOp> (*this); };


private:

    std::shared_ptr<TriangleQuadratureBase<3>> tri_quad_;

};


/**
* @brief Class for computing the rotationally-tested RWG identity operator.
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
class RotVectorIdentityOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `RotVectorIdentityOp` object with a specified quadrature object for integration.
    * @tparam TriangleQuadratureType - Type of the triangle quadrature object, derived from
    * `TriangleQuadratureBase<3>`.
    * @param[in] tri_quad - Triangle quadrature object to use for integration.
    */
    template <typename TriangleQuadratureType = GaussTriangleQuadrature<3>>
    RotVectorIdentityOp(const TriangleQuadratureType tri_quad = GaussTriangleQuadrature<3>()):
        tri_quad_(std::make_shared<TriangleQuadratureType> (tri_quad)) {};


    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    OperatorDof obs_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Returns the observation-triangle integrals required.
    * @return Integrals required, none of which this operator uses.
    */
    ObsTerms obs_terms() const override { return { false, false, false, false }; };


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
        ) const override;


    /**
    * @brief Assembles the computed integrals into the final operator values.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the source's local coordinate system.
    * @param[in] src_tri - Source triangle in its local coordinate system.
    * @param[in] obs_result - Integration result.
    * @return Operator values for each observation triangle edge and source triangle face.
    */
    EigMat<Complex> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        ) const override { return compute(k, obs_tri, src_tri); };


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<RotVectorIdentityOp> (*this); };


private:

    std::shared_ptr<TriangleQuadratureBase<3>> tri_quad_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/operators/gram.cpp"
#endif

#endif
