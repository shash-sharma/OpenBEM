// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based single-layer potential BEM operators.
*/

#ifndef BEM_RWG_OPS_SINGLE_LAYER_H
#define BEM_RWG_OPS_SINGLE_LAYER_H

#include <memory>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"
#include "rwg/operators/base.hpp"
#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/strategic.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgops
* @{
*/

/**
* @brief Class for computing the vector single-layer potential operator.
* @details
* Computes
* \f[
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{f_m}(\vec{r})\cdot
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,\vec{f_n}(\vec{r}\,')
* \f]
* for each pair \f$ (m, n) \f$ of observation and source triangle edges, where
* \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, and \f$ \vec{f_i}(\vec{r}) \f$ is the RWG
* function associated with edge \f$ i \f$.
* Rows of the output matrix correspond to observation edges, and columns
* correspond to source edges.
*/
class VectorSingleLayerOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `VectorSingleLayerOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    VectorSingleLayerOp(const ObsIntegratorBase& obs_integrator = ObsStrategic()):
        obs_integrator_(obs_integrator.clone()) {};


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
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
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
    * @return Operator values for each pair of observation and source triangle edges.
    */
    EigMat<Complex> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        ) const override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<VectorSingleLayerOp> (*this); };


protected:

    std::shared_ptr<ObsIntegratorBase> obs_integrator_;

};


/**
* @brief Class for computing the rotationally-tested vector single-layer potential operator.
* @details
* Computes
* \f[
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\hat{n}\times\vec{f_m}(\vec{r})\cdot
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,\vec{f_n}(\vec{r}\,')
* \f]
* for each pair \f$ (m, n) \f$ of observation and source triangle edges, where
* \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, \f$ \vec{f_i}(\vec{r}) \f$ is the RWG
* function associated with edge \f$ i \f$, and \f$ \hat{n} \f$ is the unit normal vector
* associated with `obs_tri`. Rows of the output matrix correspond to observation edges,
* and columns correspond to source edges.
*/
class RotVectorSingleLayerOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `RotVectorSingleLayerOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    RotVectorSingleLayerOp(const ObsIntegratorBase& obs_integrator = ObsStrategic()):
        obs_integrator_(obs_integrator.clone()) {};


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
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
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
    * @return Operator values for each pair of observation and source triangle edges.
    */
    EigMat<Complex> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        ) const override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<RotVectorSingleLayerOp> (*this); };


protected:

    std::shared_ptr<ObsIntegratorBase> obs_integrator_;

};


/**
* @brief Class for computing the scalar single-layer potential operator.
* @details
* Computes
* \f[
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,h(\vec{r})\,
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,h(\vec{r}\,')
* \f]
* where \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, and \f$ h(\vec{r}) \f$ is a pulse
* function that is a non-zero constant inside the associated triangle, and zero outside.
*/
class ScalarSingleLayerOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `ScalarSingleLayerOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    ScalarSingleLayerOp(const ObsIntegratorBase& obs_integrator = ObsStrategic()):
        obs_integrator_(obs_integrator.clone()) {};


    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    OperatorDof obs_dof() const override { return OperatorDof::FACE; };


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::FACE; };


    /**
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
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
    * @return Operator value for the observation and source triangle faces.
    */
    EigMat<Complex> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        ) const override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<ScalarSingleLayerOp> (*this); };


protected:

    std::shared_ptr<ObsIntegratorBase> obs_integrator_;

};


/**
* @brief Class for computing the scalar single layer operator applied to the divergence of 
* a vector field.
* @details
* Computes
* \f{equation*}{
* -\int_{\mathrm{obs\_tri}} d\mathcal{S}\,p_m(\vec{r})
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,
* \nabla\cdot\vec{f_n}(\vec{r}\,')
* \f}
* for each pair \f$ (m, n) \f$ of observation triangle faces and source triangle edges, where
* \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, \f$ p_i(\vec{r}) \f$ is the pulse
* function associated with face \f$ i \f$, and \f$ \vec{f_i}(\vec{r}) \f$ is the RWG function
* associated with edge \f$ i \f$. Rows of the output matrix correspond to observation faces,
* and columns correspond to source edges.
*/
class ScalarSingleLayerDivOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `ScalarSingleLayerDivOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    ScalarSingleLayerDivOp(const ObsIntegratorBase& obs_integrator = ObsStrategic()):
        obs_integrator_(obs_integrator.clone()),
        op_g_(obs_integrator) {};


    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    OperatorDof obs_dof() const override { return OperatorDof::FACE; };


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
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
    * @return Operator values for each observation triangle face and source triangle edge.
    */
    EigMat<Complex> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
        ) const override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<ScalarSingleLayerDivOp> (*this); };


protected:

    std::shared_ptr<ObsIntegratorBase> obs_integrator_;
    ScalarSingleLayerOp op_g_;

};


/**
* @brief Class for computing the tangentially tested gradient of the scalar single layer operator.
* @details
* Computes
* \f{equation*}{
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{f_m}(\vec{r})\cdot\nabla
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,p_n(\vec{r}\,')
* \f}
* for each pair \f$ (m, n) \f$ of observation triangle edges and source triangle faces, where
* \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, \f$ \vec{f_i}(\vec{r}) \f$ is the RWG
* function associated with edge \f$ i \f$, and \f$ p_i(\vec{r}) \f$ is the pulse function
* associated with face \f$ i \f$. Rows of the output matrix correspond to observation edges,
* and columns correspond to source faces.
*/
class GradScalarSingleLayerOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `GradScalarSingleLayerOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    GradScalarSingleLayerOp(const ObsIntegratorBase& obs_integrator = ObsStrategic()):
        obs_integrator_(obs_integrator.clone()),
        op_g_(obs_integrator) {};


    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    OperatorDof obs_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::FACE; };


    /**
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
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
        ) const override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<GradScalarSingleLayerOp> (*this); };


protected:

    std::shared_ptr<ObsIntegratorBase> obs_integrator_;
    ScalarSingleLayerOp op_g_;

};


/**
* @brief Class for computing the rotationally-tested gradient of the scalar single-layer potential operator.
* @details
* Computes
* \f[
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\hat{n}\times\vec{f_m}(\vec{r})\cdot
* \nabla\int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,h(\vec{r}\,')
* \f]
* for observation triangle edge \f$ m \f$, where \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a
* scalar kernel, \f$ \vec{f_i}(\vec{r}) \f$ is the RWG function associated with edge
* \f$ i \f$, \f$ h(\vec{r}) \f$ is a pulse function that is a non-zero constant inside the
* associated triangle, and zero outside, and \f$ \hat{n} \f$ is the unit normal vector
* associated with `obs_tri`. Rows of the output matrix correspond to observation edges.
*/
class RotGradScalarSingleLayerOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `RotGradScalarSingleLayerOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    RotGradScalarSingleLayerOp(const ObsIntegratorBase& obs_integrator = ObsStrategic()):
        obs_integrator_(obs_integrator.clone()) {};


    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    OperatorDof obs_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::FACE; };


    /**
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
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
        ) const override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<RotGradScalarSingleLayerOp> (*this); };


protected:

    std::shared_ptr<ObsIntegratorBase> obs_integrator_;

};


/**
* @brief Class for computing the vector hypersingular operator.
* @details
* Computes
* \f{align*}{
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{f_m}(\vec{r})\cdot
* &\int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\vec{f_n}(\vec{r}\,')\, +\\
* \frac{1}{k^2}&\int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{f_m}(\vec{r})\cdot
* \nabla\nabla\cdot\int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,\vec{f_n}(\vec{r}\,')
* \f}
* for each pair \f$ (m, n) \f$ of observation and source triangle edges, where
* \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, and \f$ \vec{f_i}(\vec{r}) \f$ is the RWG
* function associated with edge \f$ i \f$. Rows of the output matrix correspond to observation
* edges, and columns correspond to source edges.
*/
class VectorHypersingularOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `VectorHypersingularOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    VectorHypersingularOp(const ObsIntegratorBase& obs_integrator = ObsStrategic()):
        obs_integrator_(obs_integrator.clone()),
        op_g_(obs_integrator),
        op_hessg_(obs_integrator) {};


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
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
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
        ) const override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<VectorHypersingularOp> (*this); };


protected:

    std::shared_ptr<ObsIntegratorBase> obs_integrator_;
    VectorSingleLayerOp op_g_;
    ScalarSingleLayerOp op_hessg_;

};


/**
* @brief Class for computing the rotationally-tested vector hypersingular operator.
* @details
* Computes
* \f{align*}{
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\hat{n}\times\vec{f_m}(\vec{r})\cdot
* &\int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\vec{f_n}(\vec{r}\,')\, +\\
* \frac{1}{k^2}&\int_{\mathrm{obs\_tri}} d\mathcal{S}\,\hat{n}\times\vec{f_m}(\vec{r})\cdot
* \nabla\nabla\cdot\int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,\vec{f_n}(\vec{r}\,')
* \f}
* for each pair \f$ (m, n) \f$ of observation and source triangle edges, where
* \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, \f$ \vec{f_i}(\vec{r}) \f$ is the RWG
* function associated with edge \f$ i \f$, and \f$ \hat{n} \f$ is the unit normal vector
* associated with `obs_tri`. Rows of the output matrix correspond to observation edges,
* and columns correspond to source edges.
*/
class RotVectorHypersingularOp: public OperatorBase
{
public:

    /**
    * @brief Constructs a `RotVectorHypersingularOp` object with a specified integration object.
    * @param[in] obs_integrator - Integration object for the observation triangle (optional).
    */
    RotVectorHypersingularOp(const ObsIntegratorBase& obs_integrator = ObsStrategic()):
        obs_integrator_(obs_integrator.clone()),
        op_g_(obs_integrator),
        op_hessg_(obs_integrator) {};


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
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
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
        ) const override;


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<RotVectorHypersingularOp> (*this); };


protected:

    std::shared_ptr<ObsIntegratorBase> obs_integrator_;
    RotVectorSingleLayerOp op_g_;
    RotGradScalarSingleLayerOp op_hessg_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/operators/single_layer.cpp"
#endif

#endif
