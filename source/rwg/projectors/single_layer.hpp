// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based single-layer potential BEM projectors.
*/

#ifndef BEM_RWG_PROJ_SINGLE_LAYER_H
#define BEM_RWG_PROJ_SINGLE_LAYER_H

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"
#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/obs/quadrature.hpp"
#include "rwg/projectors/base.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgproj
* @{
*/

/**
* @brief Class for computing the vector single-layer potential projector.
*/
template <typename SrcIntegratorType = SrcStrategic<>>
class VectorSingleLayerProj: public ProjectorBase
{

    static_assert(
        std::is_base_of<SrcIntegratorBase, SrcIntegratorType>::value,
        "VectorSingleLayerProj: `SrcIntegratorType` must derive from `SrcIntegratorBase`"
        );

public:

    /**
    * @brief Constructs a `VectorSingleLayerProj` object with a specified integration object.
    * @param[in] src_integrator - Integration object for the source triangle (optional).
    */
    VectorSingleLayerProj(const SrcIntegratorType src_integrator = SrcStrategic<>()):
        src_integrator_(src_integrator) {};


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Computes the vector single-layer projector.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_points - Observation coordinates on which to project the generated field.
    * @param[in] src_tri - Source triangle.
    * @return Projected field components at each observation point, for each source triangle edge.
    * @details
    * Computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,\vec{f_n}(\vec{r}\,')
    * \f]
    * for each source triangle edge \f$ n \f$, where \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel,
    * and \f$ \vec{f_n}(\vec{r}) \f$ is the RWG function associated with edge \f$ n \f$.
    * Rows of the output matrix are ordered by field components and then by observation points, and columns
    * correspond to source edges. I.e., each column of the output looks like
    * \f$ (F_{x1}, F_{y1}, F_{z1}), (F_{x2}, F_{y2}, F_{z2}), \ldots \f$, where
    * \f$ (F_{xi}, F_{yi}, F_{zi}) \f$ are the components of the projected field \f$ \vec{F} \f$
    * defined at the observation point \f$ (x_i, y_i, z_i) \f$.
    */
    EigMat<Complex> compute(
        const Complex k,
        ConstEigRef<EigMatNX<Float, 3>> obs_points,
        const Triangle<3>& src_tri
        ) override;


private:

    SrcIntegratorType src_integrator_;

};


/**
* @brief Class for computing the scalar single-layer potential projector.
*/
template <typename SrcIntegratorType = SrcStrategic<>>
class ScalarSingleLayerProj: public ProjectorBase
{

    static_assert(
        std::is_base_of<SrcIntegratorBase, SrcIntegratorType>::value,
        "ScalarSingleLayerProj: `SrcIntegratorType` must derive from `SrcIntegratorBase`"
        );

public:

    /**
    * @brief Constructs a `ScalarSingleLayerProj` object with a specified integration object.
    * @param[in] src_integrator - Integration object for the source triangle (optional).
    */
    ScalarSingleLayerProj(const SrcIntegratorType src_integrator = SrcStrategic<>()):
        src_integrator_(src_integrator) {};


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::FACE; };


    /**
    * @brief Computes the scalar single-layer projector.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_points - Observation coordinates on which to project the generated field.
    * @param[in] src_tri - Source triangle.
    * @return Projected field at each observation point, for each source triangle edge.
    * @details
    * Computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,h(\vec{r}\,')
    * \f]
    * where \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, and \f$ h(\vec{r}) \f$ is the
    * pulse function associated with the source triangle. Rows of the output matrix correspond to
    * observation points.
    */
    EigMat<Complex> compute(
        const Complex k,
        ConstEigRef<EigMatNX<Float, 3>> obs_points,
        const Triangle<3>& src_tri
        ) override;


private:

    SrcIntegratorType src_integrator_;

};


/**
* @brief Class for computing the gradient of the scalar single-layer potential projector.
*/
template <typename SrcIntegratorType = SrcStrategic<>>
class GradScalarSingleLayerProj: public ProjectorBase
{

    static_assert(
        std::is_base_of<SrcIntegratorBase, SrcIntegratorType>::value,
        "GradScalarSingleLayerProj: `SrcIntegratorType` must derive from `SrcIntegratorBase`"
        );

public:

    /**
    * @brief Constructs a `GradScalarSingleLayerProj` object with a specified integration object.
    * @param[in] src_integrator - Integration object for the source triangle (optional).
    */
    GradScalarSingleLayerProj(const SrcIntegratorType src_integrator = SrcStrategic<>()):
        src_integrator_(src_integrator) {};


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::FACE; };


    /**
    * @brief Computes the gradient of the scalar single-layer projector.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_points - Observation coordinates on which to project the generated field.
    * @param[in] src_tri - Source triangle.
    * @return Projected field components at each observation point, for each source triangle edge.
    * @details
    * Computes
    * \f[
    * \nabla\int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\,h(\vec{r}\,')
    * \f]
    * where \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel, and \f$ h(\vec{r}) \f$ is the
    * pulse function associated with the source triangle. Rows of the output matrix are ordered by
    * field components and then by observation points, and columns correspond to source edges.
    * I.e., each column of the output looks like
    * \f$ (F_{x1}, F_{y1}, F_{z1}), (F_{x2}, F_{y2}, F_{z2}), \ldots \f$, where
    * \f$ (F_{xi}, F_{yi}, F_{zi}) \f$ are the components of the gradient of the projected scalar field,
    * defined at the observation point \f$ (x_i, y_i, z_i) \f$.
    */
    EigMat<Complex> compute(
        const Complex k,
        ConstEigRef<EigMatNX<Float, 3>> obs_points,
        const Triangle<3>& src_tri
        ) override;

private:

    SrcIntegratorType src_integrator_;

};


/**
* @brief Class for computing the vector hypersingular potential projector.
*/
template <typename SrcIntegratorType = SrcStrategic<>>
class VectorHypersingularProj: public ProjectorBase
{

    static_assert(
        std::is_base_of<SrcIntegratorBase, SrcIntegratorType>::value,
        "VectorHypersingularProj: `SrcIntegratorType` must derive from `SrcIntegratorBase`"
        );

public:

    /**
    * @brief Constructs a `VectorHypersingularProj` object with a specified integration object.
    * @param[in] src_integrator - Integration object for the source triangle (optional).
    */
    VectorHypersingularProj(const SrcIntegratorType src_integrator = SrcStrategic<>()):
        proj_g_(src_integrator), proj_gradg_(src_integrator) {};


    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    OperatorDof src_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Computes the vector hypersingular projector.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_points - Observation coordinates on which to project the generated field.
    * @param[in] src_tri - Source triangle.
    * @return Projected field components at each observation point, for each source triangle edge.
    * @details
    * Computes
    * \f{align*}{
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')\vec{f_n}(\vec{r}\,')\, +
    * \frac{1}{k^2}\nabla\nabla\cdot\int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')
    * \,\vec{f_n}(\vec{r}\,')
    * \f}
    * for each source triangle edge \f$ n \f$, where \f$ G(k, \vec{r}, \vec{r}\,') \f$ is a scalar kernel,
    * and \f$ \vec{f_n}(\vec{r}) \f$ is the RWG function associated with edge \f$ n \f$.
    * Rows of the output matrix are ordered by field components and then by observation points, and columns
    * correspond to source edges. I.e., each column of the output looks like
    * \f$ (F_{x1}, F_{y1}, F_{z1}), (F_{x2}, F_{y2}, F_{z2}), \ldots \f$, where
    * \f$ (F_{xi}, F_{yi}, F_{zi}) \f$ are the components of the projected field \f$ \vec{F} \f$
    * defined at the observation point \f$ (x_i, y_i, z_i) \f$.
    */
    EigMat<Complex> compute(
        const Complex k,
        ConstEigRef<EigMatNX<Float, 3>> obs_points,
        const Triangle<3>& src_tri
        ) override;


private:

    VectorSingleLayerProj<SrcIntegratorType> proj_g_;
    GradScalarSingleLayerProj<SrcIntegratorType> proj_gradg_;

};

/**
* @}
*/

}

#include "rwg/projectors/single_layer.tpp"

#endif
