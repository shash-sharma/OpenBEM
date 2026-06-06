// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* RWG-based BEM operators defining incidence relationships between degrees of freedom.
*/

#ifndef BEM_RWG_OPS_INCIDENCE_H
#define BEM_RWG_OPS_INCIDENCE_H

#include "types.hpp"
#include "geometry/operations.hpp"
#include "rwg/function_space.hpp"
#include "rwg/operators/base.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgops
* @{
*/

/**
* @brief Class for computing the discrete divergence matrix for RWG functions.
* @details
* Computes
* \f[
* \int_{\mathrm{obs\_tri}} d\mathcal{S}\,h(\vec{r})\,
* \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla'\cdot\vec{f_n}(\vec{r}\,')
* \f]
* for each source triangle edge \f$ n \f$, when `obs_tri` and `src_tri` coincide, otherwise zeros.
* Here, the pulse function \f$ h(\vec{r}) \f$ and RWG function \f$ \vec{f_i}(\vec{r}) \f$ for edge
* \f$ i \f$ are considered normalized such that the output contains \f$ 1 \f$, \f$ -1 \f$, or \f$ 0 \f$.
* Columns of the output correspond to source edges.
*/
class DivergenceOp: public OperatorBase
{
public:

    /**
    * @brief Returns the number of degrees of freedom per triangle for the testing function space.
    * @return Number of observation degrees of freedom per triangle.
    */
    uint8_t obs_dof() const override { return 1; };


    /**
    * @brief Returns the number of degrees of freedom per triangle for the expansion function space.
    * @return Number of source degrees of freedom per triangle.
    */
    uint8_t src_dof() const override { return 3; };


    /**
    * @brief Computes an incidence matrix that can be used to take the divergence of RWG functions.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each source triangle edge.
    */
    EigMat<Complex> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) override
    {
        EigMatMN<Complex, 1, 3> result = EigMatMN<Complex, 1, 3>::Zero(1, 3);

        if (GeometryOps<3>::common_vertices(obs_tri, src_tri) < 3)
            return result;

        result = src_tri.edge_polarities();
        return result;
    };


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
        ) override { return compute(k, obs_tri, src_tri); };


    /**
    * @brief Returns a unique pointer to a deep copy of this object.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<OperatorBase> clone() const override
    { return std::make_unique<DivergenceOp> (*this); };

};

/**
* @}
*/

}

#endif
