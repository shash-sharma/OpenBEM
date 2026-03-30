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
class DivRwgOp: public OperatorBase<1, 3>
{
public:

    /**
    * @brief Computes an incidence matrix that can be used to take the divergence of RWG functions.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each source triangle edge.
    */
    EigMatMN<Complex, 1, 3> compute(
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

};

/**
* @}
*/

}

#endif
