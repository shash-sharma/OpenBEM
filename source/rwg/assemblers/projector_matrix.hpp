// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling RWG-based BEM projector matrices.
*/

#ifndef BEM_RWG_PROJ_ASSEMBLER_H
#define BEM_RWG_PROJ_ASSEMBLER_H

#include <vector>

#include "types.hpp"
#include "rwg/assemblers/base.hpp"


namespace bem
{
// Forward declarations
template <typename T> class MatrixBase;
template <uint8_t dim> class PointCloud;
}


namespace bem::rwg
{

// Forward declarations
template <uint8_t src_num_dof> class ProjectorBase;

/**
* \addtogroup assm
* @{
*/

/**
* @brief Class for generating projector matrices for edge-based RWG source functions.
* @tparam obs_dim - Dimension of the projected fields.
*/
template <uint8_t obs_dim = 3>
class EdgeProjectorAssembler: public ProjectorAssemblerBase<obs_dim, 3>
{

    using base = ProjectorAssemblerBase<obs_dim, 3>;
    using base::base;

public:

    /**
    * @brief Assembles the projector matrix for edge-based RWG source functions.
    * @param[out] mat - Matrix to store the assembled projector coefficients, with columns corresponding
    * to source mesh edges, and rows corresponding to observation points.
    * @param[in] op - Projector object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        ProjectorBase<3>& op,
        const Complex k
        ) override;

};


/**
* @brief Class for generating projector matrices for face-based pulse source functions.
* @tparam obs_dim - Dimension of the projected fields.
*/
template <uint8_t obs_dim = 3>
class FaceProjectorAssembler: public ProjectorAssemblerBase<obs_dim, 1>
{

    using base = ProjectorAssemblerBase<obs_dim, 1>;
    using base::base;

public:

    /**
    * @brief Assembles the projector matrix for face-based pulse source functions.
    * @param[out] mat - Matrix to store the assembled projector coefficients, with columns corresponding
    * to source mesh faces, and rows corresponding to observation points.
    * @param[in] op - Projector object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        ProjectorBase<1>& op,
        const Complex k
        ) override;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/projector_matrix.cpp"
#endif

#endif
