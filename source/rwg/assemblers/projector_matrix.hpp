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
class ProjectorBase;

/**
* \addtogroup assm
* @{
*/

/**
* @brief Class for generating projector matrices for RWG-based systems.
* @tparam obs_dim - Dimension of the projected fields.
*/
template <uint8_t obs_dim = 3>
class ProjectorAssembler: public ProjectorAssemblerBase<obs_dim>
{

    using base = ProjectorAssemblerBase<obs_dim>;

public:

    /**
    * @brief Constructs a `ProjectorAssembler` for given observation points and source mesh.
    * @param[in] obs_cloud - Observation point cloud on which to project fields.
    * @param[in] mesh - Source triangle mesh for which the projector matrix is to be assembled.
    * @param[in] elems - Source triangle indices for which the projector matrix is to be assembled.
    */
    ProjectorAssembler(
        const PointCloud<3>& obs_cloud,
        const TriangleMesh<3>& mesh,
        EigRowVec<Index> elems = EigRowVec<Index>::Zero(1, 0)
        ):
            obs_cloud_(obs_cloud),
            mesh_(mesh),
            elems_(elems)
    {
        if (elems_.cols() == 0)
            elems_ = EigRowVec<Index>::LinSpaced(mesh_.num_elems(), 0, mesh_.num_elems() - 1);
        return;
    };


    /**
    * @brief Assembles the projector matrix for edge-based RWG source functions.
    * @param[out] mat - Matrix to store the assembled projector coefficients, with columns corresponding
    * to source mesh edges, and rows corresponding to observation points.
    * @param[in] op - Projector object that computes the coefficients to assemble into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        ProjectorBase& op,
        const Complex k
        ) override;


protected:

    const PointCloud<3>& obs_cloud_;
    const TriangleMesh<3>& mesh_;
    EigRowVec<Index> elems_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/projector_matrix.cpp"
#endif

#endif
