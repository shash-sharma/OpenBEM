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

#ifndef BEM_RWG_PROJ_ASSEMBLER_I
#define BEM_RWG_PROJ_ASSEMBLER_I

#include "rwg/assemblers/projector_matrix.hpp"

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/point_cloud.hpp"
#include "matrix/base.hpp"


namespace bem::rwg
{

template <typename ExpansionSpace, uint8_t obs_dim>
void ProjectorAssembler<ExpansionSpace, obs_dim>::assemble(
    MatrixBase<Complex>& mat,
    ProjectorBase<ExpansionSpace::dof>& op,
    const Complex k
    )
{

    if constexpr (ExpansionSpace::dof == 3)
    {

        mat.resize(obs_cloud_.num_points() * obs_dim, mesh_.num_edges());
        mat.preallocate(
            obs_cloud_.num_points() * obs_dim * elems_.size() * EDGE_ELEM_RATIO
            );

        for (Index ii = 0; ii < elems_.size(); ++ii)
        {
            Triangle<3> src_tri = mesh_.elem_primitive(elems_[ii]);

            EigMatXN<Complex, 3> values = op.compute(k, obs_cloud_.points(), src_tri);

            for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
            {
                Index col = mesh_.elem_edges()(src_edge, elems_[ii]);

                for (Index obs_point = 0; obs_point < obs_cloud_.num_points(); ++obs_point)
                {
                    for (uint8_t obs_dim_idx = 0; obs_dim_idx < obs_dim; ++obs_dim_idx)
                    {
                        Index row = obs_dim_idx + obs_point * obs_dim;
                        mat.add_value(row, col, values(row, src_edge));
                    }
                }
            }
        }

    }

    else if constexpr (ExpansionSpace::dof == 1)
    {

        mat.resize(obs_cloud_.num_points() * obs_dim, mesh_.num_elems());
        mat.preallocate(obs_cloud_.num_points() * obs_dim * elems_.size());

        for (Index ii = 0; ii < elems_.size(); ++ii)
        {
            Triangle<3> src_tri = mesh_.elem_primitive(elems_[ii]);

            EigMatXN<Complex, 1> values = op.compute(k, obs_cloud_.points(), src_tri);

            for (Index obs_point = 0; obs_point < obs_cloud_.num_points(); ++obs_point)
            {
                for (uint8_t obs_dim_idx = 0; obs_dim_idx < obs_dim; ++obs_dim_idx)
                {
                    Index row = obs_dim_idx + obs_point * obs_dim;
                    mat.add_value(row, elems_[ii], values(row, 0));
                }
            }
        }

    }

    mat.assemble();

    return;

};

}

#endif
