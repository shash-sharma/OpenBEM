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

#include "rwg/assemblers/projector_assembler.hpp"

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/point_cloud.hpp"
#include "matrix/base.hpp"


namespace bem::rwg
{

template <uint8_t obs_dim>
void ProjectorAssembler<obs_dim>::assemble(
    MatrixBase<Complex>& mat,
    ProjectorBase& op,
    const Complex k
    )
{

    if (op.src_dof() == DofSpace::EDGE)
    {

        mat.resize(obs_cloud_.num_points() * obs_dim, mesh_.num_edges());
        mat.preallocate(
            obs_cloud_.num_points() * obs_dim * faces_.size() * EDGE_ELEM_RATIO
            );

        for (Index ii = 0; ii < faces_.size(); ++ii)
        {
            Triangle<3> src_tri = mesh_.face_primitive(faces_[ii]);

            EigMatXN<Complex, 3> values = op.compute(k, obs_cloud_.points(), src_tri);

            for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
            {
                Index col = mesh_.face_edges()(src_edge, faces_[ii]);

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

    else if (op.src_dof() == DofSpace::FACE)
    {

        mat.resize(obs_cloud_.num_points() * obs_dim, mesh_.num_faces());
        mat.preallocate(obs_cloud_.num_points() * obs_dim * faces_.size());

        for (Index ii = 0; ii < faces_.size(); ++ii)
        {
            Triangle<3> src_tri = mesh_.face_primitive(faces_[ii]);

            EigMatXN<Complex, 1> values = op.compute(k, obs_cloud_.points(), src_tri);

            for (Index obs_point = 0; obs_point < obs_cloud_.num_points(); ++obs_point)
            {
                for (uint8_t obs_dim_idx = 0; obs_dim_idx < obs_dim; ++obs_dim_idx)
                {
                    Index row = obs_dim_idx + obs_point * obs_dim;
                    mat.add_value(row, faces_[ii], values(row, 0));
                }
            }
        }

    }

    mat.assemble();

    return;

};

template class ProjectorAssembler<1>;
template class ProjectorAssembler<3>;

}

