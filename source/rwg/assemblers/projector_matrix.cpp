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

#include "rwg/assemblers/projector_matrix.hpp"

#include <vector>
#include <stdexcept>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/point_cloud.hpp"
#include "matrix/base.hpp"


namespace bem::rwg
{

template <uint8_t obs_dim>
void EdgeProjectorAssembler<obs_dim>::assemble(
    MatrixBase<Complex>& mat,
    ProjectorBase<3>& op,
    const Complex k
    )
{
    mat.resize(base::obs_cloud_.num_points() * obs_dim, base::src_mesh_.num_edges());
    mat.preallocate(base::obs_cloud_.num_points() * obs_dim * base::elems_.size() * 3);

    for (Index ii = 0; ii < base::elems_.size(); ++ii)
    {
        Triangle<3> src_tri = base::src_mesh_.elem_primitive(base::elems_[ii]);

        EigMatXN<Complex, 3> values = op.compute(k, base::obs_cloud_.points(), src_tri);

        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = base::src_mesh_.elem_edges()(src_edge, base::elems_[ii]);

            for (Index obs_point = 0; obs_point < base::obs_cloud_.num_points(); ++obs_point)
            {
                for (uint8_t obs_dim_idx = 0; obs_dim_idx < obs_dim; ++obs_dim_idx)
                {
                    Index row = obs_dim_idx + obs_point * obs_dim;
                    mat.add_value(row, col, values(row, src_edge));
                }
            }
        }
    }

    mat.assemble();

    return;
};


template <uint8_t obs_dim>
void FaceProjectorAssembler<obs_dim>::assemble(
    MatrixBase<Complex>& mat,
    ProjectorBase<1>& op,
    const Complex k
    )
{
    mat.resize(base::obs_cloud_.num_points() * obs_dim, base::src_mesh_.num_elems());
    mat.preallocate(base::obs_cloud_.num_points() * obs_dim * base::elems_.size());

    for (Index ii = 0; ii < base::elems_.size(); ++ii)
    {
        Triangle<3> src_tri = base::src_mesh_.elem_primitive(base::elems_[ii]);

        EigMatXN<Complex, 1> values = op.compute(k, base::obs_cloud_.points(), src_tri);

        for (Index obs_point = 0; obs_point < base::obs_cloud_.num_points(); ++obs_point)
        {
            for (uint8_t obs_dim_idx = 0; obs_dim_idx < obs_dim; ++obs_dim_idx)
            {
                Index row = obs_dim_idx + obs_point * obs_dim;
                mat.add_value(row, base::elems_[ii], values(row, 0));
            }
        }
    }

    mat.assemble();

    return;
};


template class EdgeProjectorAssembler<1>;
template class EdgeProjectorAssembler<2>;
template class EdgeProjectorAssembler<3>;

template class FaceProjectorAssembler<1>;
template class FaceProjectorAssembler<2>;
template class FaceProjectorAssembler<3>;

}
