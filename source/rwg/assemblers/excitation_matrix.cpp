// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling excitation vectors and matrices for RWG-based BEM systems.
*/

#include "rwg/assemblers/excitation_matrix.hpp"

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "matrix/base.hpp"


namespace bem::rwg
{

void ExcitationAssembler::assemble(
    MatrixBase<Complex>& mat,
    ExcitationBase& exc,
    const Complex k
    )
{

    if (exc.obs_dof() == 3)
    {

        Index obs_num_edges = mesh_.num_edges();

        mat.resize(obs_num_edges, exc.num_excitations());
        mat.preallocate(obs_num_edges * exc.num_excitations());

        for (Index ii = 0; ii < elems_.size(); ++ii)
        {
            Index face = elems_[ii];
            Triangle<3> obs_tri = mesh_.elem_primitive(face);

            EigMatNX<Complex, 3> values = exc.compute(k, obs_tri);

            for (Index col = 0; col < values.cols(); col++)
            {
                for (uint8_t edge = 0; edge < 3; edge++)
                {
                    Index row = mesh_.elem_edges()(edge, face);
                    mat.add_value(row, col, values(edge, col));
                }
            }
        }

    }

    else if (exc.obs_dof() == 1)
    {

        Index obs_num_faces = mesh_.num_elems();

        mat.resize(obs_num_faces, exc.num_excitations());
        mat.preallocate(obs_num_faces * exc.num_excitations());

        for (Index ii = 0; ii < elems_.size(); ++ii)
        {
            Index face = elems_[ii];
            Triangle<3> obs_tri = mesh_.elem_primitive(face);

            EigMatNX<Complex, 1> values = exc.compute(k, obs_tri);

            for (Index rhs = 0; rhs < values.cols(); rhs++)
                mat.set_value(face, rhs, values(0, rhs));
        }

    }

    mat.assemble();

    return;

};

}

