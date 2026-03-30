// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Functionality for setting infinitesimal gap excitation coefficients for RWG-based BEM systems.
*/

#include "rwg/excitations/inf_gap.hpp"

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/primitives/edge.hpp"


namespace bem::rwg
{

EigMatNX<Complex, 3> InfinitesimalGap::compute(const Complex k, const Triangle<3>& obs_tri)
{
    EigMatNX<Complex, 3> result = EigMatNX<Complex, 3>::Zero(3, num_excitations());

    for (Index rhs = 0; rhs < num_excitations(); ++rhs)
    {
        for (Index seg = 0; seg < segments_[rhs].cols() - 1; ++seg)
        {
            Edge<3> line_segment (
                segments_[rhs].col(seg),
                segments_[rhs].col((seg + 1))
                );

            for (uint8_t ii = 0; ii < 3; ++ii)
            {
                if (
                    line_segment.point_on_edge(obs_tri.v(ii), tol_) &&
                    line_segment.point_on_edge(obs_tri.v((ii + 1) % 3), tol_)
                    )
                {
                    EigColVecN<Float, 3> edge_vec = obs_tri.v((ii + 1) % 3) - obs_tri.v(ii);
                    Float sign = (edge_vec).dot(line_segment.v(1) - line_segment.v(0));
                    sign /= std::abs(sign);
                    result(ii, rhs) = sign * obs_tri.edge_polarities(ii) * amp_[rhs] / two;
                }
            }
        }
    }

    return result;
};

}

