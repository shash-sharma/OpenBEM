// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Strategic auto-integration over the observation and source triangles for RWG-based BEM operators.
*/

#include "rwg/integrators/obs/strategic.hpp"

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/operations.hpp"


namespace bem::rwg
{

ObsResult ObsStrategic::integrate(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<2>& src_tri,
    const bool g_term,
    const bool rs_g_terms,
    const bool grad_g_terms,
    const bool rot_grad_g_terms
    )
{

    Float dist = (obs_tri.centroid() - src_tri.to_3d().centroid()).norm();
    Float longest_edge_wvl = src_tri.longest_edge_length() * std::real(k) / two_pi;

    Float skin_depth = -one / std::imag(k);
    if (std::isfinite(skin_depth) && settings_.threshold_skin_depths > float_eps)
        if (dist > std::max(settings_.threshold_skin_depths * skin_depth, 2 * src_tri.longest_edge_length()))
            return ObsResult();

    Float threshold_dist = settings_.threshold_dist_singularity < 0 ?
        src_tri.longest_edge_length() * 2 : settings_.threshold_dist_singularity;

    Float threshold_line_int = settings_.threshold_length_line_int < 0 ?
        one : settings_.threshold_length_line_int;

    bool line_integration = threshold_line_int >= 0 &&
        longest_edge_wvl >= threshold_line_int;

    bool singularity_subtraction = (
        GeometryOps<3>::common_vertices(obs_tri, src_tri.to_3d()) > 0 ||
        dist <= threshold_dist
        );

    bool singularity_separation = false;

    if (line_integration)
        return line_.integrate(
            k, obs_tri, src_tri, g_term, rs_g_terms, grad_g_terms, rot_grad_g_terms
            );

    else if (singularity_subtraction)
        return sthgf_.integrate(k, obs_tri, src_tri, g_term, rs_g_terms, grad_g_terms, rot_grad_g_terms);

    else if (singularity_separation)
        return shgf_.integrate(k, obs_tri, src_tri, g_term, rs_g_terms, grad_g_terms, rot_grad_g_terms);

    else
        return hgf_.integrate(k, obs_tri, src_tri, g_term, rs_g_terms, grad_g_terms, rot_grad_g_terms);

};

}

