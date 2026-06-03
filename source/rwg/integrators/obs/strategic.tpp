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

#ifndef BEM_RWG_OPINT_OBS_STRATEGIC_I
#define BEM_RWG_OPINT_OBS_STRATEGIC_I

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/operations.hpp"


namespace bem::rwg
{

template <typename ObsTriangleQuadratureType, typename SrcTriangleQuadratureType, typename LineQuadratureType>
ObsResult ObsStrategic<ObsTriangleQuadratureType, SrcTriangleQuadratureType, LineQuadratureType>::integrate(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<2>& src_tri
    )
{

    Float dist = (obs_tri.centroid() - src_tri.to_3d().centroid()).norm();
    Float dist_wvl = dist * std::real(k) / two_pi;
    Float longest_edge_wvl = src_tri.longest_edge_length() * std::real(k) / two_pi;

    Float skin_depth = -one / std::imag(k);
    if (std::isfinite(skin_depth) && settings_.threshold_skin_depths > float_eps)
        if (dist > std::max(settings_.threshold_skin_depths * skin_depth, 2 * src_tri.longest_edge_length()))
            return ObsResult();


    Float threshold_wvl = settings_.threshold_wvl_singularity < 0 ?
        (Float)0.1 : settings_.threshold_wvl_singularity;

    Float threshold_dist = settings_.threshold_dist_singularity < 0 ?
        src_tri.longest_edge_length() * 5 : settings_.threshold_dist_singularity;

    Float threshold_line_int = settings_.threshold_length_line_int < 0 ?
        one : settings_.threshold_length_line_int;


    bool line_integration = threshold_line_int >= 0 &&
        longest_edge_wvl >= threshold_line_int;

    bool singularity_subtraction = (
        GeometryOps<3>::common_vertices(obs_tri, src_tri.to_3d()) > 0 ||
        dist <= threshold_dist ||
        dist_wvl <= threshold_wvl
        );

    bool singularity_separation = false;

    if (line_integration)
    {
        line_.set_compute_terms(base::compute_g_term_, base::compute_rs_g_terms_, base::compute_grad_g_terms_, base::compute_rot_grad_g_terms_);
        return line_.integrate(k, obs_tri, src_tri);
    }
    else if (singularity_subtraction)
    {
        sthgf_.set_compute_terms(base::compute_g_term_, base::compute_rs_g_terms_, base::compute_grad_g_terms_, base::compute_rot_grad_g_terms_);
        return sthgf_.integrate(k, obs_tri, src_tri);
    }
    else if (singularity_separation)
    {
        shgf_.set_compute_terms(base::compute_g_term_, base::compute_rs_g_terms_, base::compute_grad_g_terms_, base::compute_rot_grad_g_terms_);
        return shgf_.integrate(k, obs_tri, src_tri);
    }
    else
    {
        hgf_.set_compute_terms(base::compute_g_term_, base::compute_rs_g_terms_, base::compute_grad_g_terms_, base::compute_rot_grad_g_terms_);
        return hgf_.integrate(k, obs_tri, src_tri);
    }

};

}

#endif
