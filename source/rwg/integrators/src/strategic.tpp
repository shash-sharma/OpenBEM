// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Strategic auto-integration over the source triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_STRATEGIC_I
#define BEM_RWG_OPINT_SRC_STRATEGIC_I

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

template <typename TriangleQuadratureType, typename LineQuadratureType>
SrcResult SrcStrategic<TriangleQuadratureType, LineQuadratureType>::integrate(
    const Complex k,
    const Triangle<2>& src_tri,
    ConstEigRef<EigMatNX<Float, 3>> r_obs
    )
{

    EigColVecN<Float, 3> src_tri_centroid = EigColVecN<Float, 3>::Zero(3, 1);
    src_tri_centroid.topRows(2) = src_tri.centroid();
    EigRowVec<Float> r_obs_dist = (r_obs.colwise() - src_tri_centroid).colwise().norm();

    Float min_dist = r_obs_dist.minCoeff();
    Float min_dist_wvl = min_dist * std::real(k) / two_pi;
    Float longest_edge_wvl = src_tri.longest_edge_length() * std::real(k) / two_pi;

    Float skin_depth = -one / std::imag(k);
    if (std::isfinite(skin_depth) && settings_.threshold_skin_depths > float_eps)
        if (min_dist > std::max(settings_.threshold_skin_depths * skin_depth, 2 * src_tri.longest_edge_length()))
            return src_hgf_.zeros(k, src_tri, r_obs);

    Float threshold_wvl = std::max(
        settings_.threshold_wvl_singularity,
        (Float)0.1
    );
    Float threshold_dist = std::max(
        settings_.threshold_dist_singularity,
        src_tri.longest_edge_length() * 5
    );
    Float threshold_line_int = std::min(
        settings_.threshold_length_line_int,
        one
    );

    bool line_integration = threshold_line_int >= 0 && longest_edge_wvl >= threshold_line_int;
    bool singularity_subtraction = (min_dist <= threshold_dist || min_dist_wvl <= threshold_wvl);
    bool singularity_separation = false;

    if (line_integration)
    {
        src_line_.set_compute_terms(base::compute_g_terms_, base::compute_grad_g_terms_);
        return src_line_.integrate(k, src_tri, r_obs);
    }
    else if (singularity_subtraction)
    {
        src_sthgf_.set_compute_terms(base::compute_g_terms_, base::compute_grad_g_terms_);
        return src_sthgf_.integrate(k, src_tri, r_obs);
    }
    else if (singularity_separation)
    {
        src_shgf_.set_compute_terms(base::compute_g_terms_, base::compute_grad_g_terms_);
        return src_shgf_.integrate(k, src_tri, r_obs);
    }
    else
    {
        src_hgf_.set_compute_terms(base::compute_g_terms_, base::compute_grad_g_terms_);
        return src_hgf_.integrate(k, src_tri, r_obs);
    }

};

}

#endif
