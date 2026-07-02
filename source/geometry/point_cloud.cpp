// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Point cloud functionality.
*/

#include "geometry/point_cloud.hpp"

#include <vector>

#include "types.hpp"
#include "geometry/operations.hpp"


namespace bem
{

template <uint8_t dim>
void PointCloud<dim>::set_line_data(
    ConstEigRef<EigColVecN<Float, dim>> start,
    ConstEigRef<EigColVecN<Float, dim>> stop,
    const Index num_pts
    )
{
    EigRowVec<Float> ref_points = EigRowVec<Float>::LinSpaced(num_pts, 0.0, 1.0);
    EigColVecN<Float, dim> diff = stop - start;
    set_data((diff * ref_points).colwise() + start);
    return;
}


template <uint8_t dim>
void PointCloud<dim>::set_block_data(
    ConstEigRef<EigColVecN<Float, dim>> start,
    ConstEigRef<EigColVecN<Float, dim>> stop,
    ConstEigRef<EigColVecN<Index, dim>> num_pts
    )
{
    EigMatNX<Float, dim> points = EigMatNX<Float, dim>::Zero(dim, num_pts.prod());
    EigColVecN<Float, dim> diff = stop - start;
    std::vector<EigRowVec<Float>> dim_points (dim);
    std::vector<Index> counters (dim);

    for (uint8_t ii = 0; ii < dim; ++ii)
    {
        EigRowVec<Float> ref_points = EigRowVec<Float>::LinSpaced(num_pts[ii], 0.0, 1.0);
        dim_points[ii] = (diff[ii] * ref_points).array() + start[ii];
    }

    Index idx = 0;
    while (counters[0] != num_pts[0])
    {
        for (uint8_t ii = 0; ii < dim; ++ii)
            points(ii, idx) = dim_points[ii][counters[ii]];
        idx++;

        ++counters[dim - 1];
        for (uint8_t ii = dim - 1; (ii > 0) && (counters[ii] == num_pts[ii]); --ii)
        {
            counters[ii] = 0;
            ++counters[ii - 1];
        }
    }

    set_data(points);

    return;
}


template <uint8_t dim>
void PointCloud<dim>::set_polar_data(
    ConstEigRef<EigColVecN<Float, dim>> start,
    ConstEigRef<EigColVecN<Float, dim>> stop,
    ConstEigRef<EigColVecN<Float, dim>> center,
    ConstEigRef<EigColVecN<Index, dim>> num_pts
    )
{
    set_block_data(start, stop, num_pts);
    points_ = GeometryOps<dim>::polar_to_cartesian(points_).colwise() + center;
    return;
}


template <uint8_t dim>
void PointCloud<dim>::set_data(ConstEigRef<EigMatNX<Float, dim>> points)
{
    points_ = points;
    return;
};


template <uint8_t dim>
void PointCloud<dim>::clear_data()
{
    points_.resize(dim, 0);
    return;
};


template class PointCloud<2>;
template class PointCloud<3>;

}
