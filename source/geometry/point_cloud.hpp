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

#ifndef GEOM_POINT_CLOUD_H
#define GEOM_POINT_CLOUD_H

#include <vector>

#include "types.hpp"


namespace bem
{

/**
* \ingroup geom
* @{
*/

/**
* @brief Point cloud class.
* @tparam dim - The dimension of the point cloud (1, 2 or 3).
*/
template <uint8_t dim>
class PointCloud
{

    static_assert((dim == 1 || dim == 2 || dim == 3), "PointCloud: `dim` must be 1, 2 or 3.");

public:

    /**
    * @brief Constructs an empty `PointCloud`.
    */
    PointCloud() {};


    /**
    * @brief Constructs a `PointCloud` with given point data.
    * @param[in] points - Coordinates of the points in the cloud.
    */
    PointCloud(ConstEigRef<EigMatNX<Float, dim>> points)
    {
        set_data(points);
        return;
    }


    /** @name Setters. */
    /**@{*/

    /**
    * @brief Sets `PointCloud` data as points along a straight line.
    * @param[in] start - Starting point coordinates.
    * @param[in] stop - Ending point coordinates.
    * @param[in] num_pts - Number of points in the cloud.
    */
    void set_line_data(
        ConstEigRef<EigColVecN<Float, dim>> start,
        ConstEigRef<EigColVecN<Float, dim>> stop,
        const Index num_pts
        );


    /**
    * @brief Sets `PointCloud` data as a block of points.
    * @param[in] start - Coordinates of the minimum-value corner of the block.
    * @param[in] stop - Coordinates of the maximum-value corner of the block.
    * @param[in] num_pts - Number of points in the block along each dimension.
    */
    void set_block_data(
        ConstEigRef<EigColVecN<Float, dim>> start,
        ConstEigRef<EigColVecN<Float, dim>> stop,
        ConstEigRef<EigColVecN<Index, dim>> num_pts
        );


    /**
    * @brief Sets `PointCloud` data as points of a polar grid.
    * @param[in] start - Starting point coordinates (radius, azimuth, elevation).
    * @param[in] stop - Ending point coordinates (radius, azimuth, elevation).
    * @param[in] center - Center point coordinates (x, y, z).
    * @param[in] num_pts - Number of points in the polar grid along each dimension.
    */
    void set_polar_data(
        ConstEigRef<EigColVecN<Float, dim>> start,
        ConstEigRef<EigColVecN<Float, dim>> stop,
        ConstEigRef<EigColVecN<Float, dim>> center,
        ConstEigRef<EigColVecN<Index, dim>> num_pts
        );


    /**
    * @brief Sets the point cloud data.
    * @param[in] points - Coordinates of the points in the cloud.
    */
    void set_data(ConstEigRef<EigMatNX<Float, dim>> points);

    /**@}*/


    /**
    * @brief Clears the point cloud data.
    */
    void clear_data();


    /**
    * @brief Returns the coordinates of the points in the cloud.
    * @return Point coordinates.
    */
    const EigMatNX<Float, dim>& points() const
    { return points_; };


    /**
    * @brief Returns the coordinates of the point at a specific index in the cloud.
    * @param[in] idx - Index of the point.
    * @return Coordinates of the specified point.
    */
    EigColVecN<Float, dim> points(Index idx) const
    { return points_.col(idx); };


    /**
    * @brief Returns the number of points in the cloud.
    * @return Number of points.
    */
    Index num_points() const
    { return points_.cols(); };


protected:

    EigMatNX<Float, dim> points_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "geometry/point_cloud.cpp"
#endif

#endif
