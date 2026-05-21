// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* File for class defining bounding boxes.
*/

#ifndef BEM_BBOX_H
#define BEM_BBOX_H

#include "types.hpp"


namespace bem
{

/**
* \ingroup geom
* @{
*/

/**
* @brief Class for defining and managing bounding boxes.
* @tparam dim - Dimension of the bounding box.
*/
template <uint8_t dim>
class BoundingBox
{
public:

    /**
    * @brief Constructs a zero-sized bounding box.
    */
    BoundingBox(): bbox_(EigMatMN<Float, dim, 2>::Zero()) {};


    /**
    * @brief Constructs a bounding box from given minimum and maximum corner points.
    * @param[in] min - Minimum corner point of the bounding box.
    * @param[in] max - Maximum corner point of the bounding box.
    */
    BoundingBox(const EigColVecN<Float, dim>& min, const EigColVecN<Float, dim>& max)
    {
        bbox_.col(0) = min;
        bbox_.col(1) = max;
        return;
    };

    
    /**
    * @brief Returns the minimum and maximum corner points of the bounding box.
    * @return Read-only reference to the bounding box matrix.
    */
    const EigMatMN<Float, dim, 2>& operator() const { return bbox_; };


    /**
    * @brief Checks whether a given bounding box is encompassed by this one.
    * @param[in] other - Bounding box to check against.
    * @param[in] tol - Relative tolerance for comparison (optional).
    */
    bool encompasses(const BoundingBox& other, const Float tol = 1e-3) const
    {
        Float rel_tol = tol * diameter();
        return (bbox_.col(0).array() <= other.bbox().col(0).array() + rel_tol).all()
            && (bbox_.col(1).array() >= other.bbox().col(1).array() - rel_tol).all();
    };


    /**
    * @brief Checks whether a given bounding box overlaps with this one.
    * @param[in] other - Bounding box to check against.
    * @param[in] strict - If `true`, then touching does not count as an overlap (optional).
    */
    bool overlaps(const BoundingBox& other, const bool strict = false) const
    {
        Float rel_tol = strict ? 0 : 1e-3 * diameter();
        return !((bbox_.col(0).array() > other.bbox().col(1).array() + rel_tol).any()
            || (bbox_.col(1).array() < other.bbox().col(0).array() - rel_tol).any());
    };


    /**
    * @brief Extends the bounding box to include a given point.
    * @param[in] point - Point used to extend the bounding box.
    */
    void extend(const EigColVecN<Float, dim>& point)
    {
        bbox_.col(0) = bbox_.col(0).cwiseMin(point);
        bbox_.col(1) = bbox_.col(1).cwiseMax(point);
        return;
    };


    /**
    * @brief Returns the diameter of the bounding box.
    * @return Diameter of the bounding box.
    */
    Float diameter() const
    {
        return (bbox_.col(1) - bbox_.col(0)).norm();
    };


protected:

    EigMatMN<Float, dim, 2> bbox_;

};

/**
* @}
*/

}

#endif
