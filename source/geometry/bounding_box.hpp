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
    const EigMatMN<Float, dim, 2>& operator()() const { return bbox_; };


    /**
    * @brief Returns the minimum corner points of the bounding box.
    * @return Read-only reference to the bounding box minimum corner points.
    */
    EigColVecN<Float, dim> min() const { return bbox_.col(0); };


    /**
    * @brief Returns the maximum corner points of the bounding box.
    * @return Read-only reference to the bounding box maximum corner points.
    */
    EigColVecN<Float, dim> max() const { return bbox_.col(1); };


    /**
    * @brief Checks whether a given bounding box is encompassed by this one.
    * @param[in] other - Bounding box to check against.
    * @param[in] tol - Relative tolerance for comparison (optional).
    */
    bool encompasses(const BoundingBox& other, const Float tol = 1e-3) const
    {
        Float rel_tol = tol * diameter();
        return (min().array() <= other.min().array() + rel_tol).all()
            && (max().array() >= other.max().array() - rel_tol).all();
    };


    /**
    * @brief Checks whether a given bounding box overlaps with this one.
    * @param[in] other - Bounding box to check against.
    * @param[in] strict - If `true`, then touching does not count as an overlap (optional).
    * @param[in] tol - Relative tolerance for comparison (optional).
    */
    bool overlaps(
        const BoundingBox& other,
        const bool strict = false,
        const Float tol = 1e-3
        ) const
    {
        Float rel_tol = strict ? 0 : tol * diameter();
        return !((min().array() > other.max().array() + rel_tol).any()
            || (max().array() < other.min().array() - rel_tol).any());
    };


    /**
    * @brief Returns the percent overlap with a given bounding box.
    * @param[in] other - Bounding box to check against.
    * @return Percent overlap.
    */
    Float percent_overlap(const BoundingBox& other) const
    {
        EigColVecN<Float, dim> maxs = min().cwiseMax(other.min());
        EigColVecN<Float, dim> mins = max().cwiseMin(other.max());

        if ((mins.array() < maxs.array()).any())
            return 0;

        Float intersection = (mins - maxs).prod();
        Float region1 = (max() - min()).prod();
        Float region2 = (other.max() - other.min()).prod();

        return intersection / (region1 + region2 - intersection) * 100.0;
    };


    /**
    * @brief Returns the signed edge-to-edge distance along each axis to a given bounding box.
    * @param[in] other - Bounding box to check against.
    * @return Signed distance to the other bounding box along each axis, where a negative distance
    * indicates an overlap.
    */
    EigColVecN<Float, dim> distance(const BoundingBox& other) const
    {
        EigColVecN<Float, dim> sep_r = other.min() - max();
        EigColVecN<Float, dim> sep_l = min() - other.max();

        EigColVecN<Float, dim> separated_dist = (sep_r.cwiseMax(sep_l)).cwiseMax(0.0);
        EigColVecN<Float, dim> overlap_dist = max().cwiseMin(other.max()) - min().cwiseMax(other.min());

        return (separated_dist.array() > 0).select(separated_dist, -overlap_dist);
    };


    /**
    * @brief Extends the bounding box to include a given point.
    * @param[in] point - Point used to extend the bounding box.
    */
    void extend(ConstEigRef<EigColVecN<Float, dim>> point)
    {
        bbox_.col(0) = min().cwiseMin(point);
        bbox_.col(1) = max().cwiseMax(point);
        return;
    };


    /**
    * @brief Returns the diameter of the bounding box.
    * @return Diameter of the bounding box.
    */
    Float diameter() const
    {
        return (max() - min()).norm();
    };


protected:

    EigMatMN<Float, dim, 2> bbox_;

};

/**
* @}
*/

}

#endif
