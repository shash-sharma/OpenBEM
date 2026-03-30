// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Triangle primitive.
*/

#ifndef GEOM_TRIANGLE_H
#define GEOM_TRIANGLE_H

#include <stdexcept>
#include <array>
#include <cassert>

#include "types.hpp"
#include "constants.hpp"


namespace bem
{

const Float TRIANGLE_DEFAULT_TOL = 1.0e-6;

/**
* \ingroup prim
* @{
*/

/**
* @brief Triangle primitive class.
* @tparam dim - The dimension of the triangle (2 or 3).
*/
template <uint8_t dim>
class Triangle
{
public:

    /**
    * @brief Constructs a default reference `Triangle` object
    */
    Triangle()
    {
        set_v(reference_triangle().v());
        return;
    };


    /**
    * @brief Constructs a `Triangle` with given vertices.
    * @param[in] v - Vertices of the triangle.
    * @param[in] edge_polarities - Polarity of each edge of the triangle (optional).
    */
    Triangle(
        ConstEigRef<EigMatMN<Float, dim, 3>> v,
        EigRowVecN<Float, 3> edge_polarities = EigRowVecN<Float, 3>::Constant(1, 3, 1)
        )
    {
        set_v(v, edge_polarities);
        return;
    };


    /**
    * @brief Constructs a `Triangle` with given vertices.
    * @param[in] v_array - Array of vertices of the triangle.
    * @param[in] edge_polarities - Polarity of each edge of the triangle (optional).
    */
    Triangle(
        std::array<EigColVecN<Float, dim>, 3> v_array,
        EigRowVecN<Float, 3> edge_polarities = EigRowVecN<Float, 3>::Constant(1, 3, 1)
        )
    {
        EigMatMN<Float, dim, 3> v;
        for (uint8_t ii = 0; ii < 3; ++ii)
            v.col(ii) = v_array[ii];
        set_v(v, edge_polarities);
        return;
    };


    /**
    * @brief Sets the vertices of this `Triangle`.
    * @param[in] v - Vertices of the triangle.
    * @param[in] edge_polarities - Polarity of each edge of the triangle (optional).
    */
    void set_v(
        ConstEigRef<EigMatMN<Float, dim, 3>> v,
        EigRowVecN<Float, 3> edge_polarities = EigRowVecN<Float, 3>::Constant(1, 3, 1)
        );


    /**
    * @brief Returns the vertices of this `Triangle`.
    * @return Vertices of the triangle.
    */
    const EigMatMN<Float, dim, 3>& v() const
    {
        return v_;
    };


    /**
    * @brief Returns the vertex of this `Triangle` at a given index.
    * @param[in] idx - Index of the desired vertex (0, 1, or 2).
    * @return Vertex of the triangle at the given index.
    */
    EigColVecN<Float, dim> v(const uint8_t idx) const
    {
        if (idx >= 3)
            throw std::out_of_range("Triangle::v(): vertex index out of bounds.");
        return v_.col(idx);
    };


    /**
    * @brief Returns the edge polarities of this `Triangle`.
    * @return Edge polarities of the triangle.
    */
    const EigRowVecN<Float, 3>& edge_polarities() const
    {
        return edge_polarities_;
    };


    /**
    * @brief Returns the edge polarity of this `Triangle` at a given index.
    * @param[in] idx - Index of the desired edge (0, 1, or 2).
    * @return Edge polarity of the triangle at the given index.
    */
    Float edge_polarities(const uint8_t idx) const
    {
        if (idx >= 3)
            throw std::out_of_range("Triangle::edge_polarities(): edge index out of bounds.");
        return edge_polarities_[idx];
    };


    /**
    * @brief Reverses the orientation of this `Triangle`.
    */
    void reverse()
    {
        set_v(v_.rowwise().reverse(), edge_polarities_.reverse());
        return;
    };


    /**
    * @brief Returns the area of this `Triangle`.
    * @return Area of the triangle.
    */
    Float area() const
    { return area_; };


    /**
    * @brief Returns the unit normal vector for this `Triangle`.
    * @return Unit normal vector of the triangle.
    */
    EigColVecN<Float, 3> normal() const
    { return normal_; };


    /**
    * @brief Returns the centroid of this `Triangle`.
    * @return Centroid of the triangle.
    */
    EigColVecN<Float, dim> centroid() const
    { return centroid_; };


    /**
    * @brief Returns the mean edge length of this `Triangle`.
    * @return Mean edge length of the triangle.
    */
    Float mean_edge_length() const
    {
        Float len = 0.0;
        for (uint8_t ii = 0; ii < 3; ii++)
            len += (v_.col(ii) - v_.col((ii + 1) % 3)).norm();
        return len / (Float)3;
    };


    /**
    * @brief Returns an equivalent triangle with coordinates in a local 2D system,
    * with the local origin at this triangle's first vertex, and with its edge
    * connecting the first two vertices lying along the local x axis.
    * @return 2D `Triangle` with coordinates in a local 2D system.
    */
    Triangle<2> to_2d() const;


    /**
    * @brief Adds a 0-valued third dimension if this `Triangle` is in 2D, otherwise just returns
    * a copy of this `Triangle`.
    * @return 3D `Triangle`.
    */
    Triangle<3> to_3d(const Float z = 0.0) const;


    /**
    * @brief Returns the local origin of this `Triangle` in the global coordinate system.
    * The local origin is just the first vertex of the triangle.
    * @return Local origin of the triangle in the global coordinate system (i.e., its first vertex).
    */
    EigColVecN<Float, dim> local_origin() const
    {
        return v_.col(0);
    };


    /**
    * @brief Returns the unit vectors of a local coordinate system, defined in the global system,
    * such that the edge connecting the first two vertices lie along the local x axis, and the
    * `Triangle` lies in the local xy plane.
    * @return Local coordinate basis of the triangle in the global coordinate system.
    */
    EigMatMN<Float, dim, dim> local_coordinate_basis() const;


    /**
    * @brief Returns the projection of given points on to the triangle's plane, along with the normal
    * distance to the plane. For a 2D `Triangle`, its third (z) dimension is assumed to be 0.
    * @param[out] r_proj - Projected points.
    * @param[out] d - Perpendicular distances of the original points to the triangle's plane.
    * @param[in] r - Points to be projected onto the triangle's plane.
    * @param[in] ref_idx - Index of the reference vertex to be used for the projection (optional).
    */
    void get_plane_projection(
        EigMatNX<Float, dim>& r_proj,
        EigRowVec<Float>& d,
        ConstEigRef<EigMatNX<Float, 3>> r,
        uint8_t ref_idx = 0
        ) const;


    /**
    * @brief Returns a reference triangle in the specified dimension.
    * @return Reference edge with vertices (0, 0, ...), (1, 0, ...), (0, 1, ...).
    */
    static Triangle<dim> reference_triangle();


    /**
    * @brief Returns the lengths of each edge of this `Triangle`.
    * @return Lengths of the triangle's edges.
    */
    EigColVecN<Float, 3> edge_lengths() const
    {
        EigColVecN<Float, 3> lengths;
        for (uint8_t ii = 0; ii < 3; ii++)
            lengths[ii] = (v_.col((ii + 1) % 3) - v_.col((ii) % 3)).norm();
        return lengths;
    };


    /**
    * @brief Returns the index associated with the longest edge of this `Triangle`.
    * @return Index of the triangle's longest edge.
    */
    uint8_t longest_edge_index() const
    {
        uint8_t result;
        edge_lengths().maxCoeff(&result);
        assert(result < 3 && "Triangle::longest_edge(): Result out of range.");
        return result;
    };


    /**
    * @brief Returns the index associated with the shortest edge of this `Triangle`.
    * @return Index of the triangle's shortest edge.
    */
    uint8_t shortest_edge_index() const
    {
        uint8_t result;
        edge_lengths().minCoeff(&result);
        assert(result < 3 && "Triangle::shortest_edge(): Result out of range.");
        return result;
    };


    /**
    * @brief Length of the longest edge of this `Triangle`.
    * @return Length of the triangle's longest edge.
    */
    Float longest_edge_length() const
    {
        return edge_lengths().maxCoeff();
    };


    /**
    * @brief Length of the shortest edge of this `Triangle`.
    * @return Length of the triangle's shortest edge.
    */
    Float shortest_edge_length() const
    {
        return edge_lengths().minCoeff();
    };


    /**
    * @brief Returns barycentric coordinates of given points lying in the `Triangle`'s plane.
    * @param[in] p - Points in the triangle's plane for which to compute barycentric coordinates.
    * @return Barycentric coordinates of the points.
    */
    EigMatNX<Float, 3> barycentric_coords(ConstEigRef<EigMatNX<Float, dim>> p) const;


    /**
    * @brief Returns flags that indicate whether the given points project inside or outside
    * the triangle, or onto a vertex or edge.
    * @param[in] r - Points to be projected.
    * @return 0 if strictly inside, 1 if on a vertex, 2 if on an edge, 3 if strictly outside.
    */
    EigRowVec<uint8_t> projection_loc(ConstEigRef<EigMatNX<Float, 3>> r) const;


    /**
    * @brief Checks whether a given point is inside the triangle, inclusive of edges and vertices.
    * @param[in] r - Point to be tested.
    * @param[in] tol - Comparison tolerance (optional).
    * @return True if the point is inside the triangle, inclusive of edges and vertices, false otherwise.
    */
    bool point_in_triangle(ConstEigRef<EigColVecN<Float, dim>> r, const Float tol = TRIANGLE_DEFAULT_TOL) const;


    /**
    * @brief Performs an affine transform to map points lying in this triangle's plane to
    * the output triangle's plane.
    * @tparam dim_out - Output triangle's dimension (2 or 3).
    * @param[in] points - Coordinates in this triangle's plane.
    * @param[in] tri_out - Target `Triangle` into whose plane `points` will be mapped.
    * @return Output coordinates in `tri_out`'s plane.
    */
    template <uint8_t dim_out>
    EigColVecN<Float, dim_out> map_points(
        ConstEigRef<EigColVecN<Float, dim>> points,
        const Triangle<dim_out>& tri_out
        ) const
    {
        EigMatNX<Float, 3> lambda = barycentric_coords(points);
        EigColVecN<Float, dim_out> points_out = tri_out.v(0) * lambda.row(0) +
                                      tri_out.v(1) * lambda.row(1) +
                                      tri_out.v(2) * lambda.row(2);
        return points_out;
    }


    /**
    * @brief Computes the area of the triangle formed by the given vertices.
    * @param[in] v - Vertices.
    * @return Area of the triangle.
    */
    static Float area(ConstEigRef<EigMatMN<Float, dim, 3>> v);


    /**
    * @brief Computes the unit normal vector for the triangle formed by the given vertices.
    * @param[in] v - Vertices.
    * @return Unit normal vector of the triangle.
    */
    static EigColVecN<Float, 3> normal(ConstEigRef<EigMatMN<Float, dim, 3>> v);


    /**
    * @brief Computes the centroid of the triangle formed by the given vertices.
    * @param[in] v - Vertices.
    * @return Centroid of the triangle.
    */
    static EigColVecN<Float, dim> centroid(ConstEigRef<EigMatMN<Float, dim, 3>> v)
    { return v.rowwise().sum() / (Float)3; };


protected:

    EigMatMN<Float, dim, 3> v_;
    Float area_;
    EigColVecN<Float, 3> normal_;
    EigColVecN<Float, dim> centroid_;
    EigRowVecN<Float, 3> edge_polarities_ = EigRowVecN<Float, 3>::Constant(1, 3, 1);

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "geometry/primitives/triangle.cpp"
#endif

#endif
