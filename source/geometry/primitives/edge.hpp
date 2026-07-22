// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Edge primitive.
*/

#ifndef EDGE_H
#define EDGE_H

#include <stdexcept>

#include "types.hpp"


namespace bem
{

const Float EDGE_DEFAULT_TOL = 1.0e-6;

/**
* \ingroup prim
* @{
*/

/**
* @brief Edge primitive class.
* @tparam dim - The dimension of the edge (1, 2 or 3).
*/
template <uint8_t dim>
class Edge
{

    static_assert((dim == 1 || dim == 2 || dim == 3), "Edge must be defined in one, two, or three dimensions.");

public:

    /**
    * @brief Constructs an `Edge` with given vertices.
    * @param[in] v - Vertices of the edge.
    */
    Edge(ConstEigRef<EigMatMN<Float, dim, 2>> v)
    {
        set_data(v);
        return;
    };


    /**
    * @brief Constructs an `Edge` with given vertices.
    * @param[in] v1 - First vertex of the edge.
    * @param[in] v2 - Second vertex of the edge.
    */
    Edge(ConstEigRef<EigColVecN<Float, dim>> v1, ConstEigRef<EigColVecN<Float, dim>> v2)
    {
        EigMatMN<Float, dim, 2> v;
        v << v1, v2;
        set_data(v);
        return;
    };


    /**
    * @brief Returns the vertices of this `Edge`.
    * @return Vertices of the edge.
    */
    const EigMatMN<Float, dim, 2>& v() const
    {
        return v_;
    };


    /**
    * @brief Returns the vertex of this `Edge` at a given index.
    * @param[in] idx - Index of the desired vertex (0 or 1).
    * @return Vertex of the edge at the given index.
    */
    EigColVecN<Float, dim> v(const uint8_t idx) const
    {
        if (idx >= 2)
            throw std::out_of_range("Edge::v(): vertex index out of bounds.");
        return v_.col(idx);
    };


    /**
    * @brief Sets the vertices of this `Edge`.
    * @param[in] v - Vertices of the edge.
    */
    void set_data(ConstEigRef<EigMatMN<Float, dim, 2>> v);


    /**
    * @brief Returns the length of this `Edge`.
    * @return Length of the edge.
    */
    Float length() const
    { return length_; };


    /**
    * @brief Returns the vector from the first to the second vertex of this `Edge`.
    * @return Vector from the first to the second vertex of the edge.
    */
    EigColVecN<Float, dim> edge_vec() const
    { return edge_vec_; };


    /**
    * @brief Returns the unit vector along this `Edge` from its first to its second vertex.
    * @return Unit vector along the edge from the first to the second vertex.
    */
    EigColVecN<Float, dim> unit_vec() const
    { return unit_vec_; };


    /**
    * @brief Returns the centroid of this `Edge`.
    * @return Centroid of the edge.
    */
    EigColVecN<Float, dim> centroid() const
    { return centroid_; };


    /**
    * @brief Returns a reference edge in the specified dimension from 0 to 1 along the first axis.
    * @return Reference edge with vertices (0, 0, ...) and (1, 0, ...).
    */
    static Edge<dim> reference_edge();


    /**
    * @brief Returns an equivalent edge with coordinates in a local 2D system, and with the local
    * origin at this edge's first vertex. If this edge is defined in 1D, the coordinates are
    * unchanged and a second 0-value dimension is added.
    * @return 2D `Edge` with coordinates in a local 2D system.
    */
    Edge<2> to_2d() const;


    /**
    * @brief Adds a 0-valued third dimension if this `Edge` is in 2D, adds a 0-valued second
    * and third dimension if it is in 1D, otherwise just returns a copy of this `Edge`.
    * @return 3D `Edge`.
    */
    Edge<3> to_3d() const;


    /**
    * @brief Returns true if the given point is on the edge, inclusive of vertices.
    * @param[in] r - Point to be tested.
    * @param[in] tol - Tolerance (optional).
    * @return True if the point is on the edge, inclusive of vertices, false otherwise.
    */
    bool point_on_edge(ConstEigRef<EigColVecN<Float, dim>> r, const Float tol = EDGE_DEFAULT_TOL) const;


private:

    /**
    * @brief Compute the vector from the first to the second vertex of this `Edge`.
    * @return Vector from the first to the second vertex of the edge.
    */
    EigColVecN<Float, dim> compute_edge_vec() const { return v_.col(1) - v_.col(0); };


    /**
    * @brief Compute the length of this `Edge`.
    * @return Length of the edge.
    */
    Float compute_length() const { return edge_vec().norm(); };


    /**
    * @brief Compute the unit vector along this `Edge` from its first to its second vertex.
    * @return Unit vector along the edge from the first to the second vertex.
    */
    EigColVecN<Float, dim> compute_unit_vec() const { return edge_vec() / length(); };


    /**
    * @brief Compute the centroid of this `Edge`.
    * @return Centroid of the edge.
    */
    EigColVecN<Float, dim> compute_centroid() const { return v_.rowwise().sum() / 2.0; };


    EigMatMN<Float, dim, 2> v_;
    Float length_;
    EigColVecN<Float, dim> edge_vec_, unit_vec_, centroid_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "geometry/primitives/edge.cpp"
#endif

#endif
