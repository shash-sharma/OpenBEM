// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Geometry operations.
*/

#ifndef BEM_GEOM_OPS_H
#define BEM_GEOM_OPS_H

#include <cassert>

#include "types.hpp"
#include "constants.hpp"


namespace bem
{

const Float GEOMETRY_DEFAULT_TOL = std::sqrt(float_eps);

// Forward declarations
template <uint8_t dim> class Edge;
template <uint8_t dim> class Triangle;

/**
* \ingroup geom
* @{
*/

/**
* @brief Geometry operations class.
* @tparam dim - The dimension of the geometry operation class (1, 2, or 3).
*/
template <uint8_t dim>
class GeometryOps
{

    static_assert((dim == 1 || dim == 2 || dim == 3), "Dimension must be one, two, or three.");

public:

    /**
    * @brief Transforms polar (2D) or spherical (3D) coordinates to Cartesian.
    * @param[in] points - Polar (\f$r\f$, \f$\phi\f$) or spherical (\f$r\f$, \f$\phi\f$, \f$\theta\f$)
    * coordinates.
    * @return Cartesian coordinates.
    */
    static EigMatNX<Float, dim> polar_to_cartesian(ConstEigRef<EigMatNX<Float, dim>> points);


    /**
    * @brief Transforms a vector field in Cartesian space to polar (2D) or spherical (3D) space.
    * @param[in] points - Cartesian coordinates at which the field is defined.
    * @param[in] field - Vector field values at the given coordinates.
    * @return Vector field in polar (\f$r\f$, \f$\phi\f$) or spherical (\f$r\f$, \f$\phi\f$, \f$\theta\f$)
    * coordinates.
    */
    static EigMatNX<Complex, dim> cartesian_to_polar_field(
        ConstEigRef<EigMatNX<Float, dim>> points,
        ConstEigRef<EigMatNX<Complex, dim>> field
        );


    /**
    * @brief Transforms the given coordinates to a new coordinate system defined by a new origin and
    * orthogonal unit vectors.
    * @param[in] v_in - Coordinates in the original system.
    * @param[in] new_origin - Origin of the new system with respect to the original system.
    * @param[in] new_uvw - Orthogonal unit vectors for the new system with respect to the original system.
    * @return Transformed coordinates in the new system.
    */
    static EigMatNX<Float, dim> transform_coordinate_system(
        ConstEigRef<EigMatNX<Float, dim>> v_in,
        ConstEigRef<EigColVecN<Float, dim>> new_origin,
        ConstEigRef<EigMatMN<Float, dim, dim>> new_uvw
        );


    /**
    * @brief Returns the angles between pairs of vectors in radians.
    * @param[in] v1 - First set of vectors.
    * @param[in] v2 - Second set of vectors.
    * @return Angles between each pair of vectors in radians.
    */
    static EigRowVec<Float> angle_between_vectors(
        ConstEigRef<EigMatNX<Float, dim>> v1,
        ConstEigRef<EigMatNX<Float, dim>> v2
        );


    /**
    * @brief Returns the directional angles between pairs of vectors in a right-hand system where
    * counter-clockwise angles are positive.
    * @param[in] v1 - First set of vectors.
    * @param[in] v2 - Second set of vectors.
    * @return Directed angles between each pair of vectors in radians.
    */
    static EigRowVec<Float> directed_angle_between_vectors(
        ConstEigRef<EigMatNX<Float, dim>> v1,
        ConstEigRef<EigMatNX<Float, dim>> v2
        );


    /**
    * @brief Returns the number of vertices common or coinciding between two `Triangle` objects.
    * @param[in] triangle1 - First triangle.
    * @param[in] triangle2 - Second triangle.
    * @param[in] tol - Comparison tolerance to test if vertices coincide (optional).
    * @return Number of vertices common or coinciding between the two triangles.
    */
    static uint8_t common_vertices(
        const Triangle<dim>& triangle1,
        const Triangle<dim>& triangle2,
        const Float tol = GEOMETRY_DEFAULT_TOL
        );


    /**
    * @brief Returns the number and indices of vertices common or coinciding between two `Triangle` objects.
    * @param[out] idx1 - Indices of common vertices for the first triangle.
    * @param[out] idx2 - Indices of common vertices for the second triangle.
    * @param[in] triangle1 - First triangle.
    * @param[in] triangle2 - Second triangle.
    * @param[in] tol - Comparison tolerance to test if vertices coincide (optional).
    * @return Number of vertices common or coinciding between the two triangles.
    * @details
    * For `N` common vertices, the first `N` entries of `idx1` and `idx2` are populated with common indices
    * of the vertices; the rest are dummy values filled with 10. If there are no common vertices, then `idx1`
    * and `idx2` will contain 10 in each entry. In the case that one or more vertex does coincide, then the
    * entries of `idx1` and `idx2` are ordered so that there is a one-to-one correspondence, e.g.,
    * the idx1[0] vertex of `triangle1` coincides with the `idx2[0]` vertex of `triangle2`.
    */
    static uint8_t common_vertices(
        EigRef<EigColVecN<Index, 3>> idx1,
        EigRef<EigColVecN<Index, 3>> idx2,
        const Triangle<dim>& triangle1,
        const Triangle<dim>& triangle2,
        const Float tol = GEOMETRY_DEFAULT_TOL
        );


    /**
    * @brief Checks if the normal vectors of two triangles are (anti-)parallel.
    * @param[in] triangle1 - First `Triangle`.
    * @param[in] triangle2 - Second `Triangle`.
    * @param[in] tol - Comparison tolerance (optional).
    * @return 0 if not parallel, 1 if parallel, -1 if anti-parallel.
    */
    static int8_t check_parallel_triangles(
        const Triangle<3>& triangle1,
        const Triangle<3>& triangle2,
        const Float tol = GEOMETRY_DEFAULT_TOL
        );


    /**
    * @brief Checks if the normal vectors of two triangles are perpendicular.
    * @param[in] triangle1 - First `Triangle`.
    * @param[in] triangle2 - Second `Triangle`.
    * @param[in] tol - Comparison tolerance (optional).
    * @return True if the triangles are perpendicular, false otherwise.
    */
    static bool check_perpendicular_triangles(
        const Triangle<3>& triangle1,
        const Triangle<3>& triangle2,
        const Float tol = GEOMETRY_DEFAULT_TOL
        );


    /**
    * @brief Checks if two triangles are coplanar.
    * @param[in] triangle1 - First `Triangle`.
    * @param[in] triangle2 - Second `Triangle`.
    * @param[in] tol - Comparison tolerance (optional).
    * @return True if the triangles are coplanar, false otherwise.
    */
    static bool check_coplanar_triangles(
        const Triangle<3>& triangle1,
        const Triangle<3>& triangle2,
        const Float tol = GEOMETRY_DEFAULT_TOL
        );


    /**
    * @brief Checks if two edges are (anti-)parallel.
    * @param[in] edge1 - First `Edge`.
    * @param[in] edge2 - Second `Edge`.
    * @param[in] tol - Comparison tolerance (optional).
    * @return 0 if not parallel, 1 if parallel, -1 if anti-parallel.
    */
    static int8_t check_parallel_edges(
        const Edge<dim>& edge1,
        const Edge<dim>& edge2,
        const Float tol = GEOMETRY_DEFAULT_TOL
        );


    /**
    * @brief Checks if two edges are perpendicular.
    * @param[in] edge1 - First `Edge`.
    * @param[in] edge2 - Second `Edge`.
    * @param[in] tol - Comparison tolerance (optional).
    * @return True if the edges are perpendicular, false otherwise.
    */
    static bool check_perpendicular_edges(
        const Edge<dim>& edge1,
        const Edge<dim>& edge2,
        const Float tol = GEOMETRY_DEFAULT_TOL
        );


    /**
    * @brief Checks if a point is inside a polygon using ray casting.
    * @param[in] point - Point to test.
    * @param[in] polygon - Polygon vertices.
    * @param[in] tol - Relative tolerance for comparison (optional).
    * @details
    * Reference: https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
    */
    static bool point_in_polygon(
        ConstEigRef<EigColVecN<Float, dim>>& point,
        ConstEigRef<EigMatNX<Float, dim>>& polygon,
        const Float tol = GEOMETRY_DEFAULT_TOL
        );

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "geometry/operations.cpp"
#endif

#endif
