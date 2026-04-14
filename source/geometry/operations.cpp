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

#include "geometry/operations.hpp"

#include <stdexcept>

#include <external/Eigen/Dense>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/edge.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem
{

template <uint8_t dim>
EigMatNX<Float, dim> GeometryOps<dim>::polar_to_cartesian(ConstEigRef<EigMatNX<Float, dim>> points)
{
    if constexpr (dim != 2 && dim != 3)
        throw std::invalid_argument("GeometryOps::polar_to_cartesian(): `dim` must be 2 or 3.");

    EigMatNX<Float, dim> points_out = EigMatNX<Float, dim>::Zero(dim, points.cols());
    points_out.row(0) = points.row(0).array() * Eigen::cos(points.row(1).array());
    points_out.row(1) = points.row(0).array() * Eigen::sin(points.row(1).array());
    if (dim == 3)
    {
        points_out.row(0).array() *= Eigen::sin(points.row(2).array());
        points_out.row(1).array() *= Eigen::sin(points.row(2).array());
        points_out.row(2) = points.row(0).array() * Eigen::cos(points.row(2).array());
    }
    return points_out;
}


template <uint8_t dim>
EigMatNX<Complex, dim> GeometryOps<dim>::cartesian_to_polar_field(
    ConstEigRef<EigMatNX<Float, dim>> points,
    ConstEigRef<EigMatNX<Complex, dim>> field
    )
{
    if constexpr (dim != 2 && dim != 3)
        throw std::invalid_argument("GeometryOps::cartesian_to_polar_field(): `dim` must be 2 or 3.");

    EigRowVec<Float> sin_phi = EigRowVec<Float>::Zero(1, points.cols());
    EigRowVec<Float> cos_phi = EigRowVec<Float>::Zero(1, points.cols());
    for (Index ii = 0; ii < points.cols(); ++ii)
    {
        Float phi = std::atan2(points(1, ii), points(0, ii));
        sin_phi[ii] = std::sin(phi);
        cos_phi[ii] = std::cos(phi);
    }

    EigRowVec<Float> sin_theta, cos_theta;
    if constexpr (dim == 3)
    {
        EigRowVec<Float> theta = Eigen::acos(points.row(2).array() / points.colwise().norm().array());
        sin_theta = Eigen::sin(theta.array());
        cos_theta = Eigen::cos(theta.array());
    }

    EigMatNX<Complex, dim> field_out = EigMatNX<Complex, dim>::Zero(dim, points.cols());
    if (dim == 3)
    {
        field_out.row(0).array() = field.row(0).array() * cos_phi.array() * sin_theta.array() +
                                    field.row(1).array() * sin_phi.array() * sin_theta.array() +
                                    field.row(2).array() * cos_theta.array();
        field_out.row(1).array() = -field.row(0).array() * sin_phi.array() +
                                    field.row(1).array() * cos_phi.array();
        field_out.row(2).array() = field.row(0).array() * cos_phi.array() * cos_theta.array() +
                                    field.row(1).array() * sin_phi.array() * cos_theta.array() -
                                    field.row(2).array() * sin_theta.array();
    }
    else if (dim == 2)
    {
        field_out.row(0).array() = field.row(0).array() * cos_phi.array() +
                                    field.row(1).array() * sin_phi.array();
        field_out.row(1).array() = -field.row(0).array() * sin_phi.array() +
                                    field.row(1).array() * cos_phi.array();
    }

    return field_out;
}


template <uint8_t dim>
EigMatNX<Float, dim> GeometryOps<dim>::transform_coordinate_system(
    ConstEigRef<EigMatNX<Float, dim>> v_in,
    ConstEigRef<EigColVecN<Float, dim>> new_origin,
    ConstEigRef<EigMatMN<Float, dim, dim>> new_uvw
    )
{
    return new_uvw.transpose() * (v_in.colwise() - new_origin);
}


template <uint8_t dim>
EigRowVec<Float> GeometryOps<dim>::angles_between_vectors(
    ConstEigRef<EigMatNX<Float, dim>> v1,
    ConstEigRef<EigMatNX<Float, dim>> v2
    )
{
    if constexpr (dim != 2 && dim != 3)
        throw std::invalid_argument("GeometryOps::angles_between_vectors(): `dim` must be 2 or 3.");

    if (v1.cols() != v2.cols())
        throw std::invalid_argument(
            "angle_between_vectors(): number of vectors in set v1 does not equal the number of vectors in set v2."
        );

    return Eigen::acos(
        (v1.cwiseProduct(v2)).colwise().sum().array() /
        v1.colwise().norm().array() /
        v2.colwise().norm().array()
        );
}


template <uint8_t dim>
Float GeometryOps<dim>::angle_between_vectors(
    ConstEigRef<EigColVecN<Float, dim>> v1,
    ConstEigRef<EigColVecN<Float, dim>> v2
    )
{
    if constexpr (dim != 2 && dim != 3)
        throw std::invalid_argument("GeometryOps::angle_between_vectors(): `dim` must be 2 or 3.");
    return std::acos(v1.dot(v2) / v1.norm() / v2.norm());
}


template <uint8_t dim>
EigRowVec<Float> GeometryOps<dim>::directed_angles_between_vectors(
    ConstEigRef<EigMatNX<Float, dim>> v1,
    ConstEigRef<EigMatNX<Float, dim>> v2
    )
{
    if constexpr (dim != 2 && dim != 3)
        throw std::invalid_argument("GeometryOps::directed_angles_between_vectors(): `dim` must be 2 or 3.");

    if (v1.cols() != v2.cols())
        throw std::invalid_argument(
            "directed_angle_between_vectors(): number of vectors in set v1 does not equal the number of vectors in set v2."
        );

    EigRowVec<Float> dot = (v1.cwiseProduct(v2)).colwise().sum();
    EigRowVec<Float> det = EigRowVec<Float>::Zero(1, v1.cols());
    if constexpr (dim == 2)
        det.noalias() = v1.row(0) * v2.row(1).asDiagonal() - v2.row(0) * v1.row(1).asDiagonal();
    else if constexpr (dim == 3)
        for (uint32_t ii = 0; ii < v1.cols(); ii++)
            det[ii] = (v1.col(ii).cross(v2.col(ii))).norm();

    EigRowVec<Float> angles = EigRowVec<Float>::Zero(1, v1.cols());
    for (uint32_t rr = 0; rr < v1.cols(); ++rr)
        angles[rr] = std::atan2(det[rr], dot[rr]);

    Float tol = std::max(
        v1.colwise().norm().maxCoeff(), v2.colwise().norm().maxCoeff()
        ) * GEOMETRY_DEFAULT_TOL;
    angles = (Eigen::abs(dot.array()) < tol).select(
        EigRowVec<Float>::Zero(1, v1.cols()), angles
        );

    return angles;
}


template <uint8_t dim>
Float GeometryOps<dim>::directed_angle_between_vectors(
    ConstEigRef<EigColVecN<Float, dim>> v1,
    ConstEigRef<EigColVecN<Float, dim>> v2
    )
{
    if constexpr (dim != 2 && dim != 3)
        throw std::invalid_argument("GeometryOps::directed_angle_between_vectors(): `dim` must be 2 or 3.");

    Float dot = v1.dot(v2);
    Float det;
    if constexpr (dim == 2)
        det = v1[0] * v2[1] - v2[0] * v1[1];
    else if constexpr (dim == 3)
        det = (v1.cross(v2)).norm();

    Float angle = std::atan2(det, dot);

    Float tol = std::max(v1.norm(), v2.norm()) * GEOMETRY_DEFAULT_TOL;
    if (std::abs(dot) < tol)
        angle = 0;

    return angle;
}


template <uint8_t dim>
uint8_t GeometryOps<dim>::common_vertices(
    const Triangle<dim>& triangle1,
    const Triangle<dim>& triangle2,
    const Float tol
    )
{

    // The tolerance is measured relative to the smaller mean edge length of both triangles
    Float toln = tol * std::min(triangle1.mean_edge_length(), triangle2.mean_edge_length());

    uint8_t num_common_vertices = 0;

    for (uint8_t ii = 0; ii < 3; ii++)
        for (uint8_t jj = 0; jj < 3; jj++)
            if ((triangle2.v(jj) - triangle1.v(ii)).norm() < toln)
            {
                num_common_vertices++;
                break;
            }

    assert(num_common_vertices <= 3 && "Triangle::common_vertices(): Result out of range.");

    return num_common_vertices;

}


template <uint8_t dim>
uint8_t GeometryOps<dim>::common_vertices(
    EigRef<EigColVecN<Index, 3>> idx1,
    EigRef<EigColVecN<Index, 3>> idx2,
    const Triangle<dim>& triangle1,
    const Triangle<dim>& triangle2,
    const Float tol
    )
{

    // The tolerance is measured relative to the smaller mean edge length of both triangles
    Float toln = tol * std::min(triangle1.mean_edge_length(), triangle2.mean_edge_length());

    uint8_t num_common_vertices = 0;
    idx1.setConstant(10);
    idx2.setConstant(10);

    for (uint8_t ii = 0; ii < 3; ii++)
        for (uint8_t jj = 0; jj < 3; jj++)
            if ((triangle2.v(jj) - triangle1.v(ii)).norm() < toln)
            {
                idx1[num_common_vertices] = ii;
                idx2[num_common_vertices] = jj;
                num_common_vertices++;
                break;
            }

    assert(num_common_vertices <= 3 && "Triangle::common_vertices(): Result out of range.");

    return num_common_vertices;

}


template <uint8_t dim>
int8_t GeometryOps<dim>::check_parallel_triangles(
    const Triangle<3>& triangle1,
    const Triangle<3>& triangle2,
    const Float tol
    )
{
    int8_t parallel = 0;
    if (std::abs(triangle1.normal().dot(triangle2.normal()) - 1.0) <= tol)
        parallel = 1;
    if (std::abs(triangle1.normal().dot(triangle2.normal()) + 1.0) <= tol)
        parallel = -1;
    return parallel;
}


template <uint8_t dim>
bool GeometryOps<dim>::check_perpendicular_triangles(
    const Triangle<3>& triangle1,
    const Triangle<3>& triangle2,
    const Float tol
    )
{
    bool perpendicular = false;
    if (std::abs(triangle1.normal().dot(triangle2.normal())) <= tol)
        perpendicular = true;
    return perpendicular;
}


template <uint8_t dim>
bool GeometryOps<dim>::check_coplanar_triangles(
    const Triangle<3>& triangle1,
    const Triangle<3>& triangle2,
    const Float tol
    )
{
    bool coplanar = false;
    if (check_parallel_triangles(triangle1, triangle2, tol) != 0)
    {
        const EigColVecN<Float, 3> dist_vec = (triangle2.centroid() - triangle1.centroid()).normalized();
        if (std::abs(triangle1.normal().dot(dist_vec)) <= tol)
            coplanar = true;
    }
    return coplanar;
}


template <uint8_t dim>
int8_t GeometryOps<dim>::check_parallel_edges(
    const Edge<dim>& edge1,
    const Edge<dim>& edge2,
    const Float tol
    )
{
    if (std::abs(edge1.unit_vec().dot(edge2.unit_vec()) - 1.0) <= tol)
        return 1;
    if (std::abs(edge1.unit_vec().dot(edge2.unit_vec()) + 1.0) <= tol)
        return -1;
    return 0;
}


template <uint8_t dim>
bool GeometryOps<dim>::check_perpendicular_edges(
    const Edge<dim>& edge1,
    const Edge<dim>& edge2,
    const Float tol
    )
{
    if (edge1.unit_vec().dot(edge2.unit_vec()) <= tol)
        return true;
    return false;
}


template <uint8_t dim>
bool GeometryOps<dim>::point_in_polygon(
    ConstEigRef<EigColVecN<Float, dim>>& point,
    ConstEigRef<EigMatNX<Float, dim>>& polygon,
    const Float tol
    )
{

    Float mean_edge_length = (polygon.rightCols(polygon.cols() - 1) + polygon.leftCols(polygon.cols() - 1)).colwise().norm().mean();
    Float abs_tol = tol * mean_edge_length;

    if (polygon.cols() < 3)
        throw std::invalid_argument("GeometryOps::point_in_polygon(): `polygon` must have at least three vertices.");

    EigMatNX<Float, 2> polygon_2d = EigMatNX<Float, 2>::Zero(2, polygon.cols());
    EigColVecN<Float, 2> point_2d;

    if constexpr (dim == 1)
        throw std::invalid_argument("GeometryOps::point_in_polygon(): `dim` must be 2 or 3.");

    else if constexpr (dim == 3)
    {
        EigColVecN<Float, 3> normal = (polygon.col(1) - polygon.col(0)).cross(
            polygon.col(2) - polygon.col(0)).normalized();
        Float dist_to_plane = std::abs((point - polygon.col(0)).dot(normal));

        if (dist_to_plane > abs_tol)
            return false;

        EigColVecN<Float, 3> u = (polygon.col(1) - polygon.col(0)).normalized();
        EigColVecN<Float, 3> v = normal.cross(u).normalized();

        for (uint8_t ii = 0; ii < polygon.cols(); ++ii)
        {
            EigColVecN<Float, 3> vec = polygon.col(ii) - polygon.col(0);
            polygon_2d(0, ii) = vec.dot(u);
            polygon_2d(1, ii) = vec.dot(v);
        }

        EigColVecN<Float, 3> vec = point - polygon.col(0);
        point_2d[0] = vec.dot(u);
        point_2d[1] = vec.dot(v);
    }

    else if constexpr (dim == 2)
    {
        polygon_2d = polygon;
        point_2d = point;
    }

    uint8_t crossings = 0;
    for (uint8_t ii = 0; ii < polygon_2d.cols(); ++ii)
    {
        const EigColVecN<Float, 2> v1 = polygon_2d.col(ii);
        const EigColVecN<Float, 2> v2 = polygon_2d.col((ii + 1) % polygon_2d.cols());

        Edge<2> edge (v1, v2);
        if (edge.point_on_edge(point_2d))
            return true;

        if (point_2d[1] > std::min(v1[1], v2[1]) - abs_tol)
        {
            if (point_2d[1] < std::max(v1[1], v2[1]) + abs_tol)
            {
                if (point_2d[0] < std::max(v1[0], v2[0]) + abs_tol)
                {
                    if (std::abs(v1[1] - v2[1]) >= abs_tol)
                    {
                        Float xinters = (point_2d[1] - v1[1]) * (v2[0] - v1[0]) / (v2[1] - v1[1]) + v1[0];
                        if (std::abs(v1[0] - v2[0]) < abs_tol || point_2d[0] < xinters + abs_tol)
                            crossings++;
                    }
                }
            }
        }
    }

    return (crossings % 2) == 1;

}


template class GeometryOps<1>;
template class GeometryOps<2>;
template class GeometryOps<3>;

}
