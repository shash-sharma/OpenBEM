// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes defining operations associated with the RWG and related function spaces.
*/

#include "rwg/function_space.hpp"

#include <functional>
#include <stdexcept>

#include <external/Eigen/Dense>

#include "types.hpp"
#include "matrix/base.hpp"

#include "geometry/primitives/triangle.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "quadrature/triangle/base.hpp"


namespace bem::rwg
{

EigMatNX<Float, 3> Rwg::value(
    const Triangle<3>& tri,
    uint8_t edge,
    ConstEigRef<EigMatNX<Float, 3>> points,
    const bool rotated
    )
{
    if (!rotated)
        return (points.colwise() - tri.v((edge + 2) % 3)) * normalization(tri)[edge];
    else
        return -(
            points.colwise() - tri.v((edge + 2) % 3)
            ).colwise().cross(tri.normal()) * normalization(tri)[edge];
};


EigColVecN<Complex, 3> Rwg::test_field(
    const Triangle<3>& tri,
    std::function<EigMatNX<Complex, 3> (ConstEigRef<EigMatNX<Float, 3>>)> field_eval,
    TriangleQuadratureBase<3>& tri_quad,
    const bool rotated
    )
{
    tri_quad.compute_points_weights(tri);
    EigMatNX<Complex, 3> field = field_eval(tri_quad.points());

    EigColVecN<Complex, 3> result = EigColVecN<Complex, 3>::Zero(3, 1);
    for (uint8_t edge = 0; edge < 3; ++edge)
    {
        result[edge] =
            ((
                value(tri, edge, tri_quad.points(), rotated).array() * field.array()
                ).colwise().sum().matrix() * tri_quad.weights().transpose())[0];
    }
    return result;
};


EigMatNX<Complex, 3> Rwg::reconstruct_field(
    const TriangleMesh<3>& mesh,
    const MatrixBase<Complex>& coeffs,
    ConstEigRef<EigMatNX<Float, 3>> points,
    const bool rotated
    )
{
    if (coeffs.num_cols() > 1)
        throw std::runtime_error("Rwg::reconstruct_field(): `coeffs` should be a column vector."
            );

    EigMatNX<Complex, 3> field = EigMatNX<Complex, 3>::Zero(3, points.cols());

    for (Index point = 0; point < points.cols(); ++point)
    {
        for (Index face = 0; face < mesh.num_elems(); ++face)
        {
            Triangle<3> tri = mesh.elem_primitive(face);

            if (!tri.point_in_triangle(points.col(point)))
                continue;

            for (uint8_t edge = 0; edge < 3; ++edge)
            {
                Index idx = mesh.elem_edges()(edge, face);
                field.col(point) += Rwg::value(tri, edge, points.col(point), rotated) * coeffs.value(idx, 0);
            }
        }
    }

    return field;

};


Complex Pulse::test_field(
    const Triangle<3>& tri,
    std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, 3>>)> field_eval,
    TriangleQuadratureBase<3>& tri_quad
    )
{
    tri_quad.compute_points_weights(tri);
    EigRowVec<Complex> field =
        field_eval(tri_quad.points()).array() *
        value(tri, tri_quad.points()).array();
    return (field * tri_quad.weights().transpose())[0];
};


EigRowVec<Complex> Pulse::reconstruct_field(
    const TriangleMesh<3>& mesh,
    const MatrixBase<Complex>& coeffs,
    ConstEigRef<EigMatNX<Float, 3>> points
    )
{

    if (coeffs.num_cols() > 1)
        throw std::runtime_error("Pulse::reconstruct_field(): `coeffs` should be a column vector."
            );

    EigRowVec<Complex> field = EigRowVec<Complex>::Zero(1, points.cols());

    for (Index point = 0; point < points.cols(); ++point)
    {
        for (Index face = 0; face < mesh.num_elems(); ++face)
        {
            Triangle<3> tri = mesh.elem_primitive(face);

            if (!tri.point_in_triangle(points.col(point)))
                continue;

            field[point] = Pulse::value(tri, points.col(point))[0] * coeffs.value(face, 0);
        }
    }

    return field;

};

}
