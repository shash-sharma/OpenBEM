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

#include "geometry/primitives/triangle.hpp"
#include "geometry/primitives/edge.hpp"

#include <stdexcept>

#include <external/Eigen/Dense>

#include "types.hpp"
#include "constants.hpp"


namespace bem
{

template <uint8_t dim>
void Triangle<dim>::set_v(
    ConstEigRef<EigMatMN<Float, dim, 3>> v,
    EigRowVecN<Float, 3> edge_polarities
    )
{
    v_ = v;
    edge_polarities_ = edge_polarities;
    area_ = area(v_);
    normal_ = normal(v_);
    centroid_ = centroid(v_);
    return;
};


template <uint8_t dim>
Triangle<2> Triangle<dim>::to_2d() const
{
    EigMatMN<Float, 2, 3> v_2d;
    EigColVecN<Float, dim> v10 = v_.col(1) - v_.col(0);

    v_2d.col(0) << 0.0, 0.0;
    v_2d.col(1) << v10.norm(), 0.0;

    if (v_2d.col(1)[0] == 0.0)
        throw std::domain_error("Triangle::get_triangle_in_2d(): divide by zero.");

    EigMatNX<Float, dim> vn0 = v_.rightCols(1) - v_.col(0);

    EigRowVec<Float> x = (v10.transpose() * vn0) / v_2d.col(1)[0];
    EigRowVec<Float> y = vn0.colwise().squaredNorm();

    y.array() -= x.array() * x.array();
    v_2d.topRightCorner(1, 1) = x;
    v_2d.bottomRightCorner(1, 1) = Eigen::sqrt(y.array()).matrix();

    return Triangle<2> (v_2d, edge_polarities_);
};


template <uint8_t dim>
Triangle<3> Triangle<dim>::to_3d(const Float z) const
{
    EigMatMN<Float, 3, 3> v_3d = EigMatMN<Float, 3, 3>::Constant(3, 3, z);
    v_3d.topRows(dim) = v_.topRows(dim);
    return Triangle<3> (v_3d, edge_polarities_);
};


template <uint8_t dim>
EigMatMN<Float, dim, dim> Triangle<dim>::local_coordinate_basis() const
{
    EigColVecN<Float, dim> v10 = v_.col(1) - v_.col(0);

    Float norm_v10 = v10.norm();
    if (norm_v10 == 0.0)
        throw std::domain_error("Triangle::local_coordinate_basis(): zero-length edge encountered.");

    EigMatMN<Float, dim, dim> uvw = EigMatMN<Float, dim, dim>::Zero(dim, dim);

    uvw.col(0) = v10 / norm_v10;

    EigColVecN<Float, 3> u_3d = EigColVecN<Float, 3>::Zero(3, 1);
    u_3d.topRows(dim) = uvw.col(0);
    EigColVecN<Float, 3> nxu = normal_.cross(u_3d);
    Float norm_nxu = nxu.norm();
    if (norm_nxu == 0.0)
        throw std::domain_error("Triangle::local_coordinate_basis(): zero-division encountered.");

    uvw.col(1) = nxu.topRows(dim) / norm_nxu;
    if constexpr (dim == 3)
        uvw.col(2) = normal_;

    return uvw;
};


template <uint8_t dim>
void Triangle<dim>::get_plane_projection(
    EigColVecN<Float, dim>& r_proj,
    Float& d,
    ConstEigRef<EigColVecN<Float, 3>> r,
    uint8_t ref_idx
    ) const
{
    EigColVecN<Float, 3> r_diff = r;
    r_diff.topRows(dim) -= v_.col(ref_idx);
    d = normal_.transpose() * r_diff;
    r_proj = (r - (normal_ * d)).topRows(dim);

    return;
};


template <uint8_t dim>
Triangle<dim> Triangle<dim>::reference_triangle()
{
    EigMatMN<Float, dim, 3> v;
    v.setZero();
    v(0, 1) = 1.0;
    v(1, 2) = 1.0;
    Triangle<dim> tri (v);
    return tri;
};


template <uint8_t dim>
EigMatNX<Float, 3> Triangle<dim>::barycentric_coords(ConstEigRef<EigMatNX<Float, dim>> p) const
{

    EigColVecN<Float, dim> v0 = v_.col(1) - v_.col(0);
    EigColVecN<Float, dim> v1 = v_.col(2) - v_.col(0);
    EigMatNX<Float, dim> v2 = p.colwise() - v_.col(0);

    Float d00 = v0.dot(v0);
    Float d01 = v0.dot(v1);
    Float d11 = v1.dot(v1);

    Float denom = (d00 * d11 - d01 * d01);

    if (denom == 0.0)
        throw std::domain_error("Triangle::barycentric_coords(): divide by zero.");
    Float inv_denom = 1.0 / denom;

    EigRowVec<Float> d20 = v0.transpose() * v2;
    EigRowVec<Float> d21 = v1.transpose() * v2;

    EigMatNX<Float, 3> lambda = EigMatNX<Float, 3>::Zero(3, p.cols());
    lambda.row(1).noalias() = (d11 * d20 - d01 * d21) * inv_denom;
    lambda.row(2).noalias() = (d00 * d21 - d01 * d20) * inv_denom;
    lambda.row(0).noalias() = EigMatNX<Float, 1>::Ones(1, p.cols()) - lambda.row(1) - lambda.row(2);

    return lambda;

};


template <uint8_t dim>
EigRowVec<uint8_t> Triangle<dim>::projection_loc(ConstEigRef<EigMatNX<Float, 3>> r) const
{
    Float tol = TRIANGLE_DEFAULT_TOL * shortest_edge_length();

    auto between_0_and_1 = [&tol] (Float val) -> bool
        {
            if (val > tol && val - 1 < -tol)
                return true;
            return false;
        };

    auto is_0 = [&tol] (Float val) -> bool
        {
            if (std::abs(val) < tol)
                return true;
            return false;
        };

    auto is_1 = [&tol] (Float val) -> bool
        {
            if (std::abs(val - 1) < tol)
                return true;
            return false;
        };

    EigMatNX<Float, 3> lambda = barycentric_coords(r.topRows(dim));
    EigRowVec<uint8_t> loc = EigRowVec<uint8_t>::Zero(1, r.cols());

    for (uint32_t col = 0; col < r.cols(); ++col)
    {
        EigColVecN<Float, 3> lambda_c = lambda.col(col);

        if (between_0_and_1(lambda_c[0]) &&
            between_0_and_1(lambda_c[1]) &&
            between_0_and_1(lambda_c[2]))
        {
            // strictly inside
            loc[col] = 0;
        }
        else if ((is_0(lambda_c[0]) && is_0(lambda_c[1]) && is_1(lambda_c[2])) ||
                    (is_0(lambda_c[0]) && is_0(lambda_c[2]) && is_1(lambda_c[1])) ||
                    (is_0(lambda_c[2]) && is_0(lambda_c[1]) && is_1(lambda_c[0])))
        {
            // at a vertex
            loc[col] = 1;
        }
        else if ((between_0_and_1(lambda_c[0]) && between_0_and_1(lambda_c[1])
                    && is_0(lambda_c[2])) ||
                (between_0_and_1(lambda_c[0]) && between_0_and_1(lambda_c[2])
                    && is_0(lambda_c[1])) ||
                (between_0_and_1(lambda_c[2]) && between_0_and_1(lambda_c[1])
                    && is_0(lambda_c[0])))
        {
            // on an edge
            loc[col] = 2;
        }
        else
        {
            // strictly outside
            loc[col] = 3;
        }
    }

    return loc;

};


template <uint8_t dim>
bool Triangle<dim>::point_in_triangle(ConstEigRef<EigColVecN<Float, dim>> r, const Float tol) const
{
    Float abs_tol = tol * shortest_edge_length();

    for (uint8_t ii = 0; ii < 3; ++ii)
    {
        Edge<dim> edge (v_.col(ii), v_.col((ii + 1) % 3));
        if (edge.point_on_edge(r, tol))
            return true;
    }

    EigMatMN<Float, dim, 3> p;

    p.col(0) = v_.col(0);
    p.col(1) = v_.col(1);
    p.col(2) = r;
    Triangle<dim> tri1 (p);

    if (tri1.shortest_edge_length() < abs_tol)
        return true;

    p.col(0) = v_.col(1);
    p.col(1) = v_.col(2);
    p.col(2) = r;
    Triangle<dim> tri2 (p);

    if (tri2.shortest_edge_length() < abs_tol)
        return true;

    p.col(0) = v_.col(2);
    p.col(1) = v_.col(0);
    p.col(2) = r;
    Triangle<dim> tri3 (p);

    if (tri3.shortest_edge_length() < abs_tol)
        return true;

    if (std::abs(tri1.normal().dot(tri2.normal()) - 1.0) < abs_tol &&
        std::abs(tri2.normal().dot(tri3.normal()) - 1.0) < abs_tol)
        return true;

    return false;
};


template <uint8_t dim>
Float Triangle<dim>::area(ConstEigRef<EigMatMN<Float, dim, 3>> v)
{
    Float area = 0;

    if constexpr (dim == 2)
    {
        area = 0.5 * std::abs(v(0, 0) * (v(1, 1) - v(1, 2)) -
                              v(1, 0) * (v(0, 1) - v(0, 2)) +
                              (v(0, 1) * v(1, 2) - v(1, 1) * v(0, 2)));
    }
    else if constexpr (dim == 3)
    {
        area = ((v.col(1) - v.col(0)).cross(v.col(2) - v.col(0))).norm() / 2.0;
    }

    return area;
};


template <uint8_t dim>
EigColVecN<Float, 3> Triangle<dim>::normal(ConstEigRef<EigMatMN<Float, dim, 3>> v)
{
    if constexpr (dim == 2)
        return {0.0, 0.0, 1.0};
    else if constexpr (dim == 3)
    {
        EigColVecN<Float, 3> orth = (v.col(1) - v.col(0)).cross(v.col(2) - v.col(1));
        Float norm = orth.norm();
        return orth / norm;
    }
};


template class Triangle<2>;
template class Triangle<3>;

}
