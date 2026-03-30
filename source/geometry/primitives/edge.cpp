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

#include "geometry/primitives/edge.hpp"

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem
{

template <uint8_t dim>
void Edge<dim>::set_v(ConstEigRef<EigMatMN<Float, dim, 2>> v)
{
    v_ = v;
    edge_vec_ = compute_edge_vec();
    length_ = compute_length();
    unit_vec_ = compute_unit_vec();
    centroid_ = compute_centroid();
    return;
};


template <uint8_t dim>
Edge<dim> Edge<dim>::reference_edge()
{
    EigMatMN<Float, dim, 2> v;
    v.setZero();
    v(0, 1) = 1.0;
    Edge<dim> edge (v);
    return edge;
};


template <uint8_t dim>
Edge<2> Edge<dim>::to_2d() const
{

    EigMatMN<Float, 2, 2> v_2d;

    if constexpr (dim == 1)
        v_2d.row(0) = v_;
    else
    {
        v_2d.col(0) << 0.0, 0.0;
        v_2d.col(1) << length(), 0.0;
    }

    Edge<2> edge_2d (v_2d);
    return edge_2d;

};


template <uint8_t dim>
Edge<3> Edge<dim>::to_3d() const
{
    EigMatMN<Float, 3, 2> v_3d = EigMatMN<Float, 3, 2>::Zero(3, 2);
    v_3d.topRows(dim) = v_.topRows(dim);
    Edge<3> edge_3d (v_3d);
    return edge_3d;
};


template <uint8_t dim>
bool Edge<dim>::point_on_edge(ConstEigRef<EigColVecN<Float, dim>> r, const Float tol) const
{
    Float abs_tol = tol * length();

    Float length_1 = (v().col(0) - r).norm();
    Float length_2 = (v().col(1) - r).norm();
    Float test_length = length_1 + length_2;

    if (std::abs(test_length - length()) < abs_tol)
        return true;

    return false;
};


template class Edge<1>;
template class Edge<2>;
template class Edge<3>;

}
