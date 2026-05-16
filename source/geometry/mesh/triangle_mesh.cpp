// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Triangular mesh functionality.
*/

#include "geometry/mesh/triangle_mesh.hpp"

#include <map>
#include <utility>
#include <algorithm>
#include <vector>
#include <array>
#include <set>
#include <stdexcept>

#include "types.hpp"
#include "geometry/mesh/base.hpp"


namespace bem
{

template <uint8_t dim>
void TriangleMesh<dim>::set_data(
    ConstEigRef<EigMatNX<Float, dim>> verts,
    ConstEigRef<EigMatNX<Index, 3>> elems,
    ConstEigRef<EigRowVec<Index>> elem_tags,
    const bool decoupled_edges
    )
{
    base::verts_ = verts;
    base::elems_ = elems;
    base::elem_tags_ = elem_tags;
    decoupled_edges_ = decoupled_edges;
    generate_edges();
    return;
};


template <uint8_t dim>
void TriangleMesh<dim>::partition_by_elems(
    MeshBase<dim, 3>& partition,
    ConstEigRef<EigRowVec<Index>> elem_inds
    ) const
{

    Index num_new_verts = 0;
    std::map<Index, Index> new_vert_map;

    for (Index jj = 0; jj < elem_inds.size(); ++jj)
    {
        for (uint8_t kk = 0; kk < 3; ++kk)
        {
            bool inserted = new_vert_map.insert(
                std::make_pair(base::elems_(kk, elem_inds[jj]), num_new_verts)
                ).second;
            if (inserted)
                num_new_verts++;
        }
    }

    EigMatNX<Float, dim> new_verts = EigMatNX<Float, dim>::Zero(dim, num_new_verts);
    for (auto const& x: new_vert_map)
        new_verts.col(x.second) = base::verts_.col(x.first);

    EigMatNX<Index, 3> new_elems = EigMatNX<Index, 3>::Zero(3, elem_inds.size());

    EigRowVec<Index> new_elem_tags = EigRowVec<Index>::Zero(1, elem_inds.size());

    for (Index jj = 0; jj < elem_inds.size(); ++jj)
    {
        for (uint8_t kk = 0; kk < 3; ++kk)
        {
            new_elems(kk, jj) = new_vert_map[
                base::elems_(kk, elem_inds[jj])
                ];
        }
        new_elem_tags[jj] = base::elem_tags_[elem_inds[jj]];
    }

    TriangleMesh<dim>& partition_cast = dynamic_cast<TriangleMesh<dim>&> (partition);
    partition_cast.set_data(new_verts, new_elems, new_elem_tags, decoupled_edges_);

    return;

};


template <uint8_t dim>
void TriangleMesh<dim>::partition_by_bbox(
    MeshBase<dim, 3>& partition,
    ConstEigRef<EigMatMN<Float, dim, 2>> bbox,
    const bool strict
    ) const
{

    std::vector<Index> elems_in_bbox;

    for (Index ii = 0; ii < base::elems_.cols(); ++ii)
    {
        EigMatMN<Float, dim, 3> coords = base::verts_(
            Eigen::placeholders::all, base::elems_.col(ii)
            );
        EigColVecN<Float, dim> centroid = coords.rowwise().mean();
        bool centroid_in_bbox =
            (centroid.array() <= (bbox.rowwise().maxCoeff()).array()).all() &&
            (centroid.array() >= (bbox.rowwise().minCoeff()).array()).all();

        if (centroid_in_bbox)
        {
            if (!strict)
                elems_in_bbox.push_back(ii);
            else
            {
                bool fully_in_bbox =
                    ((coords.rowwise().maxCoeff()).array() <= (bbox.rowwise().maxCoeff()).array()
                    ).all() &&
                    ((coords.rowwise().minCoeff()).array() >= (bbox.rowwise().minCoeff()).array()
                    ).all();

                if (fully_in_bbox)
                    elems_in_bbox.push_back(ii);
            }
        }
    }

    partition_by_elems(
        partition,
        Eigen::Map<EigRowVec<Index>> (elems_in_bbox.data(), elems_in_bbox.size())
        );

    return;

};


template <uint8_t dim>
void TriangleMesh<dim>::generate_edges()
{

    Index num_edges = 0;
    std::vector<std::pair<Index, uint8_t>> edge_counts;
    std::vector<std::pair<std::pair<Index, Index>, Index>> edges;

    elem_edges_.resize(3, base::elems_.cols());
    elem_edge_polarities_.resize(3, base::elems_.cols());

    std::set<Index> unique_tags { base::elem_tags_.begin(), base::elem_tags_.end() };

    for (Index tag: unique_tags)
    {
        std::map<std::pair<Index, Index>, Index> unique_edges;

        for (Index elem = 0; elem < base::elems_.cols(); ++elem)
        {
            if (base::elem_tags_[elem] != tag)
                continue;

            for (uint8_t edge = 0; edge < 3; ++edge)
            {
                std::pair<Index, Index> verts = std::make_pair(
                    base::elems_(edge, elem),
                    base::elems_((edge + 1) % 3, elem)
                    );

                if (!decoupled_edges_ && verts.first > verts.second)
                    std::swap(verts.first, verts.second);

                bool inserted = unique_edges.insert(std::make_pair(verts, num_edges)).second;

                if (inserted)
                {
                    elem_edges_(edge, elem) = num_edges;
                    elem_edge_polarities_(edge, elem) = 1;
                    edge_counts.push_back(std::make_pair(num_edges, 1));
                    edges.push_back(std::make_pair(verts, num_edges));
                    num_edges++;
                }
                else
                {
                    elem_edges_(edge, elem) = unique_edges[verts];
                    elem_edge_polarities_(edge, elem) = -1;
                    edge_counts[unique_edges[verts]].second++;
                }
            }
        }
    }

    edges_.resize(2, num_edges);
    for (const auto& edge: edges)
        edges_.col(edge.second) = EigColVecN<Index, 2> ({ edge.first.first, edge.first.second });

    edge_elems_.resize(2, num_edges);
    for (Index elem = 0; elem < elem_edges_.cols(); ++elem)
    {
        for (uint8_t edge = 0; edge < 3; ++edge)
        {
            if (elem_edge_polarities_(edge, elem) > 0)
                edge_elems_(0, elem_edges_(edge, elem)) = elem;
            else
                edge_elems_(1, elem_edges_(edge, elem)) = elem;
        }
    }

    std::vector<Index> boundary_edges, junction_edges, internal_edges;
    for (Index edge = 0; edge < num_edges; ++edge)
    {
        if (edge_counts[edge].second == 1 && !decoupled_edges_)
            boundary_edges.push_back(edge);
        else if (edge_counts[edge].second > 2 && !decoupled_edges_)
            junction_edges.push_back(edge);
        if (edge_counts[edge].second > 1 && !decoupled_edges_)
            internal_edges.push_back(edge);
    }

    boundary_edges_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        boundary_edges.data(), boundary_edges.size()
        );
    junction_edges_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        junction_edges.data(), junction_edges.size()
        );
    internal_edges_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        internal_edges.data(), internal_edges.size()
        );

    std::vector<Index> boundary_elems, junction_elems, internal_elems;
    for (Index elem = 0; elem < base::elems_.cols(); ++elem)
    {
        for (uint8_t edge = 0; edge < 3; ++edge)
        {
            if (edge_counts[elem_edges_(edge, elem)].second == 1 && !decoupled_edges_)
                boundary_elems.push_back(elem);
            else if (edge_counts[elem_edges_(edge, elem)].second > 2 && !decoupled_edges_)
                junction_elems.push_back(elem);
            if (edge_counts[elem_edges_(edge, elem)].second > 1 && !decoupled_edges_)
                internal_elems.push_back(elem);
        }
    }

    auto remove_duplicates = [] (std::vector<Index> &v)
    {
        std::sort(v.begin(), v.end());
        auto last = std::unique(v.begin(), v.end());
        v.erase(last, v.end());
        return;
    };

    remove_duplicates(boundary_elems);
    remove_duplicates(junction_elems);
    remove_duplicates(internal_elems);

    boundary_elems_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        boundary_elems.data(), boundary_elems.size()
        );
    junction_elems_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        junction_elems.data(), junction_elems.size()
        );
    internal_elems_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        internal_elems.data(), internal_elems.size()
        );

    return;

};


template class TriangleMesh<2>;
template class TriangleMesh<3>;

}
